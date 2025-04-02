from operator import index
from inlinino.instruments import Instrument
import configparser
import numpy as np
from time import sleep
from threading import Lock
import threading

class LISST200X(Instrument):
    REQUIRED_CFG_FIELDS = ['ini_file',
                           'model', 'serial_number', 'module',
                           'log_path', 'log_raw', 'log_products',
                           'variable_names', 'variable_units', 'variable_precision']


    def __init__(self, uuid, cfg, signal, *args, **kwargs):
        super().__init__(uuid, cfg, signal, setup=False, *args, **kwargs)
        # Instrument Specific attributes
        self._parser = None
        self._received_packet = False  # Used to 
        # Default serial communication parameters
        self.default_serial_baudrate = 9600
        self.default_serial_timeout = 10
        # Init Auxiliary Data widget
        self.widget_aux_data_enabled = True
        self.widget_aux_data_variables_selected = []
        self.widget_aux_data_variable_names = []
        # Init Channels to Plot widget
        self.widget_select_channel_enabled = True
        self.widget_active_timeseries_variables_names = []
        self.widget_active_timeseries_variables_selected = []
        self.active_timeseries_variables_lock = Lock()
        self.active_timeseries_variables_reset = False
        self.active_timeseries_angles = None
        # Init Spectrum Plot widget
        self.spectrum_plot_enabled = True
        self.spectrum_plot_axis_labels = dict(x_label_name='log10(theta)', x_label_units='',
                                              y_label_name='test', y_label_units='1/m/sr')
        self.spectrum_plot_trace_names = ['test']
        self.spectrum_plot_x_values = []
        
        # Setup
        self.setup(cfg)

        
        
    def setup(self, cfg, **kwargs):

        self._parser = LISST200XParser(ini_file)

        
        # Overload cfg with LISST specific parameters
        cfg['variable_names'] = ['beta']
        cfg['variable_names'].extend(self._parser.AUX_NAMES)
        cfg['variable_units'] = ['counts\tangle=' + ' '.join('%.2f' % x for x in self._parser.angles)]
        cfg['variable_units'].extend(self._parser.aux_units)
        # cfg['variable_precision'] = ['%s', '%.6f', '%.2f', '%.2f', '%.6f', '%.2f', '%.2f', "%.6f"]
        cfg['variable_precision'] = ["%s"]
        cfg['variable_precision'] += ["%.6f"] * len(self._parser.AUX_NAMES) #['%s', '%.6f', '%.2f', '%.2f', '%.6f', '%.2f', '%.2f', "%.6f"]
        cfg['terminator'] = b'L200x:>'

        
        # Set standard configuration and check cfg input
        super().setup(cfg)
        # Update logger configuration
        self._log_raw.registration = self._terminator.decode(self._parser.ENCODING, self._parser.UNICODE_HANDLING)
        self._log_raw.terminator = ''  # Remove terminator
        self._log_raw.variable_names = []  # Disable header in raw file
        # Update wavelengths for Spectrum Plot (plot is updated after the initial instrument setup or button click)
        self.spectrum_plot_x_values = [np.log10(self._parser.angles)]

        # Update Active Timeseries Variables
        self.widget_active_timeseries_variables_names = ['beta(%.5f)' % x for x in self._parser.angles]
        self.widget_active_timeseries_variables_selected = []
        self.active_timeseries_angles = np.zeros(len(self._parser.angles), dtype=bool)
        for theta in [0.08, 0.32, 1.28, 5.12]:
            channel_name = 'beta(%.5f)' % self._parser.angles[np.argmin(np.abs(self._parser.angles - theta))]
            self.update_active_timeseries_variables(channel_name, True)
        # Update Auxiliary widget
        self.widget_aux_data_variables_selected = [0, 3, 5] # laser power, temperature, battery
        self.widget_aux_data_variable_names = [self._parser.AUX_NAMES[i] + ' (' + self._parser.aux_units[i] + ')'
                                               for i in self.widget_aux_data_variables_selected]


    def parse(self, packet):
        return (self._parser.unpack_packet(packet),)

    # def handle_packet(self, packet, timestamp):
    #     self._received_packet = True
    #     super().handle_packet(packet, timestamp) # using super you keep the handle_packet from __init__?



    def handle_packet(self, packet, timestamp):

        self._received_packet = True
    
        try:
            data = self._parser.unpack_packet(packet)
    
            # If it's a battery packet, update voltage reading
            if isinstance(data, tuple) and len(data) == 2:
                scan_number, battery_voltage = data
                self.latest_battery_voltage = battery_voltage
                self.latest_scan_number = scan_number

                print(f"Battery voltage updated: {battery_voltage}V")
                
    
            else:
                # For regular data packets, pass the raw data to handle_data()
                self.handle_data(data, timestamp)
    
        except Exception as e:
            print(f"Failed to handle packet: {e}")


    def handle_data(self, raw_data, timestamp):
        raw_data = raw_data[0] if isinstance(raw_data, tuple) else raw_data
    
        # Calibrate the data ONCE here
        beta, c, aux = self._parser.calibrate(raw_data)
    
        # Append battery voltage to the last two aux columns if available
        if self.latest_battery_voltage is not None:
            if aux.shape[0] >= 2:
                aux[-2] = self.latest_battery_voltage
                aux[-1] = self.latest_scan_number
                print(f"Battery voltage added to last two aux columns: {aux[-2:]}")
            else:
                print("Warning: Aux data too short to append battery voltage.")
    
        # Prepare data for logging and display
        data = [raw_data[:35]] + aux.tolist()
    
        # Emit signals for timeseries, aux data, and spectrum data
        if self.active_timeseries_variables_lock.acquire(timeout=0.5):
            try:
                self.signal.new_ts_data[object, float, bool].emit(
                    beta[self.active_timeseries_angles], timestamp, self.active_timeseries_variables_reset
                )
                self.active_timeseries_variables_reset = False
            finally:
                self.active_timeseries_variables_lock.release()
        else:
            self.logger.error('Unable to acquire lock to update timeseries plot')
    
        self.signal.new_aux_data.emit(
            self.format_aux_data([data[i + 1] for i in self.widget_aux_data_variables_selected])
        )
        self.signal.new_spectrum_data.emit([beta])
    
        # Log raw data with updated aux (battery included)
        if self.log_prod_enabled and self._log_active:
            data[0] = np.array2string(data[0], threshold=np.inf, max_line_width=np.inf)
            self._log_prod.write(data, timestamp)
            if not self.log_raw_enabled:
                self.signal.packet_logged.emit()
    
        print(f"Final processed data (aux with battery): {aux}")

    def init_interface(self):
        try:
            # first read the zscat, config and ring area 
            
            # XZS - Sends current zscat
            # CONFIG - Prints config. record
            # XRINGA - Sends ring area array
            
            commands = [b'XZS', b'CONFIG', b'XRING']
            
            self._interface.write(b'XZS' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
            self.zsc = self._interface.read_all()
            
            print(self.zsc)


            sleep(0.1)
            
            
            commands = [b'OM 1', b'BI 6', b'SB 1', b'SI 6', b'MA 250', b'XR 3\n']
            for cmd in commands:
                self._interface.write(cmd + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
                sleep(0.1)
        
            # Flush any stale data
            self._interface._serial.reset_input_buffer()
            # self._interface.reset_input_buffer()
            sleep(0.5)
        
            # Kickstart LISST200X data flow
            
            self._interface.write(b'GX\n' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
            print("LISST200X initialized and data request sent")
        
        except Exception as e:
            print(f"Failed to initialize LISST200X: {e}")
    
    def data_received(self, data, timestamp):
        try:
            # Rebuild broken lines and clean null bytes
            self._buffer.extend(data.replace(b'\x00', b''))
    
            # Look for complete LISST200X lines
            while b'\n' in self._buffer:
                line, self._buffer = self._buffer.split(b'\n', 1)
                line = line.decode('utf-8', errors='ignore').strip()
                self.handle_packet(line, timestamp)
                if not line:
                    continue    
                try:
                    self.parse_lisst200x_data(line, timestamp)
                except ValueError as ve:
                    self.logger.error(f"Failed to parse LISST200X data line: {line} — {ve}")

        except Exception as e:
            self.logger.exception(f"Error processing LISST200X data: {e}")
    
    def parse_lisst200x_data(self, line, timestamp):
        """Parse LISST200X data line format."""
        try:
            if ',' in line:
                values = line.split(',')
                # Emit signal or store parsed data
            else:
                print(f"Unrecognized LISST200X line: {line}")
    
        except Exception as e:
            print(f"Failed to parse LISST200X data: {e}")
            
            values = line.split(',')


    @staticmethod
    def format_aux_data(data):
        """Formats auxiliary data for display, handling numeric and string values."""
        formatted_data = []
        for v in data:
            try:
                # Try to format as a float with 2 decimal places
                formatted_data.append('%.2f' % float(v))
            except ValueError:
                # If it fails (e.g., "11.55V"), keep it as a string
                formatted_data.append(str(v).strip())
        return formatted_data

    def update_active_timeseries_variables(self, name, state):
        if not ((state and name not in self.widget_active_timeseries_variables_selected) or
                (not state and name in self.widget_active_timeseries_variables_selected)):
            return
        if self.active_timeseries_variables_lock.acquire(timeout=0.25):
            self.active_timeseries_variables_reset = True
            try:
                index = self.widget_active_timeseries_variables_names.index(name)
                self.active_timeseries_angles[index] = state
            finally:
                self.active_timeseries_variables_lock.release()
        else:
            self.logger.error('Unable to acquire lock to update active timeseries variables')
        # Update list of active variables for GUI keeping the order
        self.widget_active_timeseries_variables_selected = \
            ['beta(%.5f)' % theta for theta in self._parser.angles[self.active_timeseries_angles]]


class LISST200XParser():

    ENCODING = 'utf-8'
    UNICODE_HANDLING = 'replace'
    LINE_ENDING = '\r\n'
    AUX_NAMES = ['laser_power',"battery","analog_input", "laser_reference", "depth", 'temperature', "year", "month", "day","HH", "MM", "SS", "analog_input2", "Sautermean", "total_vol_concentration",
                 "RH", "AccX", "AccY", "AccZ", "pressure", "pressure2", "NU", "NU", "NU"]
    aux_units=['counts']*len(AUX_NAMES)
    AUX_N = len(AUX_NAMES)
    print(AUX_N)
    INDEX_YY_M = [7, 8]
    INDEX_DD_HH = [i + 1 for i in INDEX_YY_M]
    INDEX_MM_SS = [i + 1 for i in INDEX_DD_HH]
    INDEX_LASER_POWER, INDEX_LASER_REFERENCE, INDEX_TEMPERATURE = 0, 3, 5

    def __init__(self, ini_file):
        # Instrument Parameters
        self.path_length = 0.025  # Instrument path length (m)
        self.phi = 1/6           # Fraction of circle covered by detectors
        self.serial_number = 2029
        self.type = "LISST200X"
        self.vcc = 1  # Volume Conversion Constant - still used for LISST200?
        self.X = 10 # divide beta by 10 if LISST X

        # Get angles in water (in degrees and in radian)
        rho = 200 ** (1 / 35)
        dynamic_range_start = 0.05
        refractive_index_water = 1.3308
        self.angles_edges = np.logspace(0, np.log10(200), 37) * dynamic_range_start / refractive_index_water
        self.angles_edges_rad = self.angles_edges * np.pi / 180
        self.angles = np.sqrt(self.angles_edges[:-1] * self.angles_edges[1:])

        # Auxiliary calibration parameters
        ini = configparser.ConfigParser()
        ini.read(ini_file)
        instrument_key = 'Instrument' + str(self.serial_number)
        if instrument_key not in ini.sections():
            raise ValueError("Initialization file (" + ini_file + ") does not contain the instrument specified in " +
                              ini_file)
        self.aux_labels, self.aux_units, self.aux_scales, self.aux_offs = [], [], [], []

        for i in range(0, len(['laser_power', 'battery', 'ext_instr', 'laser_reference', 'depth', 'temperature'])):
           self.aux_labels.append(ini[instrument_key]['HK' + str(i) + 'Label'])
           self.aux_units.append(ini[instrument_key]['HK' + str(i) + 'Units'])
           self.aux_scales.append(float(ini[instrument_key]['HK' + str(i) + 'Scale']))
           self.aux_offs.append(float(ini[instrument_key]['HK' + str(i) + 'Off']))
        self.aux_scales = np.asarray(self.aux_scales)
        self.aux_offs = np.asarray(self.aux_offs)
        # Special Aux for day and time
        self.aux_labels.append('Day')
        self.aux_units.append('decimal day')

       
    def unpack_packet(self, packet):
        try:
            # Try parsing as integers (regular data packet)
            data = np.asarray(packet.split(','), dtype='int')
            return data
    
        except ValueError:
            # If parsing fails, check for battery voltage format
            parts = packet.split(',')
            if len(parts) == 2 and 'V' in parts[1]:
                scan_number = int(parts[0].strip())
                battery_voltage = float(parts[1].replace('V', '').strip())
                return scan_number, battery_voltage
    
            # If it still fails, raise the error
            raise UnexpectedPacket('Unable to parse input into numpy array')
        
        if len(data) != 59:
            raise UnexpectedPacket('Incorrect number of variables in packet')
            
        print(data)
        
        return data
    
        
    def calibrate_auxiliaries(self, raw_aux):
        
        
        if len(raw_aux) != self.AUX_N-1:
            raise UnexpectedAuxiliaries('Incorrect number of auxiliary parameters')
        
        for idx, name in enumerate(['laser_power', 'battery', 'ext_instr', 'laser_reference', 'depth', 'temperature']):
            aux = self.aux_scales[idx] * np.asarray(raw_aux[name]) + self.aux_offs[idx]

        if raw_aux[self.INDEX_TEMPERATURE] > 32767:  # Needed to translate range 0:65535 to -32768:-32767 for signed int
            raw_aux[self.INDEX_TEMPERATURE] = raw_aux[self.INDEX_TEMPERATURE] - 65536
        
        decimal_day = raw_aux[self.INDEX_DD_HH] // 100 + (raw_aux[self.INDEX_DD_HH] % 100) / 24 +\
                      raw_aux[self.INDEX_MM_SS] // 100 / 1440 + (raw_aux[self.INDEX_MM_SS] % 100) / 86400
        
        return np.append(aux, decimal_day)


    def calibrate(self, raw):
        raw_beta, raw_aux = raw[:36], raw[36:]
        print(raw_beta)
        print(raw_aux)
        # Calibrate Auxiliaries
        aux = raw_aux #self.calibrate_auxiliaries(raw_aux)
        # Calibrate VSF
        # Capture change in laser reference (to adjust for drift in laser output power over time)
        tau = aux[self.INDEX_LASER_POWER] / aux[self.INDEX_LASER_REFERENCE]
        # Compute Beam C
        c = -np.log(tau) / self.path_length
        # Rescale counts for LISST type X (using X factor), correct for attenuation (using tau), and substract zsc normalized to sample reference
        beta = raw_beta / self.X / tau
        # Correct particulate scattering for detector responsivness (dcal)
        # TODO Compute VSF HERE
        # vd = invert=(beta, instrument_type=2, 0, spherical=0|non-spherical=1, 0, 0, 0)
        # vd = vd / self.vcc * self.zsc_aux[self.INDEX_LASER_REFERENCE] / aux[self.INDEX_LASER_REFERENCE]
        # Correct for attenuation within sample and detector geometry
        beta = beta / (self.path_length *
                       np.pi * self.phi * (self.angles_edges_rad[1:]**2 - self.angles_edges_rad[:36]**2))

        return beta, c, aux


# Error Management
class LISSTError(Exception):
    pass


class UnexpectedPacket(LISSTError):
    pass


class UnexpectedAuxiliaries(LISSTError):
    pass
