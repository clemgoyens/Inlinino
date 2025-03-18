from operator import index
from inlinino.instruments import Instrument
import configparser
import numpy as np
from time import sleep
from threading import Lock


class LISST200X(Instrument):
    REQUIRED_CFG_FIELDS = ['model', 'serial_number', 'module',
                           'log_path', 'log_raw', 'log_products',
                           'variable_names', 'variable_units', 'variable_precision']

    def __init__(self, uuid, cfg, signal, *args, **kwargs):
        super().__init__(uuid, cfg, signal, setup=False, *args, **kwargs)
        # Instrument Specific attributes
        self._parser = None
        self._received_packet = False  # Used to query new measurement
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

        self._parser = LISST200XParser()

        # Overload cfg with LISST specific parameters
        cfg['variable_names'] = ['beta']
        cfg['variable_names'].extend(self._parser.AUX_NAMES)
        cfg['variable_units'] = ['counts\tangle=' + ' '.join('%.2f' % x for x in self._parser.angles)]
        cfg['variable_units'].extend(self._parser.aux_units)
        cfg['variable_precision'] = ['%s', '%.6f', '%.2f', '%.2f', '%.6f', '%.2f', '%.2f', "%.6f"]
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
        self.widget_aux_data_variables_selected = [0, 3, 5] # variables selected out the list from AUX not all!
        self.widget_aux_data_variable_names = [self._parser.AUX_NAMES[i] + ' (' + self._parser.aux_units[i] + ')'
                                               for i in self.widget_aux_data_variables_selected]


    def parse(self, packet):
        return (self._parser.unpack_packet(packet),)

    def handle_packet(self, packet, timestamp):
        self._received_packet = True
        super().handle_packet(packet, timestamp)


    def handle_data(self, raw, timestamp):
        raw = raw[0]  # data is numpy array passed as tuple to go through handle_packet of generic module
        # Apply calibration
        beta, c, aux = self._parser.calibrate(raw)
        data = [raw[:35]] + aux.tolist()  # Write uncalibrated beta and calibrated auxiliaries
        # Update plots
        print(data)
        if self.active_timeseries_variables_lock.acquire(timeout=0.5):
            try:
                self.signal.new_ts_data[object, float, bool].emit(beta[self.active_timeseries_angles], timestamp,
                                                                  self.active_timeseries_variables_reset)
                self.active_timeseries_variables_reset = False  # Reset here as potentially set by update_active_timeseries_variables
            finally:
                self.active_timeseries_variables_lock.release()
        else:
            self.logger.error('Unable to acquire lock to update timeseries plot')
        self.signal.new_aux_data.emit(self.format_aux_data([data[i+1] for i in self.widget_aux_data_variables_selected]))
        self.signal.new_spectrum_data.emit([beta])
        # Log raw beta and calibrated aux
        if self.log_prod_enabled and self._log_active:
            # np arrays must be pre-formated to be written
            data[0] = np.array2string(data[0], threshold=np.inf, max_line_width=np.inf)
            self._log_prod.write(data, timestamp)
            if not self.log_raw_enabled:
                self.signal.packet_logged.emit()

    def init_interface(self):
        # Configuration commands
        self._interface.write(b'OM 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'BI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'SB 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'SI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'MA 250' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'XR 3' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        sleep(0.1)
        
        # TODO Check if OM, BI, SB, SI commands are necessary
        # TODO Check if configuration is correct
        #self._interface.write(b'OM 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Operating Mode: Real Time
        #self._interface.write(b'BI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Seconds between Bursts: 6
        #self._interface.write(b'SB 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Samples per Burst: 1
        #self._interface.write(b'SI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Seconds between Samples: 6
        #self._interface.write(b'MA 250' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))  # Measurements per Average: 250
        #sleep(0.1)
        response = self._interface.read()
        print('this is response')
        print(response)
        # Query first data
        self._interface.write(b'GX' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'GX' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))


    def write_to_interface(self):
        if self._received_packet:
            self._received_packet = False
            self._interface.write(b'GX' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))

    @staticmethod
    def format_aux_data(data):
        return ['%.2f' % v for v in data]

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
    AUX_NAMES = ['laser_power', 'laser_reference', 'depth', 'temperature',
                 'timestamp','battery', 'ext_instr']
    AUX_N = len(AUX_NAMES)-1
    aux_units=['counts', 'counts', 'counts', 'counts', '-', '-', '-']
    INDEX_YY_M = [42, 43]
    INDEX_DD_HH = [i + 1 for i in INDEX_YY_M]
    INDEX_MM_SS = [i + 1 for i in INDEX_DD_HH]
    INDEX_LASER_POWER, INDEX_LASER_REFERENCE, INDEX_TEMPERATURE = 36, 39, 41

    def __init__(self):
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
        self.angles_edges = np.logspace(0, np.log10(200), 36) * dynamic_range_start / refractive_index_water
        self.angles_edges_rad = self.angles_edges * np.pi / 180
        self.angles = np.sqrt(self.angles_edges[:-1] * self.angles_edges[1:])

        # Auxiliary calibration parameters
        #ini = configparser.ConfigParser()
        #instrument_key = 'Instrument' + str(self.serial_number)

        #self.aux_labels, self.aux_units, self.aux_scales, self.aux_offs = [], [], [], []
        #self.off = []
        #for i in range(self.AUX_N):
        #    self.aux_labels.append(ini[instrument_key]['HK' + str(i) + 'Label'])
        #    self.aux_units.append(ini[instrument_key]['HK' + str(i) + 'Units'])
        #    self.aux_scales.append(float(ini[instrument_key]['HK' + str(i) + 'Scale']))
        #    self.aux_offs.append(float(ini[instrument_key]['HK' + str(i) + 'Off']))
        #self.aux_scales = np.asarray(self.aux_scales)
        #self.aux_offs = np.asarray(self.aux_offs)
        ## Special Aux for day and time
        #self.aux_labels.append('Day')
        #self.aux_units.append('decimal day')

       
    def unpack_packet(self, packet):
        try:
            packet = packet.decode(self.ENCODING, self.UNICODE_HANDLING)
            data = np.asarray(packet[packet.find('{')+2:packet.find('}')-1].split(self.LINE_ENDING), dtype='int')
            print(data)
        except Exception:
            raise UnexpectedPacket('Unable to parse input into numpy array')
        if len(data) != 59:
            raise UnexpectedPacket('Incorrect number of variables in packet')
        return data


    def calibrate(self, raw):
        raw_beta, raw_aux = raw[:36], raw[36:]
        # Calibrate Auxiliaries
        aux = self.calibrate_auxiliaries(raw_aux)
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
