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
        self.serial_port = cfg["serial_port"]
        self.baudrate = cfg["baudrate"]

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

        # Init Spectrum Plot widget
        self.spectrum_plot_enabled = True
        self.spectrum_plot_axis_labels = dict(y_label_name='c', y_label_units='-')
        self.spectrum_plot_trace_names = ['c']
        self.spectrum_plot_x_values = []

        # Setup
        self.setup(cfg)
        self._parser = LISST200XParser()

    def setup(self, cfg):
        # Overload cfg with LISST specific parameters
        cfg['variable_names'] = ['c']
        cfg['variable_units'] = ['-']
        cfg['variable_precision'] = ["%.6f"]
        cfg['terminator'] = b'L200x:>'

        # Set standard configuration and check cfg input
        super().setup(cfg)

        # Update logger configuration
        self.widget_active_timeseries_variables_names = ['c']
        self.update_active_timeseries_variables('c', True)

    # def parse(self, packet):
    #     return (self._parser.unpack_packet(packet),)

    def handle_packet(self, packet, timestamp):
        self._received_packet = True
        super().handle_packet(packet, timestamp)

    def handle_data(self, raw, timestamp):
        raw = raw[0]
        c, aux = self._parser.calibrate(raw)

        if self.active_timeseries_variables_lock.acquire(timeout=0.5):
            try:
                self.signal.new_ts_data.emit(c, timestamp)
            finally:
                self.active_timeseries_variables_lock.release()
        else:
            self.logger.error('Unable to acquire lock to update timeseries plot')

        self.signal.new_aux_data.emit(
            self.format_aux_data([aux[i + 1] for i in self.widget_aux_data_variables_selected]))
        self.signal.new_spectrum_data.emit([c])

        if self.log_prod_enabled and self._log_active:
            self._log_prod.write(data, timestamp)
            if not self.log_raw_enabled:
                self.signal.packet_logged.emit()

    def init_interface(self):
        self._interface.open(self.serial_port, self.baudrate, timeout=2)
        self._interface.write(b'\x03')  # Ctrl-C
        sleep(0.5)
        self._interface.write(b'\x03')  # Ctrl-C again
        sleep(1)

        # Configuration commands
        self._interface.write(b'OM 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'BI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'SB 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'SI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'MA 250' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        self._interface.write(b'XR 3' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        sleep(0.1)

        # Query first data
        self._interface.write(b'GO' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))

        while True:
            raw_line = self.read_line_from_serial(self._interface)
            if raw_line:
                # Check if the line is numeric (contains digits, `.` or `-`)
                if all(c.isdigit() or c in ',.-' for c in raw_line.replace(',', '')):
                    # Process numeric data
                    data = list(map(float, raw_line.split(',')))
                    print(f"Processed data: {data}")
                else:
                    print("Ignored non-numeric line")

    @staticmethod
    def read_line_from_serial(serial_interface):
        line = bytearray()
        while True:
            char = serial_interface.read(1)
            if char == b'\n':
                break
            line.extend(char)
        return line.decode('utf-8').strip()

    def write_to_interface(self):
        if self._received_packet:
            self._received_packet = False
            self._interface.write(b'GX' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))

    @staticmethod
    def format_aux_data(data):
        return ['%.2f' % v for v in data]

    def update_active_timeseries_variables(self, name, state):
        self.widget_active_timeseries_variables_selected = ()


class LISST200XParser:
    ENCODING = 'utf-8'
    UNICODE_HANDLING = 'replace'
    LINE_ENDING = '\r\n'
    AUX_NAMES = ['laser_power', 'laser_reference', 'depth', 'temperature',
                 'timestamp']  # ['laser_power', 'battery', 'ext_instr', 'laser_reference', 'depth', 'temperature', 'timestamp']
    INDEX_YY_M = [42, 43]
    INDEX_DD_HH = [i + 1 for i in INDEX_YY_M]
    INDEX_MM_SS = [i + 1 for i in INDEX_DD_HH]
    INDEX_LASER_POWER, INDEX_LASER_REFERENCE, INDEX_TEMPERATURE = 36, 39, 41

    def __init__(self):
        # Instrument Parameters
        self.path_length = 0.025  # Instrument path length (m)
        # Get angles in water (in degrees and in radian)

    # def unpack_packet(self, packet):
    #     try:
    #         packet = packet.decode(self.ENCODING, self.UNICODE_HANDLING)
    #         data = np.asarray(packet[packet.find('{')+2:packet.find('}')-1].split(self.LINE_ENDING), dtype='int')
    #         print(data)
    #     except Exception:
    #         raise UnexpectedPacket('Unable to parse input into numpy array')
    #     if len(data) != 59:
    #         raise UnexpectedPacket('Incorrect number of variables in packet')
    #     return data

    def calibrate(self, raw):
        # Calibrate Auxiliaries
        if raw[self.INDEX_TEMPERATURE] > 32767:  # Needed to translate range 0:65535 to -32768:-32767 for signed int
            raw[self.INDEX_TEMPERATURE] = raw[self.INDEX_TEMPERATURE] - 65536
        decimal_day = raw[self.INDEX_DD_HH[0]] + (raw[self.INDEX_DD_HH[1]]) / 24 + \
                      raw[self.INDEX_MM_SS[0]] / 1440 + (raw[self.INDEX_MM_SS][1]) / 86400
        tau = raw[self.INDEX_LASER_POWER] / raw[self.INDEX_LASER_REFERENCE]
        c = -np.log(tau) / self.path_length
        return c, raw  # beta, c, aux


# Error Management
class LISSTError(Exception):
    pass


class UnexpectedPacket(LISSTError):
    pass


class UnexpectedAuxiliaries(LISSTError):
    pass
