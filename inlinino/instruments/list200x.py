from operator import index

from inlinino.instruments import Instrument
import configparser
import numpy as np
from time import sleep
from threading import Lock

from parso.python.parser import INDENT


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
        self.active_timeseries_angles = None
        # Init Spectrum Plot widget
        self.spectrum_plot_enabled = True

        # Init Spectrum Plot widget
        self.spectrum_plot_enabled = True
        self.spectrum_plot_axis_labels = dict(y_label_name='tau', y_label_units='1/m/sr')
        self.spectrum_plot_trace_names = ['tau']
        self.spectrum_plot_x_values = []
        # Setup
        self.setup(cfg)

    def setup(self, cfg):
        # Overload cfg with LISST specific parameters
        cfg['variable_names'] = ['transmission']
        cfg['variable_units'] = ['-']
        cfg['variable_precision'] = ["%.6f"]
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
        self.active_timeseries_angles = np.zeros(len(self._parser.angles), dtype=bool)
        for theta in [0.08, 0.32, 1.28, 5.12]:
            channel_name = 'beta(%.5f)' % self._parser.angles[np.argmin(np.abs(self._parser.angles - theta))]
            self.update_active_timeseries_variables(channel_name, True)
        # Update Auxiliary widget
        self.widget_aux_data_variables_selected = [0, 3, 5]
        self.widget_aux_data_variable_names = [self._parser.aux_labels[i] + ' (' + self._parser.aux_units[i] + ')'
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
        # data = [raw[:32]] + raw[32:].tolist()  # Write uncalibrated data
        data = [raw[:32]] + aux.tolist()  # Write uncalibrated beta and calibrated auxiliaries
        # Update plots
        if self.active_timeseries_variables_lock.acquire(timeout=0.5):
            try:
                self.signal.new_ts_data.emit(beta[self.active_timeseries_angles], timestamp)
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
        # wake up sensor
        self._interface.write(b'\x03')  # Ctrl-C
        sleep(0.5)
        self._interface.write(b'\x03')  # Ctrl-C again
        sleep(1)
        # TODO Check if OM, BI, SB, SI commands are necessary
        # TODO Check if configuration is correct
        self._interface.write(b'OM 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Operating Mode: Real Time
        self._interface.write(b'BI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Seconds between Bursts: 6
        self._interface.write(b'SB 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Samples per Burst: 1
        self._interface.write(b'SI 6' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))    # Seconds between Samples: 6
        self._interface.write(b'MA 250' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))  # Measurements per Average: 250
        # Set XR mode for data transmission
        self._interface.write(b'XR 1' + bytes(self._parser.LINE_ENDING, self._parser.ENCODING))
        sleep(0.1)
        response = self._interface.read()
        # Query first data
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


class LISSTParser:

    ENCODING = 'utf-8'
    UNICODE_HANDLING = 'replace'
    LINE_ENDING = '\r\n'
    AUX_NAMES = ['laser_power', 'laser_reference', 'depth', 'temperature', 'timestamp'] #['laser_power', 'battery', 'ext_instr', 'laser_reference', 'depth', 'temperature', 'timestamp']
    INDEX_YY_M = [42,43]
    INDEX_DD_HH = INDEX_YY_M +1
    INDEX_MM_SS = INDEX_DD_HH + 1
    INDEX_LASER_POWER, INDEX_LASER_REFERENCE, INDEX_TEMPERATURE = 36, 39, 41

    def __init__(self):

        # Instrument Parameters
        self.path_length = 0.025  # Instrument path length (m)
        # Get angles in water (in degrees and in radian)

    def unpack_packet(self, packet):
        try:
            packet = packet.decode(self.ENCODING, self.UNICODE_HANDLING)
            data = np.asarray(packet[packet.find('{')+2:packet.find('}')-1].split(self.LINE_ENDING), dtype='int')
        except Exception:
            raise UnexpectedPacket('Unable to parse input into numpy array')
        if len(data) != 59:
            raise UnexpectedPacket('Incorrect number of variables in packet')
        return data

    def calibrate(self, raw):
        # Calibrate Auxiliaries
        if raw[self.INDEX_TEMPERATURE] > 32767:  # Needed to translate range 0:65535 to -32768:-32767 for signed int
            raw[self.INDEX_TEMPERATURE] = raw[self.INDEX_TEMPERATURE] - 65536
        decimal_day = raw[self.INDEX_DD_HH[0]]  + (raw[self.INDEX_DD_HH[1]]) / 24 +\
                      raw[self.INDEX_MM_SS[0]] / 1440 + (raw[self.INDEX_MM_SS][1]) / 86400
        tau = raw[self.INDEX_LASER_POWER] /raw[self.INDEX_LASER_REFERENCE]
        c= -np.log(tau) / self.path_length
        return c, raw #beta, c, aux


# Error Management
class LISSTError(Exception):
    pass


class UnexpectedPacket(LISSTError):
    pass


class UnexpectedAuxiliaries(LISSTError):
    pass
