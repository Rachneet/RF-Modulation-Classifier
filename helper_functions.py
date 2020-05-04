import os
import sys
import struct
import numpy as np
import logging
import warnings
import copy


def read_binary(file_path, num_samples=-1):
    """
    reads a number of IQ samples from a binary file.
    Assumed format of binary file: 32 bit floats.
    If num_samples == -1 all samples from the files are read.
    returns: np.array(size=num_samples, dtype=complex64)
    """

    if (not os.path.isfile(file_path)):
        critical('sys', "File %s does not exist." % file_path)
        sys.exit()

    statinfo = os.stat(file_path)
    size = statinfo.st_size
    total_samples = int(size / 8)  # each samples = 4 byte float (I) + 4 byte float (Q)
    if(num_samples == -1):
        num_samples = total_samples

    #debug("OS", "Found %s with %r bytes" % (file_path, size))

    # some checks
    assert (num_samples % 10) == 0,\
        "make num_samples a multiple of 10"
    assert (num_samples <= total_samples),\
        "trying to read %d samples but file contains only %d samples" % (num_samples, total_samples)

    batch_size = int(num_samples / 10)  # load files in 10% blocks

    # pre-allocate memory
    data = np.array([0 + 0 * 1j] * num_samples, dtype=np.complex64)

    with open(file_path, "rb") as binary_file:

        for i in np.arange(0, num_samples, batch_size):
            binary_data = binary_file.read(8 * batch_size)
            unpacked_data = np.array(struct.unpack('f' * batch_size * 2, binary_data))
            data[i:i + batch_size].real = unpacked_data[::2]
            data[i:i + batch_size].imag = unpacked_data[1::2]

    return data


# Scripts

def SINR_calibration(transmitter, receiver, interferer, target_SNR, target_SIR, tolerance=0.1, iters=10, t=0.1):
    """Calibrates for a given SNR, SIR. Assumes transmitter data to be loaded"""

    isdebug = (logging.getLogger('toplevel').streamlevel == logging.DEBUG)
    debug("sys", "Calibrating SNR to %5.2f dB" % target_SNR)
    delta_SNR = 0
    for i in range(iters):
        unclipped = transmitter.change_gain(delta_SNR)
        transmitter.transmit_samples_from_memory()
        receiver.receive_samples(t * receiver.srate, 'cache.bin', verbose=isdebug)
        if isdebug:
            receiver.visualize_samples('unfiltered')
        delta_SNR = target_SNR - receiver.measure_SNR()
        debug(receiver.name, "Delta_SNR: %f dB" % delta_SNR)
        if (not unclipped or abs(delta_SNR) <= tolerance):
            break

    transmitter.stop()

    if (i == iters - 1):
        error("sys", "Target SNR not met after %d iterations! SNR: %5.3fdB" % (i + 1, target_SNR - delta_SNR))

    if ((interferer is None) or (target_SIR is None)):
        info("SINR", "SNR: %5.3f, TX gain: %5.3f after %d of %d iterations"
            % (target_SNR - delta_SNR, transmitter.gain, i+1, iters))
        return delta_SNR
    else:

        debug("sys", "Calibrating SIR to %5.2f dB" % target_SIR)
        delta_SIR = 0
        for k in range(iters):
            unclipped = interferer.change_gain(delta_SIR)
            interferer.transmit_samples_from_memory()  # transmit samples
            receiver.receive_samples(t * receiver.srate, 'cache.bin', verbose=False)
            delta_SIR = -(target_SIR - receiver.measure_SIR())
            debug(receiver.name, "Delta_SIR: %f dB" % delta_SIR)
            if (not unclipped or abs(delta_SIR) <= tolerance):
                break

        interferer.stop()
        if (k == iters - 1):
            error("sys", "Target SIR not met after %d iterations! SIR: %5.3fdB" % (k + 1, target_SIR + delta_SIR))

        info("SINR", "SNR: %5.3f, TX gain: %5.3f after %d of %d iterations"
            % (target_SNR - delta_SNR, transmitter.gain, i + 1, iters))
        info("SINR", "SIR: %5.3f, TX gain: %5.3f after %d of %d iterations"
            % (target_SIR + delta_SIR, interferer.gain, k + 1, iters))
        return delta_SNR, delta_SIR


def cfo_correction(transmitter, receiver, offset=0, iters=5, tolerance=0, t=0.01):

    debug('CFO', "Correct CFO between %s and %s" % (transmitter.name, receiver.name))
    isdebug = (logging.getLogger('toplevel').streamlevel == logging.DEBUG)

    for i in range(iters):
        transmitter.cfo_measurement(offset)
        cfo = receiver.get_cfo(120e3, plot=isdebug)
        transmitter.correct_cfo(cfo)

        if(abs(cfo - offset) <= tolerance):
            break

    info(transmitter.name, "Residual CFO: %d Hz after %d of %d iterations" % (cfo - offset, i + 1, iters))
    transmitter.stop()

    if(isdebug):
        visualize_BPSK_fullframe(transmitter, receiver, t)
    return cfo  # last corrected cfo


def visualize_BPSK_fullframe(transmitter, receiver, t):

    info('CFO', "Showing BPSK envelope for CFO analysis over %5.3f miliseconds" % (t * 1e3))
    transmitter.load_samples("SC_BPSK_4_0.35.bin")  # load samples
    transmitter.change_gain(transmitter.init_gain - transmitter.gain)
    transmitter.transmit_samples_from_memory(40e6)  # transmit samples
    receiver.receive_and_filter(t * receiver.srate)
    receiver.visualize_BPSK_fullframe()
    transmitter.stop()


# Logging

class deviceFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        self.known_devices = ["sys"]
        logging.Formatter.__init__(self, *args, **kwargs)

    def get_index(self, device_name):

        if (device_name not in self.known_devices):
            self.known_devices.append(device_name)
        index = self.known_devices.index(device_name)
        if (index > 13):
            warnings.warn("Too many devices: color coded logging might be ambigious", RuntimeWarning)
        return index

    def format(self, record):

        record = copy.copy(record)

        index = self.get_index(record.device_name) if hasattr(record, 'device_name') else 0
        device_name = self.known_devices[index]
        message = record.getMessage()
        level_short = {
            'DEBUG': '\033[42m DEBG \033[0m',
            'INFO': '\033[44m INFO \033[0m',
            'WARNING': '\033[43m WARN \033[0m',
            'ERROR': '\033[41m ERR  \033[0m',
            'CRITICAL': '\033[101m CRIT \033[0m',
        }
        record.msg = level_short[record.levelname] + '\t'
        if (index == 0):
            message = "\033[1m" + message + "\033[0m"  # 'sys' msgs in boldface
        record.msg += "\033[38;5;%dm\033[1m[%s]\033[0m\t%s" % \
            (index, device_name, message)

        return logging.Formatter.format(self, record)


def info(name, msg):
    msg = clean_message(msg)
    if (msg):
        logger = logging.getLogger('toplevel')
        logger.info(msg, extra={'device_name': name})


def debug(name, msg):
    msg = clean_message(msg)
    if (msg):
        logger = logging.getLogger('toplevel')
        logger.debug(msg, extra={'device_name': name})


def warning(name, msg):
    msg = clean_message(msg)
    if (msg):
        logger = logging.getLogger('toplevel')
        logger.warning(msg, extra={'device_name': name})


def error(name, msg):
    msg = clean_message(msg)
    if (msg):
        logger = logging.getLogger('toplevel')
        logger.error(msg, extra={'device_name': name})


def critical(name, msg):
    msg = clean_message(msg)
    if (msg):
        logger = logging.getLogger('toplevel')
        logger.critical(msg, extra={'device_name': name})


def clean_message(msg):
    msg = msg.rstrip('\r\n')
    if (msg != ""):
        return msg
    return False
