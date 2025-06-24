import os
import subprocess
import time
from typing import Tuple

import numpy as np


def uint_to_float(x_int, x_min, x_max, bits):
    """Converts unsigned int to float, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    return (x_int * span / ((1 << bits) - 1)) + offset


def float_to_uint(x, x_min, x_max, bits):
    """Converts a float to an unsigned int, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    x = min(x, x_max)
    x = max(x, x_min)
    return int((x - offset) * ((1 << bits) - 1) / span)


def split_int16_to_uint8(data: int) -> Tuple[int, int]:
    """Split a signed 16-bit integer into two unsigned 8-bit integers.

    Args:
        data (int): The 16-bit integer to split.

    Returns:
        Tuple[int, int]: The high and low bytes as two 8-bit unsigned integers.
    """
    # Ensure the data is within the int16 range
    data = max(min(data, 32767), -32768)
    data_int16 = np.int16(data)

    # Split the int16 into two uint8s
    low_byte = data_int16 & 0xFF
    high_byte = (data_int16 >> 8) & 0xFF
    return high_byte, low_byte


class RateRecorder:
    def __init__(self, report_interval=10):
        """
        Initialize the rate recorder.
        :param report_interval: Interval in seconds at which the rate should be reported.
        """
        self.report_interval = report_interval
        self.start_time = None
        self.last_report_time = None
        self.iteration_count = 0
        self.message = ""

    def __enter__(self):
        return self.start()

    def start(self):
        # Record the start time and initialize variables when the context manager is entered
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.iteration_count = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Final report when exiting the context manager
        self._report_rate()

    def _report_rate(self):
        # Calculate and print the rate of iterations per second
        elapsed_time = time.time() - self.start_time
        rate = self.iteration_count / elapsed_time if elapsed_time > 0 else 0
        print(
            f"Total rate: {rate:.2f} iterations per second over {elapsed_time:.2f} seconds. Used message: {self.message}"
        )

    def track(self, message=""):
        """
        This method should be called once every loop iteration. It tracks and reports the rate
        every `report_interval` seconds.
        """
        self.iteration_count += 1
        current_time = time.time()
        self.message = message
        # Report the rate every `self.report_interval` seconds
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            self._report_rate()
            # reset the iteration count
            self.iteration_count = 0
            self.start_time = time.time()


def is_can_up(can_name):
    # Check CAN interface status
    status = os.system(f"ip link show {can_name} | grep UP > /dev/null 2>&1") == 0
    print(f"can {can_name} is up: {status}")
    return status


def setup_can_interface(can_name, bitrate):
    try:
        # Set CAN interface type and bitrate
        subprocess.run(["sudo", "ip", "link", "set", can_name, "type", "can", "bitrate", str(bitrate)], check=True)
        # Start CAN interface
        subprocess.run(["sudo", "ip", "link", "set", "up", can_name], check=True)
        print(f"{can_name} is set up with bitrate {bitrate} and turned up.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up {can_name}: {e}")
        return False