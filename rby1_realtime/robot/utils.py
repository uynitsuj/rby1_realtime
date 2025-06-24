
import signal
import sys
import time
from typing import List, Optional

import numpy as np

from rby1_realtime.robot.robot import Robot


class TimeoutException(Exception):
    """Custom exception to be raised when a timeout occurs."""

    pass


TIMEOUT_INIT = False


class Timeout:
    def __init__(self, seconds, name: Optional[str] = None, mode: str = "error"):
        """
        Initialize the Timeout context manager.

        :param seconds: Timeout duration in seconds.
        :param name: Optional name for the operation.
        :param mode: Timeout mode. Either 'error' to raise an exception or 'warning' to print a warning.
        """
        self.seconds = seconds
        self.name = name
        self.mode = mode.lower()
        if self.mode not in {"error", "warning"}:
            raise ValueError("Mode must be either 'error' or 'warning'")

    def handle_timeout(self, signum, frame):
        """
        Handle the timeout event.
        """
        if self.mode == "error":
            if self.name:
                raise TimeoutException(f"Operation '{self.name}' timed out after {self.seconds} seconds")
            else:
                raise TimeoutException(f"Operation timed out after {self.seconds} seconds")
        elif self.mode == "warning":
            message = "\033[91m[WARNING]\033[0m Operation"
            if self.name:
                message += f" '{self.name}'"
            message += f" exceeded {self.seconds} seconds but continues."
            print(message, file=sys.stderr)

    def __enter__(self):
        """
        Enter the context and set the timeout alarm.
        """
        global TIMEOUT_INIT
        if not TIMEOUT_INIT:
            TIMEOUT_INIT = True
        else:
            raise NotImplementedError("Nested timeouts are not supported")
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context and clear the timeout alarm.
        """
        global TIMEOUT_INIT
        TIMEOUT_INIT = False
        signal.alarm(0)  # Disable the alarm


class Rate:
    def __init__(self, rate: Optional[float]):
        self.last = time.time()
        self.rate = rate  # when rate is None, it means we are not using rate control

    @property
    def dt(self) -> float:
        if self.rate is None:
            return 0.0
        return 1.0 / self.rate

    def sleep(self) -> None:
        if self.rate is None:
            return
        while self.last + self.dt > time.time():
            time.sleep(0.0001)
        self.last = time.time()


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def small_joint_motion_control(robot: Robot):
    steps = 100

    target_joint_poses = np.array([np.pi / 10, 0, 0, np.pi / 10, np.pi / 10, np.pi / 10, 1])
    start_time = time.time()
    for i in range(steps):
        cmd = np.zeros(7)
        cmd = easeInOutQuad(float(i) / steps) * target_joint_poses
        robot.command_joint_pos(cmd)
        time.sleep(0.01)  # Assuming a control loop time step
    print(f"current joint_pos: {robot.get_joint_pos()}")
    for i in range(steps):
        cmd = np.zeros(7)
        cmd = easeInOutQuad(1 - float(i) / steps) * target_joint_poses
        robot.command_joint_pos(cmd)
        time.sleep(0.01)  # Assuming a control loop time step

    end_time = time.time()
    loop_duration = end_time - start_time
    frequency = steps * 2 / loop_duration
    print(f"Loop frequency: {frequency:.2f} Hz")

    # Print final joint positions
    final_joint_positions = robot.get_joint_pos()
    print(f"Final joint positions: {final_joint_positions}")



def aging_motion_control(
    robot: Robot, motion_duration: float = 5.0, motion_list: List[np.ndarray] = None, save_log: bool = True
):
    cycle_count = 0
    sleep_time = 0.01
    num_steps = int(motion_duration / sleep_time)
    save_log_path = f"aging_motion_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    while True:
        for motion in motion_list:
            current_joint_pos = robot.get_joint_pos()
            for i in range(num_steps):
                interpolated_motion = current_joint_pos + i / num_steps * (motion - current_joint_pos)
                robot.command_joint_pos(interpolated_motion)
                time.sleep(sleep_time)
            # Increment cycle count
        cycle_count += 1
        timestamp = time.ctime()
        # Print the cycle count
        print(f"Cycle: {cycle_count}, Timestamp: {timestamp}")
        if save_log:
            with open(save_log_path, "a") as f:
                f.write(f"Cycle: {cycle_count}, Timestamp: {timestamp}\n")


def apply_offset_and_sign(joint_pos: np.ndarray, joint_offsets: np.ndarray, joint_signs: np.ndarray) -> np.ndarray:
    # to better match human intuition, we first adjust joint direction with signs, then apply the offset
    return joint_offsets + joint_signs * joint_pos
