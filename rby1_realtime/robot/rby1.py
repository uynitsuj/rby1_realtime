
# Note code works with rby1_sdk==0.4.1

import threading
import time
from typing import Any, Dict, List

import numpy as np
import rby1_sdk

from rby1_realtime.robot.motor_utils import RateRecorder
from rby1_realtime.robot.robot import Robot

SPEED_RATIO = 0.4

# These parameters are from https://github.com/RainbowRobotics/rby1-sdk/blob/e0362f5c71164faf1e72e333307dc7516dbcdb4a/examples/python/17_teleoperation_with_joint_mapping.py
ARM_MINIMUM_TIME = 0.3

ARM_ACC_LIMIT = np.full(7, 1200.0, dtype=np.float64)
ARM_ACC_LIMIT = np.deg2rad(ARM_ACC_LIMIT) * SPEED_RATIO

ARM_VEL_LIMIT = np.array([160, 160, 160, 160, 330, 330, 330], dtype=np.float64)
ARM_VEL_LIMIT = np.deg2rad(ARM_VEL_LIMIT) * SPEED_RATIO

CONTROL_HOLD_TIME = 4.0  # when this value is too small, the stream might expire


class RBY1(Robot):
    def __init__(
        self,
        address: str = "192.168.12.1:50051",
        model: str = "a",
        power_device: str = ".*",
        servo: str = ".*",
        fix_torso: bool = True,
        gripper_power: bool = True,
    ):
        """
        Initialize RBY1 robot, complete connection, power on, servo on, fault reset and control manager enable.
        """
        self.model = model
        if not model == "a":
            raise ValueError(f"Unsupported model: {model}")
        self.power_device = power_device
        self.servo = servo
        self.fix_torso = fix_torso
        # Create and connect robot
        self.robot = rby1_sdk.create_robot(address, model)
        self.robot.connect()
        self._command_stream = self.robot.create_command_stream()

        if not self.robot.is_connected():
            raise RuntimeError("Robot connection failed!")

        # enable robot arm
        self.enable_arm()
        if gripper_power:
            self.disable_gripper_power()
            self.enable_gripper_power()

            self.init_gripper_controller()

        # Check control manager state and try to reset if fault occurs
        control_manager_state = self.robot.get_control_manager_state()
        if control_manager_state.state in {
            rby1_sdk.ControlManagerState.State.MinorFault,
            rby1_sdk.ControlManagerState.State.MajorFault,
        }:
            if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
                print("Warning: Control Manager Major Fault detected.")
            else:
                print("Warning: Control Manager Minor Fault detected.")
            print("Attempting to reset fault...")
            if not self.robot.reset_fault_control_manager():
                raise RuntimeError("Control Manager fault reset failed!")
            print("Fault reset successful.")

        print("Control Manager status normal, no faults.")
        print("Enabling Control Manager...")
        if not self.robot.enable_control_manager():
            raise RuntimeError("Failed to enable Control Manager!")
        print("Control Manager enabled.")

        # Initialize current joint state (since SDK example does not provide joint feedback interface, store the last commanded joint values)
        self.model_info = self.robot.model()
        self.total_dofs = (
            len(self.model_info.torso_idx) + len(self.model_info.right_arm_idx) + len(self.model_info.left_arm_idx)
        )
        self.dof_indexes = self.model_info.torso_idx + self.model_info.right_arm_idx + self.model_info.left_arm_idx
        self._last_joint_state = np.zeros(self.total_dofs)
        self._rby1_state = None  # Initialize the state variable
        self._state_thread = threading.Thread(target=self._update_state, daemon=True)
        self._state_thread.start()
        while self._rby1_state is None:
            print("Waiting for RBY1 state...")
            time.sleep(1)

    def init_gripper_controller(self):
        from xdof.motors.dynamixel.driver import DynamixelDriver

        self.gripper_driver = DynamixelDriver(
            (0, 1),
            port="/dev/ttyUSB0",
            baudrate=2000000,
        )
        self.gripper_driver.set_position_with_torque_limit_mode()
        self.gripper_driver.set_torque_mode(True)

    def set_gripper_pos(self, pos: List[float], max_torque: List[float]):
        # right idx 0, fully open 3.14, fully closed 12.22
        # left idx 1, fully open 6.25, fully closed 15.27
        raw_motor_pos = np.array(pos) * 9 + np.array([3.14, 0])
        self.gripper_driver.set_joints_with_torque_limit(raw_motor_pos, max_torque)

    def get_gripper_pos(self):
        raw_motor_pos = self.gripper_driver.get_joints()
        return (raw_motor_pos - np.array([3.14, 0])) / 9

    def enable_gripper_power(self):
        if self.robot.is_power_on("48v"):
            self.robot.set_tool_flange_output_voltage("right", 12)
            self.robot.set_tool_flange_output_voltage("left", 12)
            print("Attempting to 12V power on for gripper")

    def disable_gripper_power(self):
        if self.robot.is_power_on("48v"):
            self.robot.set_tool_flange_output_voltage("right", 0)
            self.robot.set_tool_flange_output_voltage("left", 0)
            print("Attempting to 0V power off for gripper")

    def _update_state(self):
        """
        Continuously update the robot state in a separate thread.
        """
        # todo: use callback to stream.
        with RateRecorder() as recorder:
            while True:
                # print("update state")
                self._rby1_state = self.robot.get_state()
                recorder.track()
                time.sleep(0.001)  # Adjust the sleep time as needed

    def get_robot_info(self) -> Dict[str, Any]:
        """
        Get robot information.
        """
        return {
            "num_dofs": self.num_dofs(),
            "torso_dofs": len(self.model_info.torso_idx),
            "right_arm_dofs": len(self.model_info.right_arm_idx),
            "left_arm_dofs": len(self.model_info.left_arm_idx),
        }

    def enable_arm(self) -> None:
        """
        Enable robot arm (for RBY1, control manager is enabled during initialization, this is for interface compatibility).
        """
        # Power on
        if not self.robot.is_power_on(self.power_device):
            if not self.robot.power_on(self.power_device):
                raise RuntimeError("Power on failed!")

        # Enable servo
        if not self.robot.is_servo_on(self.servo):
            if not self.robot.servo_on(self.servo):
                raise RuntimeError("Servo enable failed!")

    def disable_arm(self) -> None:
        """
        Disable robot arm (if SDK supports, call corresponding interface; otherwise leave empty).
        """
        self.robot.servo_off(self.servo)
        self.robot.power_off(self.power_device)

    def num_dofs(self) -> int:
        """
        Return number of degrees of freedom (torso + right arm + left arm).
        """
        return self.total_dofs

    def get_joint_pos(self) -> np.ndarray:
        """
        Get current joint positions.
        Note: Since rby1_sdk example does not provide real-time joint feedback interface, return the last commanded joint values.
        """
        # only return torso and arm joints
        return self._rby1_state.position[self.dof_indexes]

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """
        Get current joint positions and velocities.
        """
        return {
            "joint_pos": self._rby1_state.position[self.dof_indexes],
            "joint_vel": self._rby1_state.velocity[self.dof_indexes],
            "joint_current": self._rby1_state.current[self.dof_indexes],
            "joint_torque": self._rby1_state.torque[self.dof_indexes],
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """
        Command robot to move to specified joint positions.
        Parameter joint_pos should be organized in following order: [torso joints, right arm joints, left arm joints].
        """
        total_dofs = self.num_dofs()
        if len(joint_pos) != total_dofs:
            raise ValueError(f"Joint array length mismatch. Expected {total_dofs} joints, received {len(joint_pos)}.")

        # Get joint segments based on model information
        model_info = self.robot.model()
        torso_dof = len(model_info.torso_idx)
        right_arm_dof = len(model_info.right_arm_idx)
        # left_arm_dof = len(model_info.left_arm_idx)  # Optional: for validation
        q_joint_waist = joint_pos[:torso_dof]
        q_joint_right_arm = joint_pos[torso_dof : torso_dof + right_arm_dof]
        q_joint_left_arm = joint_pos[torso_dof + right_arm_dof :]

        # If fix_torso is True, set the torso joint positions to zero
        if self.fix_torso:
            if not np.allclose(q_joint_waist, 0):
                # print("Warning: fix_torso is True but torso joints are non-zero. Setting to ready_pose.")
                q_joint_waist = np.array([0.0, 0.6108, -1.221, 0.6908, 0.0, 0.0])
                # q_joint_waist = np.array([0.0, 0.7108, -0.83,  0.7108, 0.0,  0.0])

        # Construct and send joint motion command

        rc = rby1_sdk.RobotCommandBuilder().set_command(
            rby1_sdk.ComponentBasedCommandBuilder().set_body_command(
                rby1_sdk.BodyComponentBasedCommandBuilder()
                .set_torso_command(
                    rby1_sdk.JointPositionCommandBuilder()
                    .set_minimum_time(ARM_MINIMUM_TIME)
                    .set_position(q_joint_waist)
                )
                .set_right_arm_command(
                    rby1_sdk.JointPositionCommandBuilder()
                    .set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME))
                    .set_minimum_time(ARM_MINIMUM_TIME)
                    .set_position(q_joint_right_arm)
                    .set_velocity_limit(ARM_VEL_LIMIT)
                    .set_acceleration_limit(ARM_ACC_LIMIT)
                )
                .set_left_arm_command(
                    rby1_sdk.JointPositionCommandBuilder()
                    .set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME))
                    .set_minimum_time(ARM_MINIMUM_TIME)
                    .set_position(q_joint_left_arm)
                    .set_velocity_limit(ARM_VEL_LIMIT)
                    .set_acceleration_limit(ARM_ACC_LIMIT)
                )
            )
        )
        self._command_stream.send_command(rc)

        # Update last commanded joint state
        self._last_joint_state = joint_pos.copy()

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get current robot state information.
        For RBY1, only return joint positions.
        """
        return {"joint_pos": self.get_joint_pos()}


if __name__ == "__main__":
    robot = RBY1(address="192.168.12.1:50051")
    current_joint_pos = robot.get_joint_pos()
    print(current_joint_pos)

    # def cb(rs):
    #     print("---")
    #     print(
    #         f"right ft sensor: [{rs.timestamp - rs.ft_sensor_right.time_since_last_update}] force {rs.ft_sensor_right.force}, torque {rs.ft_sensor_right.torque}")
    #     print(f"left ft sensor: [{rs.timestamp - rs.ft_sensor_left.time_since_last_update}] force {rs.ft_sensor_left.force}, torque {rs.ft_sensor_left.torque}")
    #     print(f"{rs.is_ready=}")

    # robot.robot.start_state_update(cb,
    #                         10  # (Hz)
    #                         )
    # exit()
    while True:
        # print(robot.get_joint_pos())
        print(robot._rby1_state)
        import ipdb

        ipdb.set_trace()
        time.sleep(0.001)
    exit()
    # Test joint motion
    target_joint_pos = np.zeros(robot.num_dofs())  # first 6 torso, then right arm, then left arm
    # reset to zeros
    robot.command_joint_pos(target_joint_pos)
    time.sleep(2)
    # import ipdb; ipdb.set_trace()
    joint_index = 4
    target_joint_pos[joint_index + 5] = -1.2
    target_joint_pos[joint_index + 12] = -1.2
    num_steps = 100
    for i in range(num_steps):
        interp_joint_pos = target_joint_pos * (i / num_steps)
        robot.command_joint_pos(interp_joint_pos)
        time.sleep(0.01)
