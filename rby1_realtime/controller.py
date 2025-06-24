"""
RBY1 Hardware Interface - abstracts between simulation and real robot control.
"""

import time
import numpy as np
import viser
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Literal
from dataclasses import dataclass
from loguru import logger

try:
    from robot_descriptions.loaders.mujoco import load_robot_description as load_mujoco_robot_description
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    logger.error("ImportError: robot_descriptions not found")
    logger.info("Install with: pip install git+https://github.com/uynitsuj/robot_descriptions.py.git")
    logger.info("Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()

try:
    import mink
    HAS_MINK = True
except ImportError:
    HAS_MINK = False
    logger.warning("mink not available. Mink-based IK will be disabled.")

try:
    import pyroki as pk
    HAS_PYROKI = True
except ImportError:
    HAS_PYROKI = False
    logger.warning("pyroki not available. Pyroki-based IK will be disabled.")

from rby1_realtime.inverse_kinematics.rby1_mink import RBY1MinkInterface
from rby1_realtime.inverse_kinematics.rby1_pyroki import RBY1PyrokiInterface


@dataclass
class RobotState:
    """Robot state data structure."""
    joint_positions: np.ndarray
    gripper_positions: np.ndarray
    ee_poses: Dict[str, np.ndarray]  # 4x4 transformation matrices
    timestamp: float


@dataclass 
class RobotCommand:
    """Robot command data structure."""
    joint_positions: Optional[np.ndarray] = None
    gripper_positions: Optional[np.ndarray] = None
    gripper_torques: Optional[np.ndarray] = None


class HardwareInterface(ABC):
    """Abstract hardware interface for RBY1 robot."""
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state."""
        pass
        
    @abstractmethod
    def send_command(self, command: RobotCommand) -> bool:
        """Send command to robot. Returns True if successful."""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        pass
        
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Emergency stop the robot."""
        pass


class SimulationInterface(HardwareInterface):
    """Simulation hardware interface."""
    
    def __init__(self):
        self.joint_positions = np.zeros(28)  # RBY1 has 28 joints
        self.gripper_positions = np.array([0.0, 0.0])  # [right, left]
        self.connected = True
        
    def get_state(self) -> RobotState:
        """Get current simulated robot state."""
        # In simulation, we return the current commanded state
        return RobotState(
            joint_positions=self.joint_positions.copy(),
            gripper_positions=self.gripper_positions.copy(),
            ee_poses={},  # Would need forward kinematics to compute
            timestamp=time.time()
        )
        
    def send_command(self, command: RobotCommand) -> bool:
        """Send command to simulation."""
        if command.joint_positions is not None:
            self.joint_positions = command.joint_positions.copy()
        if command.gripper_positions is not None:
            self.gripper_positions = command.gripper_positions.copy()
        return True
        
    def is_connected(self) -> bool:
        return self.connected
        
    def emergency_stop(self) -> bool:
        return True


class RealRobotInterface(HardwareInterface):
    """Real robot hardware interface."""
    
    def __init__(self, address: str = "192.168.30.1:50051"):
        self.address = address
        self.robot = None
        self._connect()
        
    def _connect(self):
        """Connect to real robot."""
        # try:
        from rby1_realtime.robot.rby1 import RBY1
        self.robot = RBY1(address=self.address)
        logger.info(f"Connected to RBY1 at {self.address}")
        # except Exception as e:
        #     logger.error(f"Failed to connect to robot: {e}")
        #     self.robot = None
            
    def get_state(self) -> RobotState:
        """Get current real robot state."""
        if not self.is_connected():
            raise RuntimeError("Robot not connected")
            
        joint_pos = self.robot.get_joint_pos()
        gripper_pos = self.robot.get_gripper_pos()
        
        return RobotState(
            joint_positions=joint_pos.copy(),
            gripper_positions=gripper_pos.copy(),
            ee_poses={},  # Would need forward kinematics impl
            timestamp=time.time() # TODO: Should get this from robot hardware if possible
        )
        
    def send_command(self, command: RobotCommand) -> bool:
        """Send command to real robot."""
        if not self.is_connected():
            logger.warning("Robot not connected")
            return False
            
        try:
            if command.joint_positions is not None:
                self.robot.command_joint_pos(command.joint_positions)
                
            if command.gripper_positions is not None:
                torques = command.gripper_torques if command.gripper_torques is not None else [0.1, 0.1]
                self.robot.set_gripper_pos(command.gripper_positions, torques)
                
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
            
    def is_connected(self) -> bool:
        return self.robot is not None
        
    def emergency_stop(self) -> bool:
        """Emergency stop the robot."""
        if not self.is_connected():
            return False
        try:
            # Implement emergency stop logic
            # This would depend on the specific robot API
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False


class RBY1HardwareInterface:
    """
    RBY1 hardware interface.
    """
    
    def __init__(
        self, 
        use_real_robot: bool = False,
        robot_address: str = "192.168.30.1:50051",
        ik_solver: Literal["mink", "pyroki"] = "mink",
        rate: float = 100.0,
        joint_filtering_alpha: float = 0.99,
        viser_server: Optional[viser.ViserServer] = None,
        **ik_kwargs
    ):
        """
        Initialize RBY1 hardware interface.
        
        Args:
            use_real_robot: Whether to use real robot or simulation
            robot_address: Address of real robot
            ik_solver: IK solver to use ("mink" or "pyroki")
            rate: Control rate in Hz
            joint_filtering_alpha: Joint position filter for real robot
            viser_server: Optional viser server instance (will create if None)
            **ik_kwargs: Additional arguments for IK solver
        """
        self.use_real_robot = use_real_robot
        self.joint_filtering_alpha = joint_filtering_alpha
        self.rate = rate
        
        # Validate solver availability
        if ik_solver == "mink" and not HAS_MINK:
            raise ImportError("Mink not available but requested as IK solver")
        if ik_solver == "pyroki" and not HAS_PYROKI:
            raise ImportError("Pyroki not available but requested as IK solver")
            
        # Initialize hardware interface
        if use_real_robot:
            self.hardware = RealRobotInterface(robot_address)
        else:
            self.hardware = SimulationInterface()
            
        # Initialize IK solver
        self._setup_ik_solver(ik_solver, rate, viser_server, **ik_kwargs)
        
        # Track previous joint positions for filtering
        self._previous_joints = None
        
    def _setup_ik_solver(self, ik_solver: str, rate: float, viser_server: Optional[viser.ViserServer], **ik_kwargs):
        """Setup the IK solver instance."""
        # Create viser server if not provided
        if viser_server is None:
            self.viser_server = viser.ViserServer()
        else:
            self.viser_server = viser_server
        
        if ik_solver == "mink":
            # Pass viser server to mink interface
            self.ik_interface = RBY1MinkInterface(rate=rate, viser_server=self.viser_server, **ik_kwargs)
            
        elif ik_solver == "pyroki":
            # Pass viser server to pyroki interface
            self.ik_interface = RBY1PyrokiInterface(viser_server=self.viser_server)
            
        else:
            raise ValueError(f"Unknown IK solver: {ik_solver}")
            
        self.ik_solver_name = ik_solver
        
    def solve_ik_and_update_hardware(self):
        """Solve IK and send commands to hardware."""
        # Solve IK using the dedicated solver
        self.ik_interface.solve_ik()
        
        # Get joint solution
        joints = self.ik_interface.joints
        if joints is None:
            return
            
        # Apply joint filtering for real robot
        if self.use_real_robot and self._previous_joints is not None:
            filtered_joints = (
                self.joint_filtering_alpha * joints + 
                (1 - self.joint_filtering_alpha) * self._previous_joints
            )
        else:
            filtered_joints = joints
            
        self._previous_joints = filtered_joints
        
        # Send command to hardware
        try:
            command = RobotCommand(
                joint_positions=filtered_joints,
                gripper_positions=np.array([0.0, 0.0]),  # Default closed
                gripper_torques=np.array([0.1, 0.1])
            )
            
            success = self.hardware.send_command(command)
            if not success:
                logger.warning("Failed to send command to hardware")
                
        except Exception as e:
            logger.error(f"Hardware update failed: {e}")
            
    def update_visualization(self):
        """Update visualization."""
        self.ik_interface.update_visualization()
        
    def home(self):
        """Reset robot to home position."""
        self.ik_interface.home()
        self._previous_joints = None
        
    def emergency_stop(self):
        """Emergency stop the robot."""
        return self.hardware.emergency_stop()
        
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware status information."""
        return {
            "connected": self.hardware.is_connected(),
            "use_real_robot": self.use_real_robot,
            "ik_solver": self.ik_solver_name,
            "rate": self.rate,
        }
        
    def get_robot_state(self) -> RobotState:
        """Get current robot state from hardware."""
        return self.hardware.get_state()
        
    def run(self):
        """Main control loop."""
        logger.info(f"Starting RBY1 hardware interface with {self.ik_solver_name} solver")
        logger.info(f"Hardware status: {self.get_hardware_status()}")
        
        try:
            while True:
                start_time = time.time()
                
                # Solve IK and update hardware
                self.solve_ik_and_update_hardware()
                
                # Update visualization
                self.update_visualization()
                
                # Update timing in IK interface
                if hasattr(self.ik_interface, 'timing_handle'):
                    elapsed_time = time.time() - start_time
                    self.ik_interface.timing_handle.value = (
                        0.99 * self.ik_interface.timing_handle.value + 
                        0.01 * (elapsed_time * 1000)
                    )
                    
                # Control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0/self.rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise


def main():
    """Main function for testing the hardware interface."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_robot", action="store_true", default=True, help="Use real robot instead of simulation")
    parser.add_argument("--robot_address", type=str, default="192.168.30.1:50051", help="Robot address")
    parser.add_argument("--ik_solver", type=str, choices=["mink", "pyroki"], default="mink", help="IK solver to use")
    parser.add_argument("--rate", type=float, default=100.0, help="Control rate in Hz")
    parser.add_argument("--head_task", action="store_true", default=True, help="Enable head task (mink only)")
    parser.add_argument("--torso_task", action="store_true", default=True, help="Enable torso task (mink only)")
    args = parser.parse_args()
    
    # Prepare IK kwargs
    ik_kwargs = {}
    if args.ik_solver == "mink":
        ik_kwargs["head_task_active"] = args.head_task
        ik_kwargs["torso_task_active"] = args.torso_task
    
    # try:
    rby1 = RBY1HardwareInterface(
        use_real_robot=args.real_robot,
        robot_address=args.robot_address,
        ik_solver=args.ik_solver,
        rate=args.rate,
        **ik_kwargs
    )
    
    rby1.run()
        
    # except KeyboardInterrupt:
    #     logger.info("Shutting down...")
    # except Exception as e:
    #     logger.error(f"Error: {e}")


if __name__ == "__main__":
    main() 