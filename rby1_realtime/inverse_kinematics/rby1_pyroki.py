"""
RBY1 Pyroki Interface - uses pyroki for inverse kinematics.
"""

import time
import numpy as np
import viser.transforms as vtf

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print(
        "pip install git+https://github.com/uynitsuj/robot_descriptions.py.git"
    )
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()

try:
    import pyroki as pk
except ImportError:
    print("ImportError: pyroki not found:")
    print("pip install git+https://github.com/chungmin99/pyroki.git")
    exit()

from rby1_realtime.base import RBY1AbstractBase
from rby1_realtime.inverse_kinematics.pyroki_snippets import solve_ik_with_multiple_targets as solve_ik_with_multiple_targets


class RBY1PyrokiInterface(RBY1AbstractBase):
    """
    RBY1 interface using pyroki for inverse kinematics.
    """
    
    def __init__(self, viser_server=None):
        # Override target link names for pyroki (includes head)
        self.pyroki_target_link_names = ["ee_left", "ee_right", "link_head_0"]
        super().__init__(viser_server=viser_server)
        
    def _setup_solver_specific(self):
        """Setup pyroki-specific components."""
        # Load robot description for pyroki
        self.pyroki_robot = pk.Robot.from_urdf(self.urdf)
        
        # Initialize solution tracking
        self.solution = None
        self.rest_pose = None
        
        # Setup head control
        self._setup_head_control()
        
    def _setup_head_control(self):
        """Setup head control handle."""
        self.head_handle = self.server.scene.add_transform_controls(
            "/ik_target_head",
            scale=self.tf_size_handle.value if hasattr(self, 'tf_size_handle') else 0.2,
            position=(0.2, 0.0, 1.3),
            wxyz=(1, 0, 0, 0)
        )
        
        # Update head control when gizmo size changes
        if hasattr(self, 'tf_size_handle'):
            original_update = self.tf_size_handle.on_update._callbacks[0] if self.tf_size_handle.on_update._callbacks else None
            
            @self.tf_size_handle.on_update
            def update_tf_size_with_head(_):
                if original_update:
                    original_update(_)
                self.head_handle.scale = self.tf_size_handle.value
        
    def _initialize_transform_handles(self):
        """Initialize transform handle positions based on current robot state."""
        # Get rest pose from URDF visualization
        if hasattr(self, 'urdf_vis') and hasattr(self.urdf_vis, '_urdf'):
            self.rest_pose = self.urdf_vis._urdf.cfg
            
        # Set initial positions for end effectors
        initial_positions = {
            'left': (0.5, 0.27, 0.66),
            'right': (0.5, -0.27, 0.66)
        }
        
        initial_orientations = {
            'left': (0, 0, 1, 0),  # wxyz
            'right': (0, 0, 1, 0)
        }
        
        for side, handle in self.transform_handles.items():
            if handle.control is not None:
                handle.control.position = initial_positions[side]
                handle.control.wxyz = initial_orientations[side]

    def solve_ik(self):
        """Solve inverse kinematics using pyroki."""
        # Get target poses from transform controls
        target_poses = self.get_target_poses()
        
        # Calculate TCP poses for arms (combining control handle with TCP offset)
        ik_target_0_tcp = target_poses['left']
        ik_target_1_tcp = target_poses['right']
        
        # Use previous solution if available, otherwise use rest pose
        if self.solution is None:
            prev_cfg = self.rest_pose if self.rest_pose is not None else {}
        else:
            prev_cfg = self.solution
            
        # Prepare target positions and orientations
        target_positions = [
            ik_target_0_tcp.wxyz_xyz[-3:],
            ik_target_1_tcp.wxyz_xyz[-3:],
        ]
        
        target_wxyzs = [
            ik_target_0_tcp.wxyz_xyz[:4],
            ik_target_1_tcp.wxyz_xyz[:4],
        ]
        
        # Add head target if available
        if hasattr(self, 'head_handle'):
            target_positions.append(self.head_handle.position)
            target_wxyzs.append(self.head_handle.wxyz)
            
        # Solve IK with multiple targets
        self.solution = solve_ik_with_multiple_targets(
            robot=self.pyroki_robot,
            target_link_names=self.pyroki_target_link_names,
            target_positions=np.array(target_positions),
            target_wxyzs=np.array(target_wxyzs),
            prev_cfg=prev_cfg,
        )
        
        # Update joints for visualization
        self.joints = self.solution
            
    def home(self):
        """Reset robot to rest pose."""
        self.solution = None
        if self.rest_pose is not None:
            self.joints = self.rest_pose


def main():
    """Main function for bimanual IK with pyroki."""
    rby1_interface = RBY1PyrokiInterface()
    rby1_interface.run()


if __name__ == "__main__":
    main()
