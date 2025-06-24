"""
Abstract base class for RBY1 robot interface.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional
import time
from dataclasses import dataclass

import numpy as np
import viser
import viser.transforms as vtf
import viser.extras

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print(
        "pip install git+https://github.com/uynitsuj/robot_descriptions.py.git"
    )
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()


@dataclass
class TransformHandle:
    """Data class to store transform handles."""
    frame_tcp: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None


class RBY1AbstractBase(ABC):
    """
    Abstract base class for RBY1 robot visualization.
    - This class provides common functionality for different IK solvers
    - Subclasses must implement the solve_ik method with their specific solver
    """
    
    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None
        ):
        self.rate = rate
        
        # Load URDF for visualization
        self.urdf = load_urdf_robot_description("rby1_description")
        
        # Initialize viser server
        self.server = viser_server if viser_server is not None else viser.ViserServer()
        
        # Common target link names
        self.target_link_names = ["left_ee", "right_ee"]
        
        # Initialize joint configuration
        self.joints = None
        
        # Allow subclasses to do solver-specific setup first
        self._setup_solver_specific()
        
        # Setup common components
        self._setup_visualization()
        
        self._setup_gui()
        self._setup_transform_handles()
        
    def _setup_visualization(self):
        """Setup basic visualization elements."""
        # Add base frame and robot URDF
        self.base_frame = self.server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base"
        )
        
        # Add ground grid
        self.server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
        
    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing display
        self.timing_handle = self.server.gui.add_number("Time (ms)", 0.01, disabled=True)
        
        # Add gizmo size control
        self.tf_size_handle = self.server.gui.add_slider(
            "Gizmo size", min=0.05, max=0.4, step=0.01, initial_value=0.2
        )
        
        # Add reset button
        self.reset_button = self.server.gui.add_button("Reset to Rest Pose")
        
        @self.reset_button.on_click
        def _(_):
            self.home()
            
    def _setup_transform_handles(self):
        """Setup transform handles for end effectors."""
        self.transform_handles = {
            'left': TransformHandle(
                frame_tcp=self.server.scene.add_frame(
                    "target_left/tcp",
                    show_axes=False,
                    position=(0.0, 0.0, 0.0),
                    wxyz=(0, 0, 1, 0)
                ),
                control=self.server.scene.add_transform_controls(
                    "target_left",
                    scale=self.tf_size_handle.value
                )
            ),
            'right': TransformHandle(
                frame_tcp=self.server.scene.add_frame(
                    "target_right/tcp",
                    show_axes=False,
                    position=(0.0, 0.0, 0.0),
                    wxyz=(0, 0, 1, 0)
                ),
                control=self.server.scene.add_transform_controls(
                    "target_right",
                    scale=self.tf_size_handle.value
                )
            )
        }
        
        # Update transform handles when size changes
        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
            # Let subclasses handle their own optional handles
            self._update_optional_handle_sizes()
                    
        # Initialize transform handle positions
        self._initialize_transform_handles()
        
    def _update_optional_handle_sizes(self):
        """Override in subclasses to update optional handle sizes."""
        pass
        
    def update_visualization(self):
        """Update visualization with current state."""
        if self.joints is not None:
            # Update robot configuration
            self.urdf_vis.update_cfg(self.joints)
            
    def get_target_poses(self):
        """Get target poses from transform controls."""
        target_poses = {}
        
        for side, handle in self.transform_handles.items():
            if handle.control is None:
                continue
                
            # Combine control handle with TCP offset
            control_pose = vtf.SE3(np.array([
                *handle.control.wxyz, 
                *handle.control.position
            ]))
            
            tcp_pose = vtf.SE3(np.array([
                *handle.frame_tcp.wxyz, 
                *handle.frame_tcp.position
            ]))
            
            target_poses[side] = control_pose @ tcp_pose
            
        return target_poses
    
    def home(self):
        """Reset robot to rest pose. Must be implemented by subclasses."""
        pass
        
    def run(self):
        """Main run loop."""
        while True:
            start_time = time.time()
            
            self.solve_ik()
            self.update_visualization()
            
            # Update timing
            elapsed_time = time.time() - start_time
            if hasattr(self, 'timing_handle'):
                self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * (elapsed_time * 1000)
                
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _setup_solver_specific(self):
        """Setup solver-specific components. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _initialize_transform_handles(self):
        """Initialize transform handle positions. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def solve_ik(self):
        """Solve inverse kinematics. Must be implemented by subclasses."""
        pass 