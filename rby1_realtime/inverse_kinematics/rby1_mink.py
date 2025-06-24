"""
RBY1 Mink Interface - uses mink for inverse kinematics.
"""

import time
import numpy as np
import viser.transforms as vtf
import mujoco

try:
    from robot_descriptions.loaders.mujoco import load_robot_description as load_mujoco_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print(
        "pip install git+https://github.com/uynitsuj/robot_descriptions.py.git"
    )
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()

import mink
from typing import Optional

from rby1_realtime.base import RBY1AbstractBase


def get_site_pose_mat4x4(data, site_name):
    site_pos = data.site(site_name).xpos.reshape(3, 1)
    site_rot = data.site(site_name).xmat.reshape(3, 3)
    site_pose = np.eye(4)
    _site_pose = np.concatenate([site_rot, site_pos], axis=1)
    site_pose[:3, :] = _site_pose
    return site_pose


def get_body_pose_mat4x4(data, body_name):
    body_pos = data.body(body_name).xpos.reshape(3, 1)
    body_rot = data.body(body_name).xmat.reshape(3, 3)
    body_pose = np.eye(4)
    _body_pose = np.concatenate([body_rot, body_pos], axis=1)
    body_pose[:3, :] = _body_pose
    return body_pose


def mj_to_urdf_map(model, qpos):
    mj_joint_dict = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): qpos[i]
        for i in range(model.njnt)
    }

    urdf_joint_names = [
        'right_wheel', 'left_wheel', 'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
        'gripper_finger_r1', 'gripper_finger_r2', 'gripper_finger_l1', 'gripper_finger_l2',
        'head_0', 'head_1'
    ]

    return np.array([mj_joint_dict[name] for name in urdf_joint_names])

class RBY1MinkInterface(RBY1AbstractBase):
    """
    RBY1 interface using mink for inverse kinematics.
    """

    def __init__(self, rate: float = 100.0, head_task_active: bool = True, torso_task_active: bool = True, viser_server=None):
        self.head_task_active = head_task_active
        self.torso_task_active = torso_task_active
        
        # Optional handles (will be created if tasks are active)
        self.head_handle = None
        self.torso_handle = None
        
        super().__init__(rate=rate, viser_server=viser_server)
        self.home()

    def _setup_solver_specific(self):
        """Setup mink-specific components."""
        # Load mujoco model for mink
        self.mujoco_model = load_mujoco_robot_description("rby1_mj_description")
        self.configuration = mink.Configuration(self.mujoco_model)
        
        # Setup mink tasks
        self._setup_mink_tasks()

    def _setup_mink_tasks(self):
        """Setup mink tasks and limits."""
        self.posture_task = mink.PostureTask(self.mujoco_model, cost=1e-1)
        self.tasks = [self.posture_task]
        
        # Arm end-effector tasks (relative to torso)
        self.arm_tasks = {}
        for link_name in self.target_link_names:
            task = mink.RelativeFrameTask(
                frame_name=link_name,
                frame_type="site",
                root_name=f"link_torso_5",
                root_type="body",
                position_cost=20.0,
                orientation_cost=5.0,
                lm_damping=1.0,
            )
            self.tasks.append(task)
            self.arm_tasks[link_name] = task
            
        # Optional head task (relative to base)
        self.head_task = mink.RelativeFrameTask(
            frame_name="link_head_2",  # or whatever your head link is called
            frame_type="body",
            root_name="base",
            root_type="body", 
            position_cost=5.0,
            orientation_cost=2.0,
            lm_damping=1.0,
        )
        
        # Optional torso task (relative to base)
        self.torso_task = mink.RelativeFrameTask(
            frame_name="link_torso_5",
            frame_type="body",
            root_name="base", 
            root_type="body",
            position_cost=3.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        
        # Add optional tasks to task list if they're active
        if self.head_task_active:
            self.tasks.append(self.head_task)
        if self.torso_task_active:
            self.tasks.append(self.torso_task)
        
        collision_pairs = [
            # Add collision pairs if needed
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.mujoco_model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        self.limits = [
            mink.ConfigurationLimit(self.mujoco_model),
            collision_avoidance_limit,
        ]

        joint_limits = self.mujoco_model.jnt_range
        # Set posture task to half of joint limits
        posture_target = np.zeros(self.mujoco_model.njnt)
        for i in range(self.mujoco_model.njnt):
            posture_target[i] = (joint_limits[i, 0] + joint_limits[i, 1]) / 2

        self.configuration.data.qpos = posture_target

        self.posture_task.set_target_from_configuration(self.configuration)
    
    def set_head_target(self, target_pose: Optional[np.ndarray] = None, enable: bool = True):
        """
        Set or enable/disable head IK target.
        
        Args:
            target_pose: 4x4 transformation matrix for head target (relative to base)
            enable: Whether to enable head IK
        """
        if enable and target_pose is not None:
            if not self.head_task_active:
                self.tasks.append(self.head_task)
                self.head_task_active = True
            self.head_task.set_target(mink.SE3.from_matrix(target_pose))
        elif not enable and self.head_task_active:
            self.tasks.remove(self.head_task)
            self.head_task_active = False
            
    def set_torso_target(self, target_pose: Optional[np.ndarray] = None, enable: bool = True):
        """
        Set or enable/disable torso IK target.
        
        Args:
            target_pose: 4x4 transformation matrix for torso target (relative to base)
            enable: Whether to enable torso IK
        """
        if enable and target_pose is not None:
            if not self.torso_task_active:
                self.tasks.append(self.torso_task)
                self.torso_task_active = True
            self.torso_task.set_target(mink.SE3.from_matrix(target_pose))
        elif not enable and self.torso_task_active:
            self.tasks.remove(self.torso_task)
            self.torso_task_active = False

    def _initialize_transform_handles(self):
        """Initialize transform handle positions based on current robot state."""
        base_pose = np.array(
            self.base_frame.wxyz.tolist() + self.base_frame.position.tolist()
        )
        
        for target_frame_handle, target_name in zip(
            list(self.transform_handles.values()), self.target_link_names
        ):  
            data = self.configuration.data
            rot_180_x = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_x_radians(np.pi), np.array([0.0, 0.0, 0.0]))
            
            if target_name == "left_ee":
                current_left_ee_pose = get_site_pose_mat4x4(data, "left_ee")
                T_target_world = vtf.SE3(base_pose) @ vtf.SE3.from_matrix(current_left_ee_pose) @ rot_180_x
            elif target_name == "right_ee":
                current_right_ee_pose = get_site_pose_mat4x4(data, "right_ee")
                T_target_world = vtf.SE3(base_pose) @ vtf.SE3.from_matrix(current_right_ee_pose) @ rot_180_x
            else:
                raise ValueError(f"Target link name {target_name} not found")

            if target_frame_handle.control is not None:
                target_frame_handle.control.position = np.array(T_target_world.translation())
                target_frame_handle.control.wxyz = np.array(T_target_world.rotation().wxyz)
                
        # Create optional transform handles if tasks are active
        if self.head_task_active:
            self.head_handle = self.server.scene.add_transform_controls(
                "target_head",
                scale=self.tf_size_handle.value,
                position=(0.0, 0.0, 1.3),
                wxyz=tuple(vtf.SO3.from_rpy_radians(0.0, np.pi/6, 0.0).wxyz)
            )
            # Set initial target from current pose
            data = self.configuration.data
            current_head_pose = get_body_pose_mat4x4(data, "link_head_2")
            self.head_task.set_target(mink.SE3.from_matrix(current_head_pose))
            
        if self.torso_task_active:
            self.torso_handle = self.server.scene.add_transform_controls(
                "target_torso",
                scale=self.tf_size_handle.value,
                position=(0.0, 0.0, 0.8),
                wxyz=tuple(vtf.SO3.from_rpy_radians(0.0, np.pi/8, 0.0).wxyz)
            )
            # Set initial target from current pose
            data = self.configuration.data
            current_torso_pose = get_body_pose_mat4x4(data, "link_torso_5")
            self.torso_task.set_target(mink.SE3.from_matrix(current_torso_pose))

    def _update_optional_handle_sizes(self):
        """Update optional handle sizes when transform size changes."""
        if self.head_handle:
            self.head_handle.scale = self.tf_size_handle.value
        if self.torso_handle:
            self.torso_handle.scale = self.tf_size_handle.value

    def _reset_transform_handles(self):
        # Reset to home pose:
        if self.transform_handles['left'].control is not None:
            self.transform_handles['left'].control.position = np.array([0.25, 0.35, 0.7832])
            self.transform_handles['left'].control.wxyz = vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0).wxyz
        if self.transform_handles['right'].control is not None:
            self.transform_handles['right'].control.position = np.array([0.25, -0.35, 0.7832])
            self.transform_handles['right'].control.wxyz = vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0).wxyz

    def solve_ik(self):
        """Solve inverse kinematics using mink."""
        data = self.configuration.data
        solver = "osqp"

        # Get target poses
        target_poses = self.get_target_poses()
        
        # Get current torso pose
        torso_pose = get_body_pose_mat4x4(data, "link_torso_5")
        
        # Get target poses in torso frame
        target_poses['left'] = vtf.SE3.from_matrix(torso_pose).inverse() @ vtf.SE3.from_matrix(target_poses['left'].as_matrix())
        target_poses['right'] = vtf.SE3.from_matrix(torso_pose).inverse() @ vtf.SE3.from_matrix(target_poses['right'].as_matrix())
        
        # Set targets for arm tasks
        self.arm_tasks['left_ee'].set_target(mink.SE3.from_matrix(target_poses['left'].as_matrix()))
        self.arm_tasks['right_ee'].set_target(mink.SE3.from_matrix(target_poses['right'].as_matrix()))
        
        # Update optional targets if controls are active
        if self.head_task_active and self.head_handle:
            head_target = np.eye(4)
            head_target[:3, :3] = vtf.SO3(self.head_handle.wxyz).as_matrix()
            head_target[:3, 3] = self.head_handle.position
            self.head_task.set_target(mink.SE3.from_matrix(head_target))
            
        if self.torso_task_active and self.torso_handle:
            torso_target = np.eye(4)
            torso_target[:3, :3] = vtf.SO3(self.torso_handle.wxyz).as_matrix()
            torso_target[:3, 3] = self.torso_handle.position
            self.torso_task.set_target(mink.SE3.from_matrix(torso_target))

        # Solve IK
        vel = mink.solve_ik(self.configuration, self.tasks, 1/self.rate, solver, 1e-1, limits=self.limits)
        self.configuration.integrate_inplace(vel, 1/self.rate)
        
        # Convert to URDF joint order
        self.joints = mj_to_urdf_map(self.configuration.model, self.configuration.data.qpos)

    def home(self):
        """Reset robot to rest pose."""
        # Reset mink configuration to rest pose
        # self.configuration = mink.Configuration(self.mujoco_model)
        # self.posture_task.set_target_from_configuration(self.configuration)
        self.joints = mj_to_urdf_map(self.configuration.model, self.configuration.data.qpos)
        self._reset_transform_handles()

if __name__ == "__main__":
    rby1 = RBY1MinkInterface()
    rby1.run()
