#!/usr/bin/env python3
"""
SO-ARM101 Unloading/Sorting Robot Node

Copyright 2025 inte-R-action Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from std_msgs.msg import String, Float32, Bool, Int32Array
from geometry_msgs.msg import Pose, Twist, PoseStamped, Point
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from vision_msgs.msg import Detection2DArray

import numpy as np
import torch
import cv2
from cv_bridge import CvBridge
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque
from enum import Enum

# Import ACT and robotic control components
from ..robotics.so_arm101_controller import SOArm101Controller
from ..robotics.act_policy import ACTPolicy
from ..robotics.safety_monitor import SafetyMonitor
from ..vision.oakd_eye_in_hand import OAKDEyeInHand
from ..utils.force_feedback import ForceFeedbackProcessor
from ..utils.performance_tracker import ManipulationPerformanceTracker


class ContaminationLevel(Enum):
    """Contamination levels for sorting decisions."""
    CLEAN = "CLEAN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    UNKNOWN = "UNKNOWN"


class SortingDestination(Enum):
    """Available sorting destinations."""
    APPROVED_CONTAINER = "approved_container"
    REJECTED_CONTAINER = "rejected_container"
    REVIEW_QUEUE = "review_queue"
    ISOLATION_CHAMBER = "isolation_chamber"
    QUARANTINE_AREA = "quarantine_area"


@dataclass
class SortingCommand:
    """Structured command for IV bag sorting operations."""
    command_type: str  # 'sort', 'isolate', 'transfer', 'inspect'
    iv_bag_id: int
    contamination_level: ContaminationLevel
    destination: SortingDestination
    handling_priority: str  # 'critical', 'high', 'normal', 'low'
    special_instructions: Dict[str, any]
    safety_requirements: Dict[str, any]


@dataclass
class SortingState:
    """Current state of the sorting operation."""
    current_iv_bag: Optional[int]
    sorting_phase: str  # 'idle', 'approaching', 'grasping', 'lifting', 'transporting', 'placing', 'releasing'
    destination_pose: Optional[Pose]
    contamination_level: ContaminationLevel
    isolation_mode: bool
    handling_confidence: float
    safety_status: str
    last_sort_time: float


class SOArm101UnloadingNode(Node):
    """
    Advanced SO-ARM101 unloading/sorting robot node
    
    def __init__(self):
        super().__init__('so_arm101_unloading_node')
        
        # Declare parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize core components
        self._initialize_robot_controller()
        self._initialize_act_policy()
        self._initialize_vision_system()
        self._initialize_safety_systems()
        self._initialize_sorting_destinations()
        
        # Initialize state management
        self._initialize_state()
        
        # Create ROS2 communication interfaces
        self._create_publishers()
        self._create_subscribers()
        self._create_action_servers()
        self._create_timers()
        
        # Performance tracking
        self.performance_tracker = ManipulationPerformanceTracker()
        
        self.get_logger().info('SO-ARM101 Unloading/Sorting Node initialized successfully')
        self.get_logger().info(f'Configuration: ACT Policy={self.act_policy_enabled}, '
                             f'Contamination Awareness={self.contamination_aware_sorting}, '
                             f'Isolation Mode={self.isolation_mode_enabled}')
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        # Robot configuration
        self.declare_parameter('robot_ip', '192.168.1.101')
        self.declare_parameter('robot_port', 9999)
        self.declare_parameter('joint_names', [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'
        ])
        
        # ACT Policy parameters
        self.declare_parameter('act_policy_enabled', True)
        self.declare_parameter('act_model_path', '/models/act_sorting_policy.pth')
        self.declare_parameter('act_chunk_size', 10)
        self.declare_parameter('act_observation_horizon', 5)
        self.declare_parameter('act_confidence_threshold', 0.8)
        
        # Vision parameters
        self.declare_parameter('eye_in_hand_enabled', True)
        self.declare_parameter('eye_in_hand_camera_id', 2)
        self.declare_parameter('visual_servoing_enabled', True)
        self.declare_parameter('pose_estimation_model', '/models/pose_estimation_sorting.pth')
        self.declare_parameter('destination_detection_enabled', True)
        
        # Safety parameters
        self.declare_parameter('force_feedback_enabled', True)
        self.declare_parameter('max_force_x', 8.0)  # Lower forces for sorting
        self.declare_parameter('max_force_y', 8.0)
        self.declare_parameter('max_force_z', 12.0)
        self.declare_parameter('collision_detection_enabled', True)
        self.declare_parameter('emergency_stop_enabled', True)
        
        # Sorting parameters
        self.declare_parameter('contamination_aware_sorting', True)
        self.declare_parameter('default_sorting_speed', 0.4)
        self.declare_parameter('contaminated_sorting_speed', 0.2)
        self.declare_parameter('isolation_mode_enabled', True)
        self.declare_parameter('quarantine_protocols_enabled', True)
        
        # Destination parameters
        self.declare_parameter('approved_container_pose', [0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
        self.declare_parameter('rejected_container_pose', [0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
        self.declare_parameter('review_queue_pose', [0.0, 0.3, 0.1, 0.0, 0.0, 0.0])
        self.declare_parameter('isolation_chamber_pose', [-0.2, 0.3, 0.1, 0.0, 0.0, 0.0])
        self.declare_parameter('quarantine_area_pose', [-0.3, 0.0, 0.1, 0.0, 0.0, 0.0])
        
        # Performance parameters
        self.declare_parameter('enable_performance_tracking', True)
        self.declare_parameter('target_sorting_success_rate', 0.95)
        self.declare_parameter('max_sorting_time', 25.0)  # seconds
        self.declare_parameter('enable_quality_logging', True)
        
        # Compliance parameters
        self.declare_parameter('compliance_mode', 'EU_GMP_ANNEX_1')
        self.declare_parameter('enable_audit_trail', True)
        self.declare_parameter('contamination_protocols_strict', True)
    
    def _get_parameters(self):
        """Get all parameter values."""
        # Robot configuration
        self.robot_ip = self.get_parameter('robot_ip').value
        self.robot_port = self.get_parameter('robot_port').value
        self.joint_names = self.get_parameter('joint_names').value
        
        # ACT Policy
        self.act_policy_enabled = self.get_parameter('act_policy_enabled').value
        self.act_model_path = self.get_parameter('act_model_path').value
        self.act_chunk_size = self.get_parameter('act_chunk_size').value
        self.act_observation_horizon = self.get_parameter('act_observation_horizon').value
        self.act_confidence_threshold = self.get_parameter('act_confidence_threshold').value
        
        # Vision
        self.eye_in_hand_enabled = self.get_parameter('eye_in_hand_enabled').value
        self.eye_in_hand_camera_id = self.get_parameter('eye_in_hand_camera_id').value
        self.visual_servoing_enabled = self.get_parameter('visual_servoing_enabled').value
        self.pose_estimation_model = self.get_parameter('pose_estimation_model').value
        self.destination_detection_enabled = self.get_parameter('destination_detection_enabled').value
        
        # Safety
        self.force_feedback_enabled = self.get_parameter('force_feedback_enabled').value
        self.max_forces = {
            'x': self.get_parameter('max_force_x').value,
            'y': self.get_parameter('max_force_y').value,
            'z': self.get_parameter('max_force_z').value
        }
        self.collision_detection_enabled = self.get_parameter('collision_detection_enabled').value
        self.emergency_stop_enabled = self.get_parameter('emergency_stop_enabled').value
        
        # Sorting
        self.contamination_aware_sorting = self.get_parameter('contamination_aware_sorting').value
        self.default_sorting_speed = self.get_parameter('default_sorting_speed').value
        self.contaminated_sorting_speed = self.get_parameter('contaminated_sorting_speed').value
        self.isolation_mode_enabled = self.get_parameter('isolation_mode_enabled').value
        self.quarantine_protocols_enabled = self.get_parameter('quarantine_protocols_enabled').value
        
        # Destinations
        self.destination_poses = {
            SortingDestination.APPROVED_CONTAINER: self._list_to_pose(
                self.get_parameter('approved_container_pose').value
            ),
            SortingDestination.REJECTED_CONTAINER: self._list_to_pose(
                self.get_parameter('rejected_container_pose').value
            ),
            SortingDestination.REVIEW_QUEUE: self._list_to_pose(
                self.get_parameter('review_queue_pose').value
            ),
            SortingDestination.ISOLATION_CHAMBER: self._list_to_pose(
                self.get_parameter('isolation_chamber_pose').value
            ),
            SortingDestination.QUARANTINE_AREA: self._list_to_pose(
                self.get_parameter('quarantine_area_pose').value
            )
        }
        
        # Performance
        self.enable_performance_tracking = self.get_parameter('enable_performance_tracking').value
        self.target_sorting_success_rate = self.get_parameter('target_sorting_success_rate').value
        self.max_sorting_time = self.get_parameter('max_sorting_time').value
        self.enable_quality_logging = self.get_parameter('enable_quality_logging').value
        
        # Compliance
        self.compliance_mode = self.get_parameter('compliance_mode').value
        self.enable_audit_trail = self.get_parameter('enable_audit_trail').value
        self.contamination_protocols_strict = self.get_parameter('contamination_protocols_strict').value
    
    def _list_to_pose(self, pose_list: List[float]) -> Pose:
        """Convert list of 6 values to Pose message."""
        pose = Pose()
        pose.position.x = pose_list[0]
        pose.position.y = pose_list[1]
        pose.position.z = pose_list[2]
        
        # Convert Euler angles to quaternion (simplified)
        pose.orientation.x = pose_list[3]
        pose.orientation.y = pose_list[4]
        pose.orientation.z = pose_list[5]
        pose.orientation.w = 1.0  # Normalized later
        
        return pose
    
    def _initialize_robot_controller(self):
        """Initialize SO-ARM101 robot controller."""
        try:
            self.robot_controller = SOArm101Controller(
                ip_address=self.robot_ip,
                port=self.robot_port,
                joint_names=self.joint_names
            )
            
            # Connect to robot
            if self.robot_controller.connect():
                self.get_logger().info('Successfully connected to SO-ARM101 unloading robot')
                
                # Initialize robot to home position
                self.robot_controller.move_to_home()
                
            else:
                self.get_logger().error('Failed to connect to SO-ARM101 unloading robot')
                raise ConnectionError('Robot connection failed')
                
        except Exception as e:
            self.get_logger().error(f'Error initializing robot controller: {e}')
            raise
    
    def _initialize_act_policy(self):
        """Initialize ACT policy for learned sorting behaviors."""
        if self.act_policy_enabled:
            try:
                self.act_policy = ACTPolicy(
                    model_path=self.act_model_path,
                    chunk_size=self.act_chunk_size,
                    observation_horizon=self.act_observation_horizon,
                    confidence_threshold=self.act_confidence_threshold
                )
                
                # Load the trained sorting model
                self.act_policy.load_model()
                
                # Initialize observation buffer for ACT
                self.observation_buffer = deque(maxlen=self.act_observation_horizon)
                
                self.get_logger().info('ACT sorting policy loaded successfully')
                
            except Exception as e:
                self.get_logger().error(f'Error loading ACT sorting policy: {e}')
                self.act_policy_enabled = False
        else:
            self.act_policy = None
    
    def _initialize_vision_system(self):
        """Initialize eye-in-hand vision system for sorting."""
        if self.eye_in_hand_enabled:
            try:
                self.eye_in_hand_camera = OAKDEyeInHand(
                    camera_id=self.eye_in_hand_camera_id,
                    pose_estimation_model=self.pose_estimation_model
                )
                
                self.bridge = CvBridge()
                
                self.get_logger().info('Eye-in-hand camera system for sorting initialized')
                
            except Exception as e:
                self.get_logger().error(f'Error initializing vision system: {e}')
                self.eye_in_hand_enabled = False
        else:
            self.eye_in_hand_camera = None
    
    def _initialize_safety_systems(self):
        """Initialize safety monitoring systems."""
        try:
            self.safety_monitor = SafetyMonitor(
                max_forces=self.max_forces,
                collision_detection_enabled=self.collision_detection_enabled,
                emergency_stop_enabled=self.emergency_stop_enabled
            )
            
            if self.force_feedback_enabled:
                self.force_processor = ForceFeedbackProcessor(
                    force_limits=self.max_forces,
                    grasp_threshold=5.0  # Gentle grasping for sorting
                )
            else:
                self.force_processor = None
            
            self.get_logger().info('Safety systems initialized for sorting operations')
            
        except Exception as e:
            self.get_logger().error(f'Error initializing safety systems: {e}')
            raise
    
    def _initialize_sorting_destinations(self):
        """Initialize sorting destination configurations."""
        # Destination-specific configurations
        self.destination_configs = {
            SortingDestination.APPROVED_CONTAINER: {
                'approach_height': 0.05,
                'place_force_limit': 3.0,
                'speed_factor': self.default_sorting_speed
            },
            SortingDestination.REJECTED_CONTAINER: {
                'approach_height': 0.08,
                'place_force_limit': 2.0,
                'speed_factor': self.contaminated_sorting_speed
            },
            SortingDestination.REVIEW_QUEUE: {
                'approach_height': 0.06,
                'place_force_limit': 2.5,
                'speed_factor': self.default_sorting_speed * 0.8
            },
            SortingDestination.ISOLATION_CHAMBER: {
                'approach_height': 0.10,
                'place_force_limit': 1.5,
                'speed_factor': self.contaminated_sorting_speed * 0.7
            },
            SortingDestination.QUARANTINE_AREA: {
                'approach_height': 0.12,
                'place_force_limit': 1.0,
                'speed_factor': self.contaminated_sorting_speed * 0.5
            }
        }
        
        self.get_logger().info('Sorting destinations configured')
    
    def _initialize_state(self):
        """Initialize state management."""
        self.sorting_state = SortingState(
            current_iv_bag=None,
            sorting_phase='idle',
            destination_pose=None,
            contamination_level=ContaminationLevel.UNKNOWN,
            isolation_mode=False,
            handling_confidence=0.0,
            safety_status='safe',
            last_sort_time=time.time()
        )
        
        # Command queue for managing sorting operations
        self.sorting_queue = deque()
        self.current_sorting_command = None
        
        # Thread safety
        self.state_lock = threading.Lock()
        
        # Contamination tracking and audit trail
        self.contamination_history = {}  # bag_id -> contamination_data
        self.sorting_audit_trail = []
        
        # Performance metrics
        self.sorting_statistics = {
            'total_sorts': 0,
            'successful_sorts': 0,
            'failed_sorts': 0,
            'contamination_isolations': 0,
            'average_sort_time': 0.0,
            'safety_incidents': 0,
            'compliance_violations': 0
        }
        
        # Destination usage tracking
        self.destination_usage = {dest: 0 for dest in SortingDestination}
    
    def _create_publishers(self):
        """Create all ROS2 publishers."""
        # Joint state publisher
        self.joint_state_pub = self.create_publisher(
            JointState, '/unloading_arm/joint_states', 10)
        
        # Sorting status publisher
        self.sorting_status_pub = self.create_publisher(
            String, '/unloading_arm/status', 10)
        
        # Sorting decisions publisher
        self.sorting_decisions_pub = self.create_publisher(
            String, '/unloading_arm/sorting_decisions', 10)
        
        # Visual feedback publisher
        self.visual_feedback_pub = self.create_publisher(
            Image, '/unloading_arm/visual_feedback', 10)
        
        # Safety status publisher
        self.safety_status_pub = self.create_publisher(
            String, '/unloading_arm/safety_status', 10)
        
        # Performance metrics publisher
        self.performance_pub = self.create_publisher(
            String, '/unloading_arm/performance_metrics', 10)
        
        # Audit trail publisher
        if self.enable_audit_trail:
            self.audit_trail_pub = self.create_publisher(
                String, '/unloading_arm/audit_trail', 10)
        
        # Quality logging publisher
        if self.enable_quality_logging:
            self.quality_log_pub = self.create_publisher(
                String, '/unloading_arm/quality_log', 10)
        
        # Force feedback publisher
        if self.force_feedback_enabled:
            self.force_feedback_pub = self.create_publisher(
                String, '/unloading_arm/force_feedback', 10)
    
    def _create_subscribers(self):
        """Create all ROS2 subscribers."""
        # Contamination feedback commands
        self.contamination_commands_sub = self.create_subscription(
            String, '/contamination_feedback/sorting_commands',
            self.contamination_commands_callback, 10)
        
        # Bag decision results
        self.bag_decisions_sub = self.create_subscription(
            String, '/contamination_analysis/bag_decision',
            self.bag_decisions_callback, 10)
        
        # System commands
        self.system_commands_sub = self.create_subscription(
            String, '/unloading_system/commands',
            self.system_commands_callback, 10)
        
        # Emergency stop
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop',
            self.emergency_stop_callback, 10)
        
        # Vision system detections for guidance
        self.detections_sub = self.create_subscription(
            Detection2DArray, '/vision_system/detections',
            self.detections_callback, 10)
        
        # Loading arm feedback for coordination
        self.loading_feedback_sub = self.create_subscription(
            String, '/loading_arm/status',
            self.loading_feedback_callback, 10)
    
    def _create_action_servers(self):
        """Create ROS2 action servers."""
        # Main sorting action server
        self.sorting_action_server = ActionServer(
            self, FollowJointTrajectory, '/unloading_arm/follow_joint_trajectory',
            self.sorting_action_callback)
    
    def _create_timers(self):
        """Create ROS2 timers."""
        # Main control loop (10 Hz)
        self.create_timer(0.1, self.control_loop_callback)
        
        # Status publishing (5 Hz)
        self.create_timer(0.2, self.publish_status_callback)
        
        # Safety monitoring (20 Hz)
        self.create_timer(0.05, self.safety_monitoring_callback)
        
        # Performance tracking (1 Hz)
        if self.enable_performance_tracking:
            self.create_timer(1.0, self.performance_tracking_callback)
        
        # Audit trail logging (0.5 Hz)
        if self.enable_audit_trail:
            self.create_timer(2.0, self.audit_trail_callback)
    
    def contamination_commands_callback(self, msg: String):
        """Handle contamination-aware sorting commands."""
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('command_type', '')
            
            if command_type == 'sort_bag':
                self._handle_sort_bag_command(command_data)
            elif command_type == 'isolate_contaminated':
                self._handle_isolate_contaminated_command(command_data)
            elif command_type == 'quarantine_severe':
                self._handle_quarantine_severe_command(command_data)
            elif command_type == 'emergency_isolation':
                self._handle_emergency_isolation_command(command_data)
            
        except Exception as e:
            self.get_logger().error(f'Error processing contamination command: {e}')
    
    def bag_decisions_callback(self, msg: String):
        """Handle final bag contamination decisions."""
        try:
            decision_data = json.loads(msg.data)
            iv_bag_id = decision_data.get('iv_bag_number', 0)
            final_decision = decision_data.get('final_decision', 'REVIEW')
            contamination_level = decision_data.get('contamination_level', 'UNKNOWN')
            
            # Update contamination history
            self.contamination_history[iv_bag_id] = {
                'decision': final_decision,
                'contamination_level': contamination_level,
                'total_particles': decision_data.get('total_particles', 0),
                'total_bubbles': decision_data.get('total_bubbles', 0),
                'timestamp': time.time(),
                'recommendation': decision_data.get('recommendation', '')
            }
            
            # Queue appropriate sorting command
            self._queue_sorting_based_on_decision(iv_bag_id, final_decision, contamination_level)
            
            self.get_logger().info(
                f'Received bag decision for bag {iv_bag_id}: {final_decision} '
                f'(contamination: {contamination_level})'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing bag decision: {e}')
    
    def system_commands_callback(self, msg: String):
        """Handle general system commands."""
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('type', '')
            
            if command_type == 'home_robot':
                self._execute_home_command()
            elif command_type == 'calibrate_destinations':
                self._execute_destination_calibration()
            elif command_type == 'enable_isolation_mode':
                self._enable_isolation_mode()
            elif command_type == 'disable_isolation_mode':
                self._disable_isolation_mode()
            elif command_type == 'clear_sorting_queue':
                self._clear_sorting_queue()
            
        except Exception as e:
            self.get_logger().error(f'Error processing system command: {e}')
    
    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop signals."""
        if msg.data:
            self.get_logger().warn('Emergency stop activated - halting all sorting operations')
            self._execute_emergency_stop()
    
    def detections_callback(self, msg: Detection2DArray):
        """Handle vision system detections for sorting guidance."""
        if self.visual_servoing_enabled and self.sorting_state.sorting_phase != 'idle':
            # Extract IV bag pose and destination information
            iv_bag_pose = self._extract_iv_bag_pose(msg.detections)
            destination_info = self._extract_destination_info(msg.detections)
            
            if iv_bag_pose:
                with self.state_lock:
                    self.sorting_state.destination_pose = iv_bag_pose
                    self.sorting_state.handling_confidence = self._calculate_handling_confidence(
                        msg.detections
                    )
    
    def loading_feedback_callback(self, msg: String):
        """Handle loading arm coordination feedback."""
        try:
            feedback_data = json.loads(msg.data)
            loading_status = feedback_data.get('loading_state', {})
            
            # Coordinate with loading operations
            if loading_status.get('loading_phase') == 'completed':
                current_bag = loading_status.get('current_iv_bag')
                if current_bag and current_bag in self.contamination_history:
                    # Trigger sorting for completed bag
                    self._trigger_automated_sorting(current_bag)
            
        except Exception as e:
            self.get_logger().error(f'Error processing loading feedback: {e}')
    
    def sorting_action_callback(self, goal_handle):
        """Handle joint trajectory following actions for sorting."""
        self.get_logger().info('Received sorting trajectory goal')
        
        # Execute trajectory using robot controller
        result = self.robot_controller.follow_joint_trajectory(goal_handle.request)
        
        goal_handle.succeed()
        return result
    
    def control_loop_callback(self):
        """Main control loop for sorting operations."""
        try:
            # Process sorting queue
            self._process_sorting_queue()
            
            # Execute current sorting command
            if self.current_sorting_command:
                self._execute_current_sorting_command()
            
            # Update ACT policy if enabled
            if self.act_policy_enabled and self.sorting_state.sorting_phase != 'idle':
                self._update_act_policy()
            
            # Publish joint states
            self._publish_joint_states()
            
        except Exception as e:
            self.get_logger().error(f'Error in sorting control loop: {e}')
    
    def publish_status_callback(self):
        """Publish sorting system status."""
        status_data = {
            'node_status': 'active',
            'sorting_state': {
                'current_iv_bag': self.sorting_state.current_iv_bag,
                'sorting_phase': self.sorting_state.sorting_phase,
                'contamination_level': self.sorting_state.contamination_level.value,
                'isolation_mode': self.sorting_state.isolation_mode,
                'handling_confidence': self.sorting_state.handling_confidence,
                'safety_status': self.sorting_state.safety_status
            },
            'robot_status': self.robot_controller.get_status(),
            'sorting_queue_length': len(self.sorting_queue),
            'act_policy_active': self.act_policy_enabled,
            'destination_usage': {dest.value: count for dest, count in self.destination_usage.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data, indent=2)
        self.sorting_status_pub.publish(status_msg)
    
    def safety_monitoring_callback(self):
        """Monitor safety conditions during sorting."""
        # Check force feedback
        if self.force_feedback_enabled:
            current_forces = self.robot_controller.get_force_feedback()
            safety_status = self.safety_monitor.check_forces(current_forces)
            
            if safety_status['emergency_stop_required']:
                self.get_logger().error('Force limits exceeded during sorting - emergency stop')
                self._execute_emergency_stop()
            
            # Update sorting state
            with self.state_lock:
                self.sorting_state.safety_status = safety_status['status']
            
            # Publish force feedback
            force_msg = String()
            force_msg.data = json.dumps(current_forces)
            self.force_feedback_pub.publish(force_msg)
        
        # Check collision detection
        if self.collision_detection_enabled:
            collision_status = self.safety_monitor.check_collisions(
                self.robot_controller.get_current_pose()
            )
            
            if collision_status['collision_detected']:
                self.get_logger().warn('Collision detected during sorting - adjusting trajectory')
                self._handle_collision_detection(collision_status)
        
        # Publish safety status
        safety_msg = String()
        safety_msg.data = json.dumps({
            'safety_status': self.sorting_state.safety_status,
            'force_limits_ok': not self.safety_monitor.force_limits_exceeded,
            'collision_free': not self.safety_monitor.collision_detected,
            'isolation_protocols_active': self.sorting_state.isolation_mode
        })
        self.safety_status_pub.publish(safety_msg)
    
    def performance_tracking_callback(self):
        """Track and publish sorting performance metrics."""
        if self.enable_performance_tracking:
            metrics = self.performance_tracker.get_current_metrics()
            
            # Update internal statistics
            self.sorting_statistics.update(metrics)
            
            # Calculate success rate
            success_rate = (self.sorting_statistics['successful_sorts'] / 
                          max(self.sorting_statistics['total_sorts'], 1))
            
            if success_rate < self.target_sorting_success_rate:
                self.get_logger().warn(
                    f'Sorting success rate ({success_rate:.2f}) below target '
                    f'({self.target_sorting_success_rate:.2f})'
                )
            
            # Publish performance metrics
            performance_data = {
                'statistics': self.sorting_statistics,
                'success_rate': success_rate,
                'target_success_rate': self.target_sorting_success_rate,
                'destination_efficiency': self._calculate_destination_efficiency(),
                'contamination_isolation_rate': (
                    self.sorting_statistics['contamination_isolations'] / 
                    max(self.sorting_statistics['total_sorts'], 1)
                ),
                'compliance_score': self._calculate_compliance_score()
            }
            
            perf_msg = String()
            perf_msg.data = json.dumps(performance_data, indent=2)
            self.performance_pub.publish(perf_msg)
    
    def audit_trail_callback(self):
        """Publish audit trail for compliance tracking."""
        if self.enable_audit_trail and self.sorting_audit_trail:
            audit_data = {
                'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'compliance_mode': self.compliance_mode,
                'recent_operations': self.sorting_audit_trail[-10:],  # Last 10 operations
                'total_operations': len(self.sorting_audit_trail),
                'contamination_isolations': self.sorting_statistics['contamination_isolations'],
                'compliance_violations': self.sorting_statistics['compliance_violations']
            }
            
            audit_msg = String()
            audit_msg.data = json.dumps(audit_data, indent=2)
            self.audit_trail_pub.publish(audit_msg)
    
    def _handle_sort_bag_command(self, command_data: Dict):
        """Handle standard bag sorting command."""
        iv_bag_id = command_data.get('iv_bag_id', 0)
        destination = command_data.get('destination', 'review_queue')
        priority = command_data.get('priority', 'normal')
        handling_instructions = command_data.get('handling_instructions', 'standard_handling')
        
        # Determine contamination level and destination
        contamination_info = self.contamination_history.get(iv_bag_id, {})
        contamination_level = ContaminationLevel(
            contamination_info.get('contamination_level', 'UNKNOWN')
        )
        
        sorting_destination = self._map_destination_string(destination)
        
        # Create sorting command
        sorting_command = SortingCommand(
            command_type='sort',
            iv_bag_id=iv_bag_id,
            contamination_level=contamination_level,
            destination=sorting_destination,
            handling_priority=priority,
            special_instructions={'handling': handling_instructions},
            safety_requirements=self._get_safety_requirements(contamination_level)
        )
        
        self.sorting_queue.append(sorting_command)
        
        self.get_logger().info(
            f'Queued sorting command for bag {iv_bag_id} to {destination} '
            f'(contamination: {contamination_level.value})'
        )
    
    def _queue_sorting_based_on_decision(self, iv_bag_id: int, decision: str, 
                                       contamination_level: str):
        """Queue sorting command based on contamination analysis decision."""
        
        # Map decision to destination
        destination_mapping = {
            'ACCEPT': SortingDestination.APPROVED_CONTAINER,
            'REJECT': SortingDestination.REJECTED_CONTAINER,
            'REVIEW': SortingDestination.REVIEW_QUEUE
        }
        
        destination = destination_mapping.get(decision, SortingDestination.REVIEW_QUEUE)
        
        # Special handling for severe contamination
        contamination_enum = ContaminationLevel(contamination_level)
        if contamination_enum in [ContaminationLevel.HIGH, ContaminationLevel.SEVERE]:
            if self.isolation_mode_enabled:
                destination = SortingDestination.ISOLATION_CHAMBER
            if contamination_enum == ContaminationLevel.SEVERE and self.quarantine_protocols_enabled:
                destination = SortingDestination.QUARANTINE_AREA
        
        # Create sorting command
        sorting_command = SortingCommand(
            command_type='sort',
            iv_bag_id=iv_bag_id,
            contamination_level=contamination_enum,
            destination=destination,
            handling_priority='critical' if contamination_enum in [ContaminationLevel.HIGH, ContaminationLevel.SEVERE] else 'normal',
            special_instructions={'decision': decision},
            safety_requirements=self._get_safety_requirements(contamination_enum)
        )
        
        # Priority insertion for contaminated bags
        if contamination_enum in [ContaminationLevel.HIGH, ContaminationLevel.SEVERE]:
            self.sorting_queue.appendleft(sorting_command)  # High priority
        else:
            self.sorting_queue.append(sorting_command)
        
        self.get_logger().info(
            f'Queued sorting for bag {iv_bag_id}: {decision} -> {destination.value} '
            f'(priority: {sorting_command.handling_priority})'
        )
    
    def _process_sorting_queue(self):
        """Process the sorting command queue."""
        if not self.current_sorting_command and self.sorting_queue:
            self.current_sorting_command = self.sorting_queue.popleft()
            self.sorting_statistics['total_sorts'] += 1
            
            # Log audit trail entry
            if self.enable_audit_trail:
                audit_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'operation': 'sorting_started',
                    'iv_bag_id': self.current_sorting_command.iv_bag_id,
                    'destination': self.current_sorting_command.destination.value,
                    'contamination_level': self.current_sorting_command.contamination_level.value,
                    'priority': self.current_sorting_command.handling_priority
                }
                self.sorting_audit_trail.append(audit_entry)
            
            self.get_logger().info(
                f'Starting sorting operation: Bag {self.current_sorting_command.iv_bag_id} '
                f'-> {self.current_sorting_command.destination.value}'
            )
    
    def _execute_current_sorting_command(self):
        """Execute the current sorting command."""
        if not self.current_sorting_command:
            return
        
        command = self.current_sorting_command
        
        # Execute based on command type
        if command.command_type == 'sort':
            self._execute_sorting_sequence(command)
        elif command.command_type == 'isolate':
            self._execute_isolation_sequence(command)
        elif command.command_type == 'transfer':
            self._execute_transfer_sequence(command)
        
        # Check if command is complete
        if self._is_sorting_command_complete():
            self._complete_current_sorting_command()
    
    def _execute_sorting_sequence(self, command: SortingCommand):
        """Execute complete IV bag sorting sequence."""
        phase = self.sorting_state.sorting_phase
        
        if phase == 'idle':
            # Start sorting sequence
            with self.state_lock:
                self.sorting_state.current_iv_bag = command.iv_bag_id
                self.sorting_state.contamination_level = command.contamination_level
                self.sorting_state.sorting_phase = 'approaching'
                self.sorting_state.isolation_mode = (
                    command.destination in [SortingDestination.ISOLATION_CHAMBER, 
                                          SortingDestination.QUARANTINE_AREA]
                )
            
            # Configure robot for contamination level
            self._configure_robot_for_contamination(command.contamination_level)
            
            # Use ACT policy or traditional control
            if self.act_policy_enabled:
                self._execute_act_policy_sorting(command)
            else:
                self._execute_traditional_sorting(command)
        
        elif phase == 'approaching':
            self._execute_approach_phase(command)
        
        elif phase == 'grasping':
            self._execute_grasp_phase(command)
        
        elif phase == 'lifting':
            self._execute_lift_phase(command)
        
        elif phase == 'transporting':
            self._execute_transport_phase(command)
        
        elif phase == 'placing':
            self._execute_place_phase(command)
        
        elif phase == 'releasing':
            self._execute_release_phase(command)
    
    def _get_safety_requirements(self, contamination_level: ContaminationLevel) -> Dict:
        """Get safety requirements based on contamination level."""
        base_requirements = {
            'gentle_handling': True,
            'force_monitoring': True,
            'collision_avoidance': True
        }
        
        if contamination_level in [ContaminationLevel.HIGH, ContaminationLevel.SEVERE]:
            base_requirements.update({
                'isolation_protocols': True,
                'reduced_speed': True,
                'enhanced_monitoring': True,
                'contamination_containment': True
            })
        
        return base_requirements
    
    def _configure_robot_for_contamination(self, contamination_level: ContaminationLevel):
        """Configure robot parameters based on contamination level."""
        if contamination_level in [ContaminationLevel.HIGH, ContaminationLevel.SEVERE]:
            # Use contaminated handling parameters
            self.robot_controller.set_speed_factor(self.contaminated_sorting_speed)
            self.robot_controller.set_force_limits({
                'x': self.max_forces['x'] * 0.7,
                'y': self.max_forces['y'] * 0.7,
                'z': self.max_forces['z'] * 0.7
            })
        else:
            # Use normal handling parameters
            self.robot_controller.set_speed_factor(self.default_sorting_speed)
            self.robot_controller.set_force_limits(self.max_forces)
    
    def _map_destination_string(self, destination_str: str) -> SortingDestination:
        """Map destination string to SortingDestination enum."""
        mapping = {
            'approved_container': SortingDestination.APPROVED_CONTAINER,
            'rejected_container': SortingDestination.REJECTED_CONTAINER,
            'review_queue': SortingDestination.REVIEW_QUEUE,
            'isolation_chamber': SortingDestination.ISOLATION_CHAMBER,
            'quarantine_area': SortingDestination.QUARANTINE_AREA
        }
        return mapping.get(destination_str, SortingDestination.REVIEW_QUEUE)
    
    def _calculate_destination_efficiency(self) -> Dict:
        """Calculate efficiency metrics for each destination."""
        total_sorts = sum(self.destination_usage.values())
        if total_sorts == 0:
            return {dest.value: 0.0 for dest in SortingDestination}
        
        return {
            dest.value: (count / total_sorts) * 100 
            for dest, count in self.destination_usage.items()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate compliance score based on operations."""
        total_operations = len(self.sorting_audit_trail)
        if total_operations == 0:
            return 100.0
        
        violations = self.sorting_statistics['compliance_violations']
        return max(0.0, ((total_operations - violations) / total_operations) * 100)
    
    def _is_sorting_command_complete(self) -> bool:
        """Check if current sorting command is complete."""
        if not self.current_sorting_command:
            return False
        
        return self.sorting_state.sorting_phase == 'idle'
    
    def _complete_current_sorting_command(self):
        """Complete the current sorting command and update statistics."""
        if self.current_sorting_command:
            # Update destination usage
            destination = self.current_sorting_command.destination
            self.destination_usage[destination] += 1
            
            # Update statistics
            if self.sorting_state.safety_status == 'safe':
                self.sorting_statistics['successful_sorts'] += 1
                
                # Count contamination isolations
                if destination in [SortingDestination.ISOLATION_CHAMBER, 
                                 SortingDestination.QUARANTINE_AREA]:
                    self.sorting_statistics['contamination_isolations'] += 1
            else:
                self.sorting_statistics['failed_sorts'] += 1
            
            # Publish sorting decision
            self._publish_sorting_decision()
            
            # Log audit trail completion
            if self.enable_audit_trail:
                audit_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'operation': 'sorting_completed',
                    'iv_bag_id': self.current_sorting_command.iv_bag_id,
                    'destination': self.current_sorting_command.destination.value,
                    'success': self.sorting_state.safety_status == 'safe',
                    'duration': time.time() - self.sorting_state.last_sort_time
                }
                self.sorting_audit_trail.append(audit_entry)
            
            # Reset state
            with self.state_lock:
                self.sorting_state.sorting_phase = 'idle'
                self.sorting_state.current_iv_bag = None
                self.sorting_state.contamination_level = ContaminationLevel.UNKNOWN
                self.sorting_state.isolation_mode = False
            
            self.current_sorting_command = None
            
            self.get_logger().info('Sorting command completed successfully')
    
    def _publish_sorting_decision(self):
        """Publish the sorting decision for tracking."""
        if self.current_sorting_command:
            decision_data = {
                'iv_bag_id': self.current_sorting_command.iv_bag_id,
                'destination': self.current_sorting_command.destination.value,
                'contamination_level': self.current_sorting_command.contamination_level.value,
                'handling_priority': self.current_sorting_command.handling_priority,
                'isolation_mode': self.sorting_state.isolation_mode,
                'timestamp': datetime.now().isoformat(),
                'success': self.sorting_state.safety_status == 'safe'
            }
            
            decision_msg = String()
            decision_msg.data = json.dumps(decision_data, indent=2)
            self.sorting_decisions_pub.publish(decision_msg)
    
    def _execute_emergency_stop(self):
        """Execute emergency stop procedure for sorting operations."""
        # Stop robot immediately
        self.robot_controller.emergency_stop()
        
        # Clear sorting queue
        self.sorting_queue.clear()
        self.current_sorting_command = None
        
        # Reset state
        with self.state_lock:
            self.sorting_state.sorting_phase = 'idle'
            self.sorting_state.safety_status = 'emergency_stop'
            self.sorting_state.isolation_mode = False
        
        # Update statistics
        self.sorting_statistics['safety_incidents'] += 1
        
        # Log emergency in audit trail
        if self.enable_audit_trail:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'emergency_stop',
                'reason': 'safety_violation',
                'affected_bag': self.sorting_state.current_iv_bag
            }
            self.sorting_audit_trail.append(audit_entry)
    
    def _publish_joint_states(self):
        """Publish current joint states."""
        joint_states = self.robot_controller.get_joint_states()
        
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = joint_states['positions']
        joint_state_msg.velocity = joint_states['velocities']
        joint_state_msg.effort = joint_states['efforts']
        
        self.joint_state_pub.publish(joint_state_msg)
    
    # Placeholder methods for phase execution (to be implemented based on specific requirements)
    def _execute_act_policy_sorting(self, command: SortingCommand): pass
    def _execute_traditional_sorting(self, command: SortingCommand): pass
    def _execute_approach_phase(self, command: SortingCommand): pass
    def _execute_grasp_phase(self, command: SortingCommand): pass
    def _execute_lift_phase(self, command: SortingCommand): pass
    def _execute_transport_phase(self, command: SortingCommand): pass
    def _execute_place_phase(self, command: SortingCommand): pass
    def _execute_release_phase(self, command: SortingCommand): pass
    def _extract_iv_bag_pose(self, detections: List) -> Optional[Pose]: return None
    def _extract_destination_info(self, detections: List) -> Optional[Dict]: return None
    def _calculate_handling_confidence(self, detections: List) -> float: return 0.5


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = SOArm101UnloadingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SO-ARM101 Unloading/Sorting Node shutting down...')
    finally:
        # Cleanup
        if hasattr(node, 'robot_controller'):
            node.robot_controller.disconnect()
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
