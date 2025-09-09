#!/usr/bin/env python3
"""
Complete System Launch File for ROS2 Pharmaceutical IV Bag Vision System

Launch with:
    ros2 launch ros2_pharma_iv_vision complete_system.launch.py

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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate the complete system launch description."""
    
    # Package directory
    pkg_dir = get_package_share_directory('ros2_pharma_iv_vision')
    
    # Declare launch arguments
    launch_args = [
        # System configuration
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level (debug, info, warn, error)'
        ),
        
        # Vision system parameters
        DeclareLaunchArgument(
            'camera_type',
            default_value='oakd',
            description='Camera type: oakd or webcam'
        ),
        DeclareLaunchArgument(
            'use_gpu',
            default_value='true',
            description='Enable GPU acceleration for vision processing'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='/models/efficientdet_contamination_model.pth',
            description='Path to the contamination detection model'
        ),
        DeclareLaunchArgument(
            'fps_target',
            default_value='30',
            description='Target FPS for vision processing'
        ),
        
        # Contamination analysis parameters
        DeclareLaunchArgument(
            'particle_threshold',
            default_value='5',
            description='Maximum allowed particles per analysis'
        ),
        DeclareLaunchArgument(
            'bubble_threshold',
            default_value='10',
            description='Maximum allowed bubbles per analysis'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.7',
            description='Minimum detection confidence threshold'
        ),
        DeclareLaunchArgument(
            'enable_adaptive_thresholds',
            default_value='true',
            description='Enable adaptive threshold management'
        ),
        
        # Robot control parameters
        DeclareLaunchArgument(
            'enable_loading_robot',
            default_value='true',
            description='Enable SO-ARM101 loading robot'
        ),
        DeclareLaunchArgument(
            'enable_unloading_robot',
            default_value='true',
            description='Enable SO-ARM101 unloading robot'
        ),
        DeclareLaunchArgument(
            'loading_robot_ip',
            default_value='192.168.1.100',
            description='IP address of loading robot'
        ),
        DeclareLaunchArgument(
            'unloading_robot_ip',
            default_value='192.168.1.101',
            description='IP address of unloading robot'
        ),
        DeclareLaunchArgument(
            'act_policy_enabled',
            default_value='true',
            description='Enable ACT policy for learned manipulation'
        ),
        DeclareLaunchArgument(
            'force_feedback_enabled',
            default_value='true',
            description='Enable force feedback for gentle handling'
        ),
        
        # Hardware control parameters
        DeclareLaunchArgument(
            'enable_gcode_control',
            default_value='true',
            description='Enable G-code stepper motor control'
        ),
        DeclareLaunchArgument(
            'stepper_port',
            default_value='/dev/ttyUSB0',
            description='Serial port for stepper motor control'
        ),
        
        # Monitoring and visualization
        DeclareLaunchArgument(
            'enable_visualization',
            default_value='true',
            description='Enable real-time visualization'
        ),
        DeclareLaunchArgument(
            'enable_performance_monitoring',
            default_value='true',
            description='Enable performance monitoring'
        ),
        DeclareLaunchArgument(
            'publish_debug_images',
            default_value='false',
            description='Publish debug images (bandwidth intensive)'
        ),
        
        # Data recording and storage
        DeclareLaunchArgument(
            'save_results',
            default_value='true',
            description='Save analysis results to disk'
        ),
        DeclareLaunchArgument(
            'results_directory',
            default_value='/tmp/iv_bag_results',
            description='Directory for saving results'
        ),
        DeclareLaunchArgument(
            'enable_rosbag_recording',
            default_value='false',
            description='Enable automatic rosbag recording'
        ),
    ]
    
    # Set global parameters
    global_params = [
        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),
    ]
    
    # Core Vision System Node
    vision_system_node = Node(
        package='ros2_pharma_iv_vision',
        executable='vision_system_node',
        name='vision_system_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'camera_type': LaunchConfiguration('camera_type'),
            'use_gpu': LaunchConfiguration('use_gpu'),
            'fps_target': LaunchConfiguration('fps_target'),
            'save_results': LaunchConfiguration('save_results'),
            'results_directory': LaunchConfiguration('results_directory'),
            'publish_debug_images': LaunchConfiguration('publish_debug_images'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # Contamination Analysis Node
    contamination_analysis_node = Node(
        package='ros2_pharma_iv_vision',
        executable='contamination_analysis_node',
        name='contamination_analysis_node',
        parameters=[{
            'particle_threshold': LaunchConfiguration('particle_threshold'),
            'bubble_threshold': LaunchConfiguration('bubble_threshold'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'enable_adaptive_thresholds': LaunchConfiguration('enable_adaptive_thresholds'),
            'save_results': LaunchConfiguration('save_results'),
            'results_directory': LaunchConfiguration('results_directory'),
            'compliance_mode': 'EU_GMP_ANNEX_1',
            'enable_quality_metrics': True,
            'enable_batch_reporting': True,
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # SO-ARM101 Loading Robot Node (conditional)
    loading_robot_node = Node(
        package='ros2_pharma_iv_vision',
        executable='so_arm101_loading_node',
        name='so_arm101_loading_node',
        parameters=[{
            'robot_ip': LaunchConfiguration('loading_robot_ip'),
            'robot_port': 9999,
            'act_policy_enabled': LaunchConfiguration('act_policy_enabled'),
            'eye_in_hand_enabled': True,
            'force_feedback_enabled': LaunchConfiguration('force_feedback_enabled'),
            'collision_detection_enabled': True,
            'emergency_stop_enabled': True,
            'enable_performance_tracking': True,
        }],
        condition=IfCondition(LaunchConfiguration('enable_loading_robot')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # SO-ARM101 Unloading Robot Node (conditional)
    unloading_robot_node = Node(
        package='ros2_pharma_iv_vision',
        executable='so_arm101_unloading_node',
        name='so_arm101_unloading_node',
        parameters=[{
            'robot_ip': LaunchConfiguration('unloading_robot_ip'),
            'robot_port': 9999,
            'act_policy_enabled': LaunchConfiguration('act_policy_enabled'),
            'eye_in_hand_enabled': True,
            'force_feedback_enabled': LaunchConfiguration('force_feedback_enabled'),
            'contamination_aware_sorting': True,
            'isolation_mode_enabled': True,
        }],
        condition=IfCondition(LaunchConfiguration('enable_unloading_robot')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # G-code Control Node for Stepper Motor (conditional)
    gcode_control_node = Node(
        package='ros2_pharma_iv_vision',
        executable='gcode_control_node',
        name='gcode_control_node',
        parameters=[{
            'serial_port': LaunchConfiguration('stepper_port'),
            'baud_rate': 115200,
            'analysis_positions': [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            'agitation_position': 40.0,
            'home_position': 0.0,
            'precision': 0.1,  # ±0.1mm precision
        }],
        condition=IfCondition(LaunchConfiguration('enable_gcode_control')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # Loading System Coordination Node
    loading_system_node = Node(
        package='ros2_pharma_iv_vision',
        executable='loading_system_node',
        name='loading_system_node',
        parameters=[{
            'iv_bag_count': 5,
            'positions_per_bag': 2,
            'analysis_duration': 6.0,
            'stabilization_time': 1.0,
            'enable_contamination_awareness': True,
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # Performance Monitor Node (conditional)
    performance_monitor_node = Node(
        package='ros2_pharma_iv_vision',
        executable='performance_monitor',
        name='performance_monitor',
        parameters=[{
            'enable_system_monitoring': True,
            'enable_resource_monitoring': True,
            'monitoring_frequency': 1.0,  # 1 Hz
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'gpu_usage': 90.0,
                'processing_fps_min': 25.0,
            },
        }],
        condition=IfCondition(LaunchConfiguration('enable_performance_monitoring')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # Vision System Visualizer Node (conditional)
    visualizer_node = Node(
        package='ros2_pharma_iv_vision',
        executable='vision_system_visualizer',
        name='vision_system_visualizer',
        parameters=[{
            'display_processed_image': True,
            'display_detection_overlay': True,
            'display_performance_metrics': True,
            'window_width': 1280,
            'window_height': 720,
            'enable_fullscreen': False,
        }],
        condition=IfCondition(LaunchConfiguration('enable_visualization')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # ROS Bag Recording (conditional)
    rosbag_recording = Node(
        package='ros2_pharma_iv_vision',
        executable='rosbag_recorder_node',
        name='rosbag_recorder',
        parameters=[{
            'output_directory': LaunchConfiguration('results_directory'),
            'topics_to_record': [
                '/vision_system/processed_image',
                '/vision_metrics/particle_count',
                '/vision_metrics/bubble_count',
                '/contamination_analysis/result',
                '/loading_arm/joint_states',
                '/unloading_arm/joint_states',
                '/current_position',
                '/analysis_active',
            ],
            'max_bag_size': '2GB',
            'compression_mode': 'file',
        }],
        condition=IfCondition(LaunchConfiguration('enable_rosbag_recording')),
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True,
    )
    
    # TF2 Static Transform Publishers for Camera Frames
    camera_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_tf_publisher',
        arguments=[
            '0', '0', '0.5',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'world', 'camera_base_frame'
        ],
    )
    
    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_tf_publisher',
        arguments=[
            '0', '0', '0',  # x, y, z
            '-0.5', '0.5', '-0.5', '0.5',  # qx, qy, qz, qw (90° rotations)
            'camera_base_frame', 'camera_optical_frame'
        ],
    )
    
    # Robot Base TF Publishers (conditional)
    loading_robot_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='loading_robot_tf_publisher',
        arguments=[
            '-0.5', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'world', 'loading_robot_base'
        ],
        condition=IfCondition(LaunchConfiguration('enable_loading_robot')),
    )
    
    unloading_robot_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='unloading_robot_tf_publisher',
        arguments=[
            '0.5', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'world', 'unloading_robot_base'
        ],
        condition=IfCondition(LaunchConfiguration('enable_unloading_robot')),
    )
    
    # Group all nodes for better organization
    core_system_group = GroupAction([
        vision_system_node,
        contamination_analysis_node,
        loading_system_node,
        gcode_control_node,
    ])
    
    robotic_system_group = GroupAction([
        loading_robot_node,
        unloading_robot_node,
    ])
    
    monitoring_group = GroupAction([
        performance_monitor_node,
        visualizer_node,
        rosbag_recording,
    ])
    
    tf_group = GroupAction([
        camera_base_tf,
        camera_optical_tf,
        loading_robot_tf,
        unloading_robot_tf,
    ])
    
    # Combine all launch components
    return LaunchDescription(
        launch_args +
        global_params +
        [
            core_system_group,
            robotic_system_group,
            monitoring_group,
            tf_group,
        ]
    )
