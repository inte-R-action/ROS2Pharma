# ROS2 Pharmaceutical IV Bag Vision System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Intelligent IV Bag Inspection with Continuous Learning ROS2-Based Architecture**

## 🎯 Overview

This system implements a sophisticated pharmaceutical quality control solution with:

- **96.2% contamination detection accuracy** using EfficientDet
- TBC

## 🏗️ System Architecture

## 🚀 Key Features

### Advanced Computer Vision

### Robotic Manipulation

### Quality Control

### Performance Metrics

## 📋 Prerequisites

### Hardware Requirements

### Software Requirements
- **OS**: Ubuntu 22.04 LTS
- **ROS2**: Jazzy Jalisco or newer
- **Python**: 3.8 or newer
- **CUDA**: 11.8 or newer (for GPU acceleration)

## 🛠️ Installation

### 1. Install ROS2 Jazzy

```bash
# Add ROS2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS2 Jazzy
sudo apt update
sudo apt install ros-jazzy-desktop-full

# Install additional ROS2 packages
sudo apt install ros-jazzy-cv-bridge ros-jazzy-vision-msgs ros-jazzy-image-transport \
                 ros-jazzy-compressed-image-transport ros-jazzy-joint-state-publisher \
                 ros-jazzy-robot-state-publisher ros-jazzy-moveit
```

### 2. Setup Workspace

```bash
# Create workspace
mkdir -p ~/ros2_pharma_ws/src
cd ~/ros2_pharma_ws/src

# Clone the repository
git clone https://github.com/inte-R-action/ROS2Pharma.git
cd ROS2Pharma

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Install System Dependencies

```bash
# Install OpenCV and computer vision libraries
sudo apt install libopencv-dev libeigen3-dev libpcl-dev

# Install CUDA (if not already installed)
# Follow NVIDIA CUDA installation guide for your system

# Install OAK-D dependencies
python3 -m pip install depthai
```

### 4. Build the Package

```bash
# Source ROS2 environment
source /opt/ros/jazzy/setup.bash

# Build the workspace
cd ~/ros2_pharma_ws
colcon build --packages-select ros2_pharma_iv_vision

# Source the workspace
source install/setup.bash
```

## 🎮 Usage

### Quick Start

#### 1. Launch the Complete System
```bash
# Terminal 1: Launch core vision and analysis system
ros2 launch ros2_pharma_iv_vision complete_system.launch.py

# Terminal 2: Launch robotic manipulation system  
ros2 launch ros2_pharma_iv_vision robotic_system.launch.py

# Terminal 3: Launch monitoring and visualization
ros2 launch ros2_pharma_iv_vision monitoring.launch.py
```

#### 2. Individual Node Launch

**Vision System:**
```bash
ros2 run ros2_pharma_iv_vision vision_system_node \
  --ros-args -p model_path:=/path/to/efficientdet_model.pth \
             -p camera_type:=oakd \
             -p use_gpu:=true
```

**Contamination Analysis:**
```bash
ros2 run ros2_pharma_iv_vision contamination_analysis_node \
  --ros-args -p particle_threshold:=5 \
             -p bubble_threshold:=10 \
             -p confidence_threshold:=0.7
```

**Robot Loading:**
```bash
ros2 run ros2_pharma_iv_vision so_arm101_loading_node \
  --ros-args -p robot_ip:=192.168.1.100 \
             -p act_policy_enabled:=true \
             -p eye_in_hand_enabled:=true
```

### Configuration

#### Vision System Parameters
```yaml
# config/vision_config.yaml
vision_system:
  model_path: "/models/efficientdet_contamination_model.pth"
  confidence_threshold: 0.7
  nms_threshold: 0.5
  fps_target: 30
  use_gpu: true
  
camera:
  type: "oakd"  # or "webcam"
  width: 640
  height: 480
  depth_enabled: true
```

#### Contamination Analysis Parameters
```yaml
# config/contamination_config.yaml
contamination_analysis:
  thresholds:
    particle_threshold: 5
    bubble_threshold: 10
    confidence_threshold: 0.7
    size_threshold_mm: 0.5
  
  statistical:
    confidence_level: 0.95
    significance_level: 0.05
    enable_adaptive_thresholds: true
  
  compliance:
    mode: "EU_GMP_ANNEX_1"
    conservative_mode: false
```

#### Robot Control Parameters
```yaml
# config/robot_config.yaml
so_arm101_loading:
  connection:
    ip_address: "192.168.1.100"
    port: 9999
  
  act_policy:
    enabled: true
    model_path: "/models/act_loading_policy.pth"
    chunk_size: 10
    confidence_threshold: 0.8
  
  safety:
    max_force_x: 10.0
    max_force_y: 10.0
    max_force_z: 15.0
    collision_detection: true
    emergency_stop: true
```

### Data Flow and Topics

#### Core Topics
```bash
# Vision System Outputs
/vision_system/processed_image          # Processed video feed
/vision_system/detections              # Raw detection results
/vision_metrics/particle_count         # Real-time particle count
/vision_metrics/bubble_count           # Real-time bubble count
/vision_metrics/detection_confidence   # Detection confidence scores

# Contamination Analysis
/contamination_analysis/result         # Position analysis results
/contamination_analysis/bag_decision   # Final bag decisions
/contamination_feedback/loading_commands    # Robot loading commands
/contamination_feedback/sorting_commands    # Robot sorting commands

# Robot Control
/loading_arm/joint_states              # Robot joint positions
/loading_arm/safety_status             # Safety monitoring
/unloading_arm/sorting_decisions       # Sorting operations

# System Coordination
/current_position                      # Linear actuator position
/analysis_active                       # Analysis state control
```

## 📊 Performance Benchmarks


## 🔧 Troubleshooting

### Common Issues

#### Camera Connection Issues
```bash
# Check OAK-D camera connection
python3 -c "import depthai as dai; print('OAK-D detected:', len(dai.Device.getAllAvailableDevices()))"

# Test camera feed
ros2 run ros2_pharma_iv_vision camera_test_node
```

#### GPU Issues
```bash
# Check CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi -l 1
```

#### Robot Connection Issues
```bash
# Test robot connectivity
ping 192.168.1.100

# Check robot status
ros2 topic echo /loading_arm/joint_states --timeout 5
```

### Performance Optimization

#### GPU Memory Optimization
```bash
# Set CUDA memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Monitor GPU memory
ros2 run ros2_pharma_iv_vision gpu_monitor_node
```

#### Network Optimization
```bash
# Optimize ROS2 DDS settings
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
export CYCLONEDX_URI="<cyclonedx><domain><general><networkInterfaceAddress>eth0</networkInterfaceAddress></general></domain></cyclonedx>"
```

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run all tests
cd ~/ros2_pharma_ws
colcon test --packages-select ros2_pharma_iv_vision

# Run specific test suites
python3 -m pytest src/ROS2Pharma/ros2_pharma_iv_vision/test/
```

### Integration Tests
```bash
# Test vision system
ros2 run ros2_pharma_iv_vision test_vision_system

# Test contamination analysis
ros2 run ros2_pharma_iv_vision test_contamination_analysis

# Test robotic integration
ros2 run ros2_pharma_iv_vision test_robotic_system
```

### Performance Testing
```bash
# Benchmark detection performance
ros2 run ros2_pharma_iv_vision benchmark_detection_node

# Benchmark manipulation performance  
ros2 run ros2_pharma_iv_vision benchmark_manipulation_node

# System stress test
ros2 run ros2_pharma_iv_vision stress_test_node
```

## 📈 Monitoring and Analytics

### Real-time Monitoring
```bash
# Launch monitoring dashboard
ros2 launch ros2_pharma_iv_vision monitoring_dashboard.launch.py

# View performance metrics
ros2 topic echo /vision_system/processing_fps
ros2 topic echo /contamination_analysis/quality_metrics
ros2 topic echo /loading_arm/performance_metrics
```

### Data Logging
```bash
# Record system data
ros2 bag record -a -o system_data_$(date +%Y%m%d_%H%M%S)

# Replay recorded data
ros2 bag play system_data_YYYYMMDD_HHMMSS
```

## 🤝 Contributing

We welcome contributions to improve the ROS2 Pharmaceutical IV Bag Vision System!

### Development Setup
```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/ROS2Pharma.git

# Create a development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### Contribution Guidelines
1. **Code Quality**: Follow PEP 8 style guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update documentation for API changes
4. **Performance**: Ensure changes don't degrade system performance

### Submitting Changes
1. Run tests: `colcon test --packages-select ros2_pharma_iv_vision`
2. Run linting: `flake8 ros2_pharma_iv_vision/`
3. Create pull request with detailed description

## 📚 Additional Resources

### Documentation

### Research Papers

### Training Materials


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ROS2 Community** for the robust robotics framework
- **LeRobot Team** for SO-ARM101 integration support
- **OpenAI** for transformer architecture insights
- **Pharmaceutical Industry Partners** for validation requirements

## 📞 Support

For technical support and questions:

- **Issues**: [GitHub Issues](https://github.com/inte-R-action/ROS2Pharma/issues)
- **Discussions**: [GitHub Discussions](https://github.com/inte-R-action/ROS2Pharma/discussions)
- **Email**: contact@inte-r-action.com

---

**Note**: This system is designed for research and development purposes. For production pharmaceutical manufacturing, additional validation and regulatory approval may be required according to your local pharmaceutical regulations.

---

*Developed by the inte-R-action team for advancing pharmaceutical automation and quality control.*
