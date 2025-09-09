#!/usr/bin/env python3
"""
Advanced Contamination Analysis Node for IV Bag Quality Control

This node implements sophisticated contamination analysis with statistical validation,
adaptive thresholds, and comprehensive quality metrics for pharmaceutical manufacturing.

Features:

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
from std_msgs.msg import String, Float32, Bool, Int32
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
from collections import defaultdict, deque
import threading
import scipy.stats as stats
from dataclasses import dataclass, asdict

from ..utils.statistical_analyzer import StatisticalAnalyzer
from ..utils.threshold_manager import AdaptiveThresholdManager
from ..utils.quality_metrics import QualityMetricsCalculator


@dataclass
class ContaminationMetrics:
    """Structured contamination metrics for statistical analysis."""
    particle_count: int
    bubble_count: int
    confidence_score: float
    particle_sizes: List[float]
    bubble_sizes: List[float]
    spatial_distribution: Dict
    temporal_pattern: Dict
    timestamp: float


@dataclass
class AnalysisResult:
    """Comprehensive analysis result with statistical validation."""
    iv_bag_id: int
    position_type: str
    decision: str  # ACCEPT, REJECT, REVIEW
    confidence_level: float
    contamination_probability: float
    statistical_significance: float
    quality_score: float
    risk_assessment: str
    metrics: ContaminationMetrics
    metadata: Dict


class ContaminationAnalysisNode(Node):
    """
    Advanced contamination analysis node with statistical validation.

    """
    
    def __init__(self):
        super().__init__('contamination_analysis_node')
        
        # Declare comprehensive parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize core components
        self._initialize_analyzers()
        self._initialize_state()
        
        # Create ROS2 communication interfaces
        self._create_subscribers()
        self._create_publishers()
        self._create_timers()
        
        # Setup results storage
        self._setup_results_storage()
        
        self.get_logger().info('Advanced Contamination Analysis Node initialized')
        self.get_logger().info(f'Quality thresholds: Particles={self.particle_threshold}, '
                             f'Bubbles={self.bubble_threshold}, Confidence={self.confidence_threshold}')
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters with comprehensive defaults."""
        # Basic contamination thresholds
        self.declare_parameter('particle_threshold', 5)
        self.declare_parameter('bubble_threshold', 10)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('size_threshold_mm', 0.5)
        
        # Statistical analysis parameters
        self.declare_parameter('statistical_confidence', 0.95)
        self.declare_parameter('significance_level', 0.05)
        self.declare_parameter('min_samples_for_stats', 10)
        self.declare_parameter('enable_adaptive_thresholds', True)
        
        # Analysis timing parameters
        self.declare_parameter('analysis_duration', 6.0)
        self.declare_parameter('stabilization_time', 1.0)
        self.declare_parameter('max_analysis_time', 10.0)
        
        # Quality control parameters
        self.declare_parameter('enable_quality_metrics', True)
        self.declare_parameter('quality_score_weight_particles', 0.4)
        self.declare_parameter('quality_score_weight_bubbles', 0.3)
        self.declare_parameter('quality_score_weight_distribution', 0.3)
        
        # Environmental adaptation
        self.declare_parameter('enable_environmental_correction', True)
        self.declare_parameter('lighting_condition_factor', 1.0)
        self.declare_parameter('temperature_correction_enabled', False)
        
        # Output and storage
        self.declare_parameter('save_results', True)
        self.declare_parameter('results_directory', '/tmp/contamination_analysis')
        self.declare_parameter('enable_batch_reporting', True)
        self.declare_parameter('compliance_mode', 'EU_GMP_ANNEX_1')
        
        # Decision support
        self.declare_parameter('conservative_mode', False)
        self.declare_parameter('enable_manual_override', True)
        self.declare_parameter('require_dual_confirmation', False)
    
    def _get_parameters(self):
        """Get all parameter values."""
        # Basic thresholds
        self.particle_threshold = self.get_parameter('particle_threshold').value
        self.bubble_threshold = self.get_parameter('bubble_threshold').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.size_threshold_mm = self.get_parameter('size_threshold_mm').value
        
        # Statistical parameters
        self.statistical_confidence = self.get_parameter('statistical_confidence').value
        self.significance_level = self.get_parameter('significance_level').value
        self.min_samples_for_stats = self.get_parameter('min_samples_for_stats').value
        self.enable_adaptive_thresholds = self.get_parameter('enable_adaptive_thresholds').value
        
        # Timing parameters
        self.analysis_duration = self.get_parameter('analysis_duration').value
        self.stabilization_time = self.get_parameter('stabilization_time').value
        self.max_analysis_time = self.get_parameter('max_analysis_time').value
        
        # Quality control
        self.enable_quality_metrics = self.get_parameter('enable_quality_metrics').value
        self.quality_weights = {
            'particles': self.get_parameter('quality_score_weight_particles').value,
            'bubbles': self.get_parameter('quality_score_weight_bubbles').value,
            'distribution': self.get_parameter('quality_score_weight_distribution').value
        }
        
        # Environmental
        self.enable_environmental_correction = self.get_parameter('enable_environmental_correction').value
        self.lighting_factor = self.get_parameter('lighting_condition_factor').value
        self.temperature_correction = self.get_parameter('temperature_correction_enabled').value
        
        # Output
        self.save_results = self.get_parameter('save_results').value
        self.results_directory = Path(self.get_parameter('results_directory').value)
        self.enable_batch_reporting = self.get_parameter('enable_batch_reporting').value
        self.compliance_mode = self.get_parameter('compliance_mode').value
        
        # Decision support
        self.conservative_mode = self.get_parameter('conservative_mode').value
        self.enable_manual_override = self.get_parameter('enable_manual_override').value
        self.require_dual_confirmation = self.get_parameter('require_dual_confirmation').value
    
    def _initialize_analyzers(self):
        """Initialize analysis components."""
        # Statistical analyzer for significance testing
        self.statistical_analyzer = StatisticalAnalyzer(
            confidence_level=self.statistical_confidence,
            significance_level=self.significance_level
        )
        
        # Adaptive threshold manager
        if self.enable_adaptive_thresholds:
            self.threshold_manager = AdaptiveThresholdManager(
                base_thresholds={
                    'particles': self.particle_threshold,
                    'bubbles': self.bubble_threshold,
                    'confidence': self.confidence_threshold
                },
                adaptation_rate=0.1
            )
        else:
            self.threshold_manager = None
        
        # Quality metrics calculator
        if self.enable_quality_metrics:
            self.quality_calculator = QualityMetricsCalculator(
                weights=self.quality_weights,
                compliance_mode=self.compliance_mode
            )
        else:
            self.quality_calculator = None
    
    def _initialize_state(self):
        """Initialize node state variables."""
        # Current analysis state
        self.current_position = 0.0
        self.is_analyzing = False
        self.current_iv_bag = None
        self.current_position_type = None
        
        # Data collection buffers
        self.current_metrics = deque(maxlen=1000)  # Ring buffer for real-time data
        self.analysis_start_time = None
        self.stabilization_complete = False
        
        # Results storage
        self.iv_bag_results = {}
        self.batch_statistics = defaultdict(list)
        
        # Session tracking
        self.analysis_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now(),
            'total_bags_analyzed': 0,
            'accepted_bags': 0,
            'rejected_bags': 0,
            'review_required_bags': 0,
            'compliance_mode': self.compliance_mode
        }
        
        # Position mapping for 5 IV bags with dual positions
        self.position_mapping = {
            4.0: {'bag': 1, 'position': 'position_1'},
            16.0: {'bag': 1, 'position': 'position_2'},
            6.0: {'bag': 2, 'position': 'position_1'},
            18.0: {'bag': 2, 'position': 'position_2'},
            8.0: {'bag': 3, 'position': 'position_1'},
            20.0: {'bag': 3, 'position': 'position_2'},
            10.0: {'bag': 4, 'position': 'position_1'},
            22.0: {'bag': 4, 'position': 'position_2'},
            12.0: {'bag': 5, 'position': 'position_1'},
            24.0: {'bag': 5, 'position': 'position_2'}
        }
        
        # Thread safety
        self.data_lock = threading.Lock()
    
    def _create_subscribers(self):
        """Create all ROS2 subscribers."""
        # Core system inputs
        self.position_sub = self.create_subscription(
            Float32, '/current_position', self.position_callback, 10)
        
        self.analysis_active_sub = self.create_subscription(
            Bool, '/analysis_active', self.analysis_active_callback, 10)
        
        # Vision system metrics
        self.particle_count_sub = self.create_subscription(
            Int32, '/vision_metrics/particle_count', self.particle_count_callback, 10)
        
        self.bubble_count_sub = self.create_subscription(
            Int32, '/vision_metrics/bubble_count', self.bubble_count_callback, 10)
        
        self.detection_confidence_sub = self.create_subscription(
            Float32, '/vision_metrics/detection_confidence', self.detection_confidence_callback, 10)
        
        # Enhanced detection results for detailed analysis
        self.detections_sub = self.create_subscription(
            Detection2DArray, '/vision_system/detections', self.detections_callback, 10)
        
        # Manual override commands
        if self.enable_manual_override:
            self.manual_override_sub = self.create_subscription(
                String, '/contamination_analysis/manual_override', 
                self.manual_override_callback, 10)
    
    def _create_publishers(self):
        """Create all ROS2 publishers."""
        # Core analysis results
        self.analysis_result_pub = self.create_publisher(
            String, '/contamination_analysis/result', 10)
        
        self.bag_decision_pub = self.create_publisher(
            String, '/contamination_analysis/bag_decision', 10)
        
        self.quality_metrics_pub = self.create_publisher(
            String, '/contamination_analysis/quality_metrics', 10)
        
        # Statistical analysis results
        self.statistical_analysis_pub = self.create_publisher(
            String, '/contamination_analysis/statistical_result', 10)
        
        # Feedback for robotic system
        self.loading_commands_pub = self.create_publisher(
            String, '/contamination_feedback/loading_commands', 10)
        
        self.sorting_commands_pub = self.create_publisher(
            String, '/contamination_feedback/sorting_commands', 10)
        
        # Batch and compliance reporting
        if self.enable_batch_reporting:
            self.batch_report_pub = self.create_publisher(
                String, '/contamination_analysis/batch_report', 10)
        
        # System status and health
        self.analysis_status_pub = self.create_publisher(
            String, '/contamination_analysis/status', 10)
    
    def _create_timers(self):
        """Create ROS2 timers for periodic operations."""
        # Continuous analysis timer (10 Hz for real-time processing)
        self.create_timer(0.1, self.continuous_analysis_callback)
        
        # Status reporting timer (1 Hz)
        self.create_timer(1.0, self.publish_status_callback)
        
        # Batch reporting timer (every 5 minutes)
        if self.enable_batch_reporting:
            self.create_timer(300.0, self.publish_batch_report_callback)
    
    def _setup_results_storage(self):
        """Setup results storage and directory structure."""
        if self.save_results:
            self.results_directory.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for organized storage
            (self.results_directory / 'individual_results').mkdir(exist_ok=True)
            (self.results_directory / 'batch_reports').mkdir(exist_ok=True)
            (self.results_directory / 'statistical_analysis').mkdir(exist_ok=True)
            (self.results_directory / 'compliance_records').mkdir(exist_ok=True)
    
    def position_callback(self, msg: Float32):
        """Handle position updates with enhanced tracking."""
        self.current_position = msg.data
        
        # Check if this is an analysis position
        if self.current_position in self.position_mapping:
            position_info = self.position_mapping[self.current_position]
            self.current_iv_bag = position_info['bag']
            self.current_position_type = position_info['position']
            
            self.get_logger().info(
                f'Positioned at IV Bag {self.current_iv_bag} - '
                f'{self.current_position_type} (pos: {self.current_position})'
            )
        else:
            self.current_iv_bag = None
            self.current_position_type = None
    
    def analysis_active_callback(self, msg: Bool):
        """Handle analysis state changes with enhanced control."""
        was_analyzing = self.is_analyzing
        self.is_analyzing = msg.data
        
        if self.is_analyzing and not was_analyzing:
            self._start_position_analysis()
        elif not self.is_analyzing and was_analyzing:
            self._complete_position_analysis()
    
    def particle_count_callback(self, msg: Int32):
        """Handle particle count updates."""
        if self._is_collecting_data():
            with self.data_lock:
                # Create contamination metrics entry
                metrics = ContaminationMetrics(
                    particle_count=msg.data,
                    bubble_count=0,  # Will be updated by bubble callback
                    confidence_score=0.0,  # Will be updated by confidence callback
                    particle_sizes=[],  # Will be populated from detailed detections
                    bubble_sizes=[],
                    spatial_distribution={},
                    temporal_pattern={},
                    timestamp=time.time()
                )
                self._update_current_metrics(metrics)
    
    def bubble_count_callback(self, msg: Int32):
        """Handle bubble count updates."""
        if self._is_collecting_data():
            with self.data_lock:
                # Update the most recent metrics entry
                if self.current_metrics:
                    self.current_metrics[-1].bubble_count = msg.data
    
    def detection_confidence_callback(self, msg: Float32):
        """Handle detection confidence updates."""
        if self._is_collecting_data():
            with self.data_lock:
                # Update the most recent metrics entry
                if self.current_metrics:
                    self.current_metrics[-1].confidence_score = msg.data
    
    def detections_callback(self, msg: Detection2DArray):
        """Handle detailed detection results for advanced analysis."""
        if self._is_collecting_data():
            # Extract detailed information from detections
            particle_sizes = []
            bubble_sizes = []
            spatial_info = {'centers': [], 'areas': []}
            
            for detection in msg.detections:
                # Extract size information from bounding box
                width = detection.bbox.size_x
                height = detection.bbox.size_y
                area = width * height
                
                # Estimate actual size (requires calibration)
                estimated_size_mm = self._estimate_physical_size(area)
                
                # Extract spatial information
                center_x = detection.bbox.center.x
                center_y = detection.bbox.center.y
                spatial_info['centers'].append((center_x, center_y))
                spatial_info['areas'].append(area)
                
                # Classify based on detection class (simplified)
                # In real implementation, use detection.results
                if estimated_size_mm < 2.0:  # Particles typically smaller
                    particle_sizes.append(estimated_size_mm)
                else:  # Bubbles typically larger
                    bubble_sizes.append(estimated_size_mm)
            
            # Update current metrics with detailed information
            with self.data_lock:
                if self.current_metrics:
                    latest_metrics = self.current_metrics[-1]
                    latest_metrics.particle_sizes = particle_sizes
                    latest_metrics.bubble_sizes = bubble_sizes
                    latest_metrics.spatial_distribution = spatial_info
    
    def manual_override_callback(self, msg: String):
        """Handle manual override commands."""
        try:
            override_data = json.loads(msg.data)
            override_type = override_data.get('type', '')
            
            if override_type == 'force_accept':
                self._apply_manual_override('ACCEPT', override_data.get('reason', ''))
            elif override_type == 'force_reject':
                self._apply_manual_override('REJECT', override_data.get('reason', ''))
            elif override_type == 'request_review':
                self._apply_manual_override('REVIEW', override_data.get('reason', ''))
            
        except Exception as e:
            self.get_logger().error(f'Error processing manual override: {e}')
    
    def continuous_analysis_callback(self):
        """Continuous analysis processing for real-time feedback."""
        if self.is_analyzing and self._is_collecting_data():
            # Check if stabilization period is complete
            if not self.stabilization_complete:
                elapsed_time = time.time() - self.analysis_start_time
                if elapsed_time > self.stabilization_time:
                    self.stabilization_complete = True
                    self.get_logger().info('Analysis stabilization period complete')
            
            # Perform real-time quality assessment
            if self.stabilization_complete and len(self.current_metrics) > 5:
                self._perform_realtime_assessment()
    
    def publish_status_callback(self):
        """Publish system status and health information."""
        status = {
            'node_status': 'active',
            'current_position': self.current_position,
            'is_analyzing': self.is_analyzing,
            'current_iv_bag': self.current_iv_bag,
            'stabilization_complete': self.stabilization_complete,
            'data_points_collected': len(self.current_metrics),
            'session_stats': self.analysis_session,
            'adaptive_thresholds_active': self.enable_adaptive_thresholds,
            'timestamp': datetime.now().isoformat()
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status, indent=2)
        self.analysis_status_pub.publish(status_msg)
    
    def publish_batch_report_callback(self):
        """Publish periodic batch analysis reports."""
        if self.batch_statistics:
            report = self._generate_batch_report()
            report_msg = String()
            report_msg.data = json.dumps(report, indent=2)
            self.batch_report_pub.publish(report_msg)
    
    def _start_position_analysis(self):
        """Start analysis for current position with enhanced initialization."""
        if self.current_iv_bag is not None:
            self.get_logger().info(
                f'Starting enhanced contamination analysis for IV Bag '
                f'{self.current_iv_bag} - {self.current_position_type}'
            )
            
            # Reset analysis state
            with self.data_lock:
                self.current_metrics.clear()
                self.analysis_start_time = time.time()
                self.stabilization_complete = False
            
            # Update adaptive thresholds if enabled
            if self.threshold_manager:
                self.threshold_manager.update_environmental_conditions({
                    'lighting_factor': self.lighting_factor,
                    'position': self.current_position
                })
    
    def _complete_position_analysis(self):
        """Complete analysis for current position with statistical validation."""
        if self.current_iv_bag is not None and self.analysis_start_time is not None:
            self.get_logger().info(
                f'Completing analysis for IV Bag {self.current_iv_bag} - '
                f'{self.current_position_type}'
            )
            
            # Perform comprehensive analysis
            analysis_result = self._perform_comprehensive_analysis()
            
            # Store results
            self._store_position_result(analysis_result)
            
            # Check if bag analysis is complete
            if self._is_bag_analysis_complete():
                self._perform_bag_level_analysis()
            
            # Publish results
            self._publish_analysis_results(analysis_result)
    
    def _perform_comprehensive_analysis(self) -> AnalysisResult:
        """Perform comprehensive statistical analysis of collected data."""
        with self.data_lock:
            metrics_data = list(self.current_metrics)
        
        if not metrics_data:
            return self._create_empty_result()
        
        # Extract arrays for statistical analysis
        particle_counts = [m.particle_count for m in metrics_data]
        bubble_counts = [m.bubble_count for m in metrics_data]
        confidence_scores = [m.confidence_score for m in metrics_data if m.confidence_score > 0]
        
        # Basic statistics
        total_particles = sum(particle_counts)
        total_bubbles = sum(bubble_counts)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Statistical significance testing
        statistical_result = self.statistical_analyzer.analyze_contamination_significance(
            particle_counts, bubble_counts, confidence_scores
        )
        
        # Quality metrics calculation
        quality_score = 1.0
        if self.quality_calculator:
            quality_score = self.quality_calculator.calculate_quality_score(
                metrics_data, self.current_position_type
            )
        
        # Decision making with statistical validation
        decision, confidence_level = self._make_contamination_decision(
            total_particles, total_bubbles, avg_confidence,
            statistical_result, quality_score
        )
        
        # Risk assessment
        risk_level = self._assess_contamination_risk(
            total_particles, total_bubbles, statistical_result
        )
        
        # Create comprehensive result
        final_metrics = ContaminationMetrics(
            particle_count=total_particles,
            bubble_count=total_bubbles,
            confidence_score=avg_confidence,
            particle_sizes=self._aggregate_particle_sizes(metrics_data),
            bubble_sizes=self._aggregate_bubble_sizes(metrics_data),
            spatial_distribution=self._analyze_spatial_distribution(metrics_data),
            temporal_pattern=self._analyze_temporal_patterns(metrics_data),
            timestamp=time.time()
        )
        
        result = AnalysisResult(
            iv_bag_id=self.current_iv_bag,
            position_type=self.current_position_type,
            decision=decision,
            confidence_level=confidence_level,
            contamination_probability=statistical_result.get('contamination_probability', 0.0),
            statistical_significance=statistical_result.get('significance_level', 0.0),
            quality_score=quality_score,
            risk_assessment=risk_level,
            metrics=final_metrics,
            metadata={
                'analysis_duration': time.time() - self.analysis_start_time,
                'sample_count': len(metrics_data),
                'position': self.current_position,
                'thresholds_used': self._get_current_thresholds(),
                'statistical_test_results': statistical_result,
                'compliance_mode': self.compliance_mode
            }
        )
        
        return result
    
    def _make_contamination_decision(self, particles: int, bubbles: int, 
                                   confidence: float, statistical_result: Dict,
                                   quality_score: float) -> Tuple[str, float]:
        """Make contamination decision with statistical validation."""
        
        # Get current thresholds (adaptive or fixed)
        thresholds = self._get_current_thresholds()
        
        # Basic threshold checks
        particle_exceeded = particles > thresholds['particles']
        bubble_exceeded = bubbles > thresholds['bubbles']
        low_confidence = confidence < thresholds['confidence']
        
        # Statistical significance check
        statistically_significant = statistical_result.get('is_significant', False)
        contamination_probability = statistical_result.get('contamination_probability', 0.0)
        
        # Quality score consideration
        quality_acceptable = quality_score > 0.7  # Configurable threshold
        
        # Decision logic with confidence levels
        if self.conservative_mode:
            # Conservative mode: err on side of caution
            if particle_exceeded or bubble_exceeded or contamination_probability > 0.3:
                return 'REJECT', max(contamination_probability, 0.8)
            elif low_confidence or not quality_acceptable:
                return 'REVIEW', 0.6
            else:
                return 'ACCEPT', min(1.0 - contamination_probability, 0.9)
        else:
            # Standard mode: balanced decision making
            if (particle_exceeded or bubble_exceeded) and statistically_significant:
                return 'REJECT', contamination_probability
            elif (particle_exceeded or bubble_exceeded) and not statistically_significant:
                return 'REVIEW', 0.5
            elif low_confidence and (particles > 0 or bubbles > 0):
                return 'REVIEW', 0.4
            else:
                return 'ACCEPT', min(1.0 - contamination_probability, 0.95)
    
    def _assess_contamination_risk(self, particles: int, bubbles: int, 
                                 statistical_result: Dict) -> str:
        """Assess contamination risk level for quality management."""
        total_contaminants = particles + bubbles
        significance = statistical_result.get('significance_level', 0.0)
        
        if total_contaminants == 0:
            return 'MINIMAL'
        elif total_contaminants <= 2 and significance < 0.1:
            return 'LOW'
        elif total_contaminants <= 5 or significance < 0.05:
            return 'MEDIUM'
        elif total_contaminants <= 15:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _get_current_thresholds(self) -> Dict:
        """Get current thresholds (adaptive or fixed)."""
        if self.threshold_manager:
            return self.threshold_manager.get_current_thresholds()
        else:
            return {
                'particles': self.particle_threshold,
                'bubbles': self.bubble_threshold,
                'confidence': self.confidence_threshold
            }
    
    def _is_collecting_data(self) -> bool:
        """Check if currently collecting analysis data."""
        return (self.is_analyzing and 
                self.current_iv_bag is not None and 
                self.analysis_start_time is not None)
    
    def _update_current_metrics(self, metrics: ContaminationMetrics):
        """Update current metrics buffer."""
        self.current_metrics.append(metrics)
    
    def _estimate_physical_size(self, pixel_area: float) -> float:
        """Estimate physical size from pixel area (requires calibration)."""
        # Placeholder implementation - requires proper camera calibration
        # This would use camera intrinsics and depth information
        pixels_per_mm = 10.0  # Example calibration value
        area_mm2 = pixel_area / (pixels_per_mm ** 2)
        return np.sqrt(area_mm2)  # Approximate diameter
    
    def _perform_realtime_assessment(self):
        """Perform real-time quality assessment during analysis."""
        # Placeholder for real-time feedback
        pass
    
    def _store_position_result(self, result: AnalysisResult):
        """Store position analysis result."""
        if self.current_iv_bag not in self.iv_bag_results:
            self.iv_bag_results[self.current_iv_bag] = {}
        
        self.iv_bag_results[self.current_iv_bag][self.current_position_type] = result
        
        # Update batch statistics
        self.batch_statistics['decisions'].append(result.decision)
        self.batch_statistics['quality_scores'].append(result.quality_score)
        self.batch_statistics['contamination_probabilities'].append(
            result.contamination_probability
        )
    
    def _is_bag_analysis_complete(self) -> bool:
        """Check if both positions analyzed for current bag."""
        return (self.current_iv_bag in self.iv_bag_results and 
                len(self.iv_bag_results[self.current_iv_bag]) == 2)
    
    def _perform_bag_level_analysis(self):
        """Perform comprehensive bag-level analysis."""
        bag_results = self.iv_bag_results[self.current_iv_bag]
        pos1_result = bag_results['position_1']
        pos2_result = bag_results['position_2']
        
        # Combined decision logic
        final_decision = self._combine_position_decisions(pos1_result, pos2_result)
        
        # Update session statistics
        self.analysis_session['total_bags_analyzed'] += 1
        if final_decision == 'ACCEPT':
            self.analysis_session['accepted_bags'] += 1
        elif final_decision == 'REJECT':
            self.analysis_session['rejected_bags'] += 1
        else:
            self.analysis_session['review_required_bags'] += 1
        
        # Publish bag-level decision
        self._publish_bag_decision(final_decision, pos1_result, pos2_result)
        
        # Generate robotic control commands
        self._generate_robotic_commands(final_decision)
    
    def _combine_position_decisions(self, pos1: AnalysisResult, 
                                   pos2: AnalysisResult) -> str:
        """Combine decisions from both positions."""
        decisions = [pos1.decision, pos2.decision]
        
        # Hierarchical decision logic
        if 'REJECT' in decisions:
            return 'REJECT'
        elif 'REVIEW' in decisions:
            return 'REVIEW'
        else:
            return 'ACCEPT'
    
    def _publish_analysis_results(self, result: AnalysisResult):
        """Publish comprehensive analysis results."""
        # Position-level result
        result_msg = String()
        result_msg.data = json.dumps(asdict(result), indent=2, default=str)
        self.analysis_result_pub.publish(result_msg)
        
        # Quality metrics
        if self.enable_quality_metrics:
            quality_data = {
                'quality_score': result.quality_score,
                'risk_assessment': result.risk_assessment,
                'confidence_level': result.confidence_level,
                'statistical_significance': result.statistical_significance
            }
            quality_msg = String()
            quality_msg.data = json.dumps(quality_data, indent=2)
            self.quality_metrics_pub.publish(quality_msg)
        
        # Statistical analysis details
        statistical_data = {
            'contamination_probability': result.contamination_probability,
            'statistical_significance': result.statistical_significance,
            'confidence_interval': result.metadata.get('statistical_test_results', {}),
            'decision_confidence': result.confidence_level
        }
        stat_msg = String()
        stat_msg.data = json.dumps(statistical_data, indent=2)
        self.statistical_analysis_pub.publish(stat_msg)
        
        self.get_logger().info(
            f'Analysis complete - IV Bag {result.iv_bag_id} {result.position_type}: '
            f'{result.decision} (confidence: {result.confidence_level:.3f}, '
            f'quality: {result.quality_score:.3f})'
        )
    
    def _publish_bag_decision(self, decision: str, pos1: AnalysisResult, 
                            pos2: AnalysisResult):
        """Publish final bag-level decision."""
        bag_summary = {
            'iv_bag_id': self.current_iv_bag,
            'final_decision': decision,
            'position_1': asdict(pos1),
            'position_2': asdict(pos2),
            'combined_metrics': {
                'total_particles': pos1.metrics.particle_count + pos2.metrics.particle_count,
                'total_bubbles': pos1.metrics.bubble_count + pos2.metrics.bubble_count,
                'avg_quality_score': (pos1.quality_score + pos2.quality_score) / 2,
                'max_contamination_probability': max(
                    pos1.contamination_probability, pos2.contamination_probability
                )
            },
            'compliance_status': self._check_compliance_status(decision),
            'timestamp': datetime.now().isoformat()
        }
        
        decision_msg = String()
        decision_msg.data = json.dumps(bag_summary, indent=2, default=str)
        self.bag_decision_pub.publish(decision_msg)
        
        self.get_logger().info(f'=== IV BAG {self.current_iv_bag} FINAL DECISION: {decision} ===')
    
    def _generate_robotic_commands(self, decision: str):
        """Generate commands for robotic manipulation system."""
        # Loading system commands
        loading_command = {
            'command_type': 'position_complete',
            'iv_bag_id': self.current_iv_bag,
            'decision': decision,
            'next_action': 'proceed_to_next' if decision != 'REVIEW' else 'hold_for_review'
        }
        
        loading_msg = String()
        loading_msg.data = json.dumps(loading_command)
        self.loading_commands_pub.publish(loading_msg)
        
        # Sorting system commands
        sorting_command = {
            'command_type': 'sort_bag',
            'iv_bag_id': self.current_iv_bag,
            'destination': self._get_sorting_destination(decision),
            'priority': 'high' if decision == 'REJECT' else 'normal',
            'handling_instructions': self._get_handling_instructions(decision)
        }
        
        sorting_msg = String()
        sorting_msg.data = json.dumps(sorting_command)
        self.sorting_commands_pub.publish(sorting_msg)
    
    def _get_sorting_destination(self, decision: str) -> str:
        """Get sorting destination based on decision."""
        destination_map = {
            'ACCEPT': 'approved_container',
            'REJECT': 'rejected_container',
            'REVIEW': 'review_queue'
        }
        return destination_map.get(decision, 'review_queue')
    
    def _get_handling_instructions(self, decision: str) -> str:
        """Get special handling instructions."""
        if decision == 'REJECT':
            return 'handle_as_contaminated'
        elif decision == 'REVIEW':
            return 'gentle_handling_for_reinspection'
        else:
            return 'standard_handling'
    
    def _check_compliance_status(self, decision: str) -> str:
        """Check compliance with regulatory standards."""
        # Simplified compliance check
        if self.compliance_mode == 'EU_GMP_ANNEX_1':
            if decision == 'ACCEPT':
                return 'compliant'
            elif decision == 'REJECT':
                return 'non_compliant_rejected'
            else:
                return 'pending_review'
        return 'not_evaluated'
    
    def _apply_manual_override(self, override_decision: str, reason: str):
        """Apply manual override to current analysis."""
        self.get_logger().info(f'Manual override applied: {override_decision} - {reason}')
        # Implementation would modify current analysis state
    
    def _generate_batch_report(self) -> Dict:
        """Generate comprehensive batch analysis report."""
        total_decisions = len(self.batch_statistics['decisions'])
        if total_decisions == 0:
            return {'status': 'no_data'}
        
        decisions_count = {
            'ACCEPT': self.batch_statistics['decisions'].count('ACCEPT'),
            'REJECT': self.batch_statistics['decisions'].count('REJECT'),
            'REVIEW': self.batch_statistics['decisions'].count('REVIEW')
        }
        
        report = {
            'batch_id': self.analysis_session['session_id'],
            'analysis_period': {
                'start': self.analysis_session['start_time'].isoformat(),
                'end': datetime.now().isoformat()
            },
            'summary_statistics': {
                'total_analyses': total_decisions,
                'decisions_breakdown': decisions_count,
                'acceptance_rate': decisions_count['ACCEPT'] / total_decisions * 100,
                'rejection_rate': decisions_count['REJECT'] / total_decisions * 100,
                'review_rate': decisions_count['REVIEW'] / total_decisions * 100
            },
            'quality_metrics': {
                'avg_quality_score': np.mean(self.batch_statistics['quality_scores']),
                'avg_contamination_probability': np.mean(
                    self.batch_statistics['contamination_probabilities']
                )
            },
            'compliance_summary': {
                'mode': self.compliance_mode,
                'total_compliant': decisions_count['ACCEPT'],
                'total_non_compliant': decisions_count['REJECT']
            }
        }
        
        return report
    
    def _create_empty_result(self) -> AnalysisResult:
        """Create empty result for error cases."""
        return AnalysisResult(
            iv_bag_id=self.current_iv_bag or 0,
            position_type=self.current_position_type or 'unknown',
            decision='REVIEW',
            confidence_level=0.0,
            contamination_probability=0.0,
            statistical_significance=0.0,
            quality_score=0.0,
            risk_assessment='UNKNOWN',
            metrics=ContaminationMetrics(0, 0, 0.0, [], [], {}, {}, time.time()),
            metadata={'error': 'insufficient_data'}
        )
    
    def _aggregate_particle_sizes(self, metrics_data: List[ContaminationMetrics]) -> List[float]:
        """Aggregate particle sizes from all metrics."""
        all_sizes = []
        for metrics in metrics_data:
            all_sizes.extend(metrics.particle_sizes)
        return all_sizes
    
    def _aggregate_bubble_sizes(self, metrics_data: List[ContaminationMetrics]) -> List[float]:
        """Aggregate bubble sizes from all metrics."""
        all_sizes = []
        for metrics in metrics_data:
            all_sizes.extend(metrics.bubble_sizes)
        return all_sizes
    
    def _analyze_spatial_distribution(self, metrics_data: List[ContaminationMetrics]) -> Dict:
        """Analyze spatial distribution patterns."""
        # Placeholder for spatial analysis
        return {'pattern': 'analysis_pending'}
    
    def _analyze_temporal_patterns(self, metrics_data: List[ContaminationMetrics]) -> Dict:
        """Analyze temporal contamination patterns."""
        # Placeholder for temporal analysis
        return {'trend': 'analysis_pending'}


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = ContaminationAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Contamination analysis node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
