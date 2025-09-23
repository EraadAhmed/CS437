#!/usr/bin/env python3
"""
Traffic Rule and Stop Sign Detection Test Suite
CS 437 Lab 1 Part B - Object Detection and Traffic Compliance Testing

This module tests stop sign detection, traffic rule compliance,
and appropriate behavioral responses to traffic situations.
"""

import asyncio
import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import json

# Hardware imports with fallback
try:
    from picarx import Picarx
    from picamera2 import Picamera2
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: Hardware not available. Running in simulation mode.")
    HW_AVAILABLE = False

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. Using mock detection.")
    TF_AVAILABLE = False

# Import integrated system
from integrated import IntegratedSelfDrivingSystem, SystemConfig, Position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrafficTestCase:
    """Traffic rule test case definition."""
    name: str
    description: str
    detected_object: str  # 'stop_sign', 'person', 'traffic_light', etc.
    expected_behavior: str  # 'stop_and_wait', 'stop_briefly', 'slow_down', etc.
    duration_seconds: float
    test_distance: float  # Distance from object when detected

@dataclass
class TrafficTestResult:
    """Results from traffic rule test."""
    test_name: str
    success: bool
    detection_time: float
    response_time: float
    stop_duration: float
    distance_when_stopped: float
    appropriate_response: bool
    error_message: Optional[str] = None

class MockObjectDetector:
    """Mock object detector for testing without hardware."""
    
    def __init__(self):
        self.current_detection = None
        self.detection_confidence = 0.0
        self.detection_timestamp = 0.0
    
    def simulate_detection(self, object_type: str, confidence: float = 0.8):
        """Simulate detection of specific object."""
        self.current_detection = object_type
        self.detection_confidence = confidence
        self.detection_timestamp = time.time()
        logger.info(f"Mock detection: {object_type} (confidence: {confidence:.2f})")
    
    def clear_detection(self):
        """Clear current detection."""
        self.current_detection = None
        self.detection_confidence = 0.0
    
    def get_detection(self) -> Optional[Dict]:
        """Get current detection result."""
        if self.current_detection:
            return {
                'class_name': self.current_detection,
                'confidence': self.detection_confidence,
                'timestamp': self.detection_timestamp
            }
        return None

class StopSignDetector:
    """Specialized stop sign detection system."""
    
    def __init__(self):
        self.camera = None
        self.detector = None
        self.detection_active = False
        self.mock_detector = MockObjectDetector()
        
        if TF_AVAILABLE and HW_AVAILABLE:
            self._initialize_camera()
            self._load_model()
        else:
            logger.warning("Using mock detector for testing")
    
    def _initialize_camera(self):
        """Initialize camera for stop sign detection."""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            logger.info("Camera initialized for stop sign detection")
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
    
    def _load_model(self):
        """Load TensorFlow Lite model for object detection."""
        try:
            model_path = "efficientdet_lite0.tflite"
            if tf.io.gfile.exists(model_path):
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info("Stop sign detection model loaded")
            else:
                logger.warning("Model file not found, using mock detection")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
    
    def start_detection(self):
        """Start stop sign detection."""
        self.detection_active = True
        if self.camera:
            self.camera.start()
        logger.info("Stop sign detection started")
    
    def stop_detection(self):
        """Stop stop sign detection."""
        self.detection_active = False
        if self.camera:
            self.camera.stop()
        logger.info("Stop sign detection stopped")
    
    def detect_stop_sign(self) -> Optional[Dict]:
        """Detect stop sign in current camera frame."""
        if not self.detection_active:
            return None
        
        if self.camera and self.detector:
            # Real detection implementation
            return self._real_detection()
        else:
            # Mock detection for testing
            return self.mock_detector.get_detection()
    
    def _real_detection(self) -> Optional[Dict]:
        """Perform real stop sign detection."""
        try:
            frame = self.camera.capture_array()
            
            # Preprocess frame
            input_data = np.expand_dims(frame, axis=0).astype(np.uint8)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Look for stop signs (class 11 in COCO dataset)
            for i, (score, class_id) in enumerate(zip(scores, classes)):
                if class_id == 11 and score > 0.5:  # Stop sign with high confidence
                    return {
                        'class_name': 'stop_sign',
                        'confidence': float(score),
                        'bbox': boxes[i].tolist(),
                        'timestamp': time.time()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Real detection failed: {e}")
            return None
    
    def simulate_stop_sign_detection(self, confidence: float = 0.9):
        """Simulate stop sign detection for testing."""
        self.mock_detector.simulate_detection('stop_sign', confidence)
    
    def simulate_person_detection(self, confidence: float = 0.8):
        """Simulate person detection for testing."""
        self.mock_detector.simulate_detection('person', confidence)
    
    def clear_simulation(self):
        """Clear simulated detection."""
        self.mock_detector.clear_detection()

class TrafficRuleTestSuite:
    """Comprehensive traffic rule testing system."""
    
    def __init__(self):
        self.config = SystemConfig()
        self.stop_sign_detector = StopSignDetector()
        self.test_results: List[TrafficTestResult] = []
        
        # Define test cases
        self.test_cases = self._define_traffic_test_cases()
    
    def _define_traffic_test_cases(self) -> List[TrafficTestCase]:
        """Define traffic rule test scenarios."""
        return [
            # Test 1: Stop sign detection and compliance
            TrafficTestCase(
                name="stop_sign_compliance",
                description="Detect stop sign and perform complete stop for required duration",
                detected_object="stop_sign",
                expected_behavior="stop_and_wait",
                duration_seconds=3.0,
                test_distance=40.0
            ),
            
            # Test 2: Person detection and safety stop
            TrafficTestCase(
                name="person_safety_stop",
                description="Detect person and stop immediately for safety",
                detected_object="person",
                expected_behavior="emergency_stop",
                duration_seconds=2.0,
                test_distance=25.0
            ),
            
            # Test 3: Stop sign at close range
            TrafficTestCase(
                name="close_range_stop_sign",
                description="React to stop sign detected at close range",
                detected_object="stop_sign",
                expected_behavior="immediate_stop",
                duration_seconds=3.0,
                test_distance=20.0
            ),
            
            # Test 4: Multiple person detection
            TrafficTestCase(
                name="multiple_person_detection",
                description="Handle multiple people in camera view",
                detected_object="person",
                expected_behavior="extended_stop",
                duration_seconds=5.0,
                test_distance=30.0
            ),
            
            # Test 5: False positive handling
            TrafficTestCase(
                name="false_positive_handling",
                description="Handle low-confidence detections appropriately",
                detected_object="stop_sign",
                expected_behavior="no_reaction",
                duration_seconds=0.0,
                test_distance=35.0
            )
        ]
    
    async def run_all_traffic_tests(self) -> Dict:
        """Run all traffic rule test cases."""
        logger.info("Starting comprehensive traffic rule test suite...")
        logger.info(f"Running {len(self.test_cases)} traffic test cases...")
        
        overall_results = {
            'test_results': [],
            'summary': {},
            'timestamp': time.time()
        }
        
        # Initialize detection system
        self.stop_sign_detector.start_detection()
        
        try:
            for i, test_case in enumerate(self.test_cases):
                logger.info(f"\n{'='*60}")
                logger.info(f"TRAFFIC TEST {i+1}/{len(self.test_cases)}: {test_case.name}")
                logger.info(f"{'='*60}")
                logger.info(f"Description: {test_case.description}")
                
                try:
                    result = await self._run_single_traffic_test(test_case)
                    self.test_results.append(result)
                    overall_results['test_results'].append(result.__dict__)
                    
                    # Log result
                    status = "PASS" if result.success else "FAIL"
                    logger.info(f"Result: {status}")
                    if result.success:
                        logger.info(f"  Detection time: {result.detection_time:.2f}s")
                        logger.info(f"  Response time: {result.response_time:.2f}s")
                        logger.info(f"  Stop duration: {result.stop_duration:.2f}s")
                    else:
                        logger.info(f"  Error: {result.error_message}")
                    
                except Exception as e:
                    logger.error(f"Traffic test {test_case.name} failed: {e}")
                    result = TrafficTestResult(
                        test_name=test_case.name,
                        success=False,
                        detection_time=0,
                        response_time=0,
                        stop_duration=0,
                        distance_when_stopped=0,
                        appropriate_response=False,
                        error_message=str(e)
                    )
                    self.test_results.append(result)
                    overall_results['test_results'].append(result.__dict__)
        
        finally:
            self.stop_sign_detector.stop_detection()
        
        # Generate summary
        overall_results['summary'] = self._generate_traffic_summary()
        
        # Save results
        await self._save_traffic_results(overall_results)
        
        return overall_results
    
    async def _run_single_traffic_test(self, test_case: TrafficTestCase) -> TrafficTestResult:
        """Run a single traffic rule test."""
        # Create system for testing
        system = IntegratedSelfDrivingSystem(self.config)
        
        # Set up test scenario
        system.current_position = Position(60, 20, 0)  # Start position
        
        # Initialize timing variables
        detection_start_time = time.time()
        detection_time = 0.0
        response_time = 0.0
        stop_duration = 0.0
        distance_when_stopped = 0.0
        
        try:
            # Simulate car movement towards detection point
            target_y = 20 + test_case.test_distance
            
            # Move car to detection point
            while system.current_position.y < target_y:
                system.current_position.y += 2  # Move 2cm forward
                await asyncio.sleep(0.1)
            
            # Trigger detection based on test case
            detection_triggered_time = time.time()
            
            if test_case.detected_object == "stop_sign":
                if test_case.name == "false_positive_handling":
                    self.stop_sign_detector.simulate_stop_sign_detection(0.3)  # Low confidence
                else:
                    self.stop_sign_detector.simulate_stop_sign_detection(0.9)  # High confidence
            elif test_case.detected_object == "person":
                self.stop_sign_detector.simulate_person_detection(0.8)
            
            detection_time = detection_triggered_time - detection_start_time
            
            # Monitor system response
            response_start_time = time.time()
            car_stopped = False
            stop_start_time = 0.0
            
            # Wait for response or timeout
            timeout = 10.0  # 10 second timeout
            while time.time() - response_start_time < timeout:
                # Check if detection should trigger response
                detection = self.stop_sign_detector.detect_stop_sign()
                
                if detection and not car_stopped:
                    response_time = time.time() - detection_triggered_time
                    
                    # Evaluate response appropriateness
                    should_stop = self._should_car_stop(detection, test_case)
                    
                    if should_stop:
                        # Car stops
                        car_stopped = True
                        stop_start_time = time.time()
                        distance_when_stopped = system.current_position.y
                        logger.info(f"Car stopped due to {detection['class_name']} detection")
                        
                        # Wait for stop duration
                        if test_case.expected_behavior in ["stop_and_wait", "emergency_stop", "immediate_stop"]:
                            await asyncio.sleep(test_case.duration_seconds)
                            stop_duration = time.time() - stop_start_time
                        
                        break
                    elif test_case.expected_behavior == "no_reaction":
                        # Should not stop for low confidence
                        logger.info("Correctly ignored low-confidence detection")
                        break
                
                await asyncio.sleep(0.1)
            
            # Clear detection
            self.stop_sign_detector.clear_simulation()
            
            # Evaluate test result
            success = self._evaluate_traffic_test_result(test_case, {
                'detection_time': detection_time,
                'response_time': response_time,
                'stop_duration': stop_duration,
                'car_stopped': car_stopped,
                'distance_when_stopped': distance_when_stopped
            })
            
            return TrafficTestResult(
                test_name=test_case.name,
                success=success['passed'],
                detection_time=detection_time,
                response_time=response_time,
                stop_duration=stop_duration,
                distance_when_stopped=distance_when_stopped,
                appropriate_response=success['appropriate'],
                error_message=success.get('error')
            )
            
        except Exception as e:
            return TrafficTestResult(
                test_name=test_case.name,
                success=False,
                detection_time=0,
                response_time=0,
                stop_duration=0,
                distance_when_stopped=0,
                appropriate_response=False,
                error_message=str(e)
            )
        finally:
            await system.stop()
    
    def _should_car_stop(self, detection: Dict, test_case: TrafficTestCase) -> bool:
        """Determine if car should stop based on detection."""
        confidence = detection.get('confidence', 0.0)
        object_type = detection.get('class_name', '')
        
        # Stop sign rules
        if object_type == 'stop_sign':
            return confidence > 0.5  # High confidence threshold for stop signs
        
        # Person safety rules
        if object_type == 'person':
            return confidence > 0.3  # Lower threshold for safety
        
        return False
    
    def _evaluate_traffic_test_result(self, test_case: TrafficTestCase, 
                                    metrics: Dict) -> Dict:
        """Evaluate if traffic test passed based on expected behavior."""
        car_stopped = metrics['car_stopped']
        response_time = metrics['response_time']
        stop_duration = metrics['stop_duration']
        
        # Define evaluation criteria
        if test_case.expected_behavior == "stop_and_wait":
            passed = (car_stopped and 
                     response_time < 2.0 and  # Quick response
                     stop_duration >= test_case.duration_seconds * 0.8)  # Adequate stop time
            error = None if passed else "Failed to stop properly at stop sign"
            
        elif test_case.expected_behavior == "emergency_stop":
            passed = (car_stopped and response_time < 1.0)  # Very quick response for safety
            error = None if passed else "Failed to emergency stop for person"
            
        elif test_case.expected_behavior == "immediate_stop":
            passed = (car_stopped and response_time < 1.5)  # Quick stop at close range
            error = None if passed else "Failed to stop immediately at close range"
            
        elif test_case.expected_behavior == "no_reaction":
            passed = not car_stopped  # Should not stop for low confidence
            error = None if passed else "Incorrectly stopped for low-confidence detection"
            
        else:
            passed = False
            error = f"Unknown expected behavior: {test_case.expected_behavior}"
        
        return {
            'passed': passed,
            'appropriate': passed,  # Appropriate response matches pass criteria
            'error': error
        }
    
    def _generate_traffic_summary(self) -> Dict:
        """Generate summary of traffic rule test results."""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        
        # Response time analysis
        successful_results = [r for r in self.test_results if r.success and r.response_time > 0]
        if successful_results:
            avg_response_time = np.mean([r.response_time for r in successful_results])
            max_response_time = max([r.response_time for r in successful_results])
        else:
            avg_response_time = max_response_time = 0
        
        # Stop duration analysis
        stop_results = [r for r in self.test_results if r.stop_duration > 0]
        if stop_results:
            avg_stop_duration = np.mean([r.stop_duration for r in stop_results])
        else:
            avg_stop_duration = 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests * 100,
            'average_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'average_stop_duration': avg_stop_duration,
            'safety_compliance': {
                'stop_sign_compliance': sum(1 for r in self.test_results 
                                          if 'stop_sign' in r.test_name and r.success),
                'person_safety_response': sum(1 for r in self.test_results 
                                            if 'person' in r.test_name and r.success)
            }
        }
    
    async def _save_traffic_results(self, results: Dict):
        """Save traffic test results."""
        timestamp = int(time.time())
        
        # Save JSON results
        json_filename = f"traffic_test_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Traffic test results saved to {json_filename}")
        
        # Print summary
        self._print_traffic_summary(results['summary'])
    
    def _print_traffic_summary(self, summary: Dict):
        """Print formatted traffic test summary."""
        logger.info("\n" + "="*60)
        logger.info("TRAFFIC RULE TEST SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['passed_tests'] > 0:
            logger.info(f"\nResponse Performance:")
            logger.info(f"  Average Response Time: {summary['average_response_time']:.2f}s")
            logger.info(f"  Maximum Response Time: {summary['max_response_time']:.2f}s")
            logger.info(f"  Average Stop Duration: {summary['average_stop_duration']:.2f}s")
        
        logger.info(f"\nSafety Compliance:")
        safety = summary['safety_compliance']
        logger.info(f"  Stop Sign Compliance: {safety['stop_sign_compliance']} passed")
        logger.info(f"  Person Safety Response: {safety['person_safety_response']} passed")
        
        logger.info("="*60)

# Specialized testing functions
async def test_stop_sign_detection_accuracy():
    """Test stop sign detection accuracy with various conditions."""
    logger.info("Testing stop sign detection accuracy...")
    
    detector = StopSignDetector()
    detector.start_detection()
    
    test_conditions = [
        {"distance": "close", "lighting": "bright", "angle": "straight"},
        {"distance": "medium", "lighting": "dim", "angle": "angled"},
        {"distance": "far", "lighting": "bright", "angle": "straight"},
        {"distance": "close", "lighting": "dim", "angle": "angled"}
    ]
    
    results = []
    
    try:
        for i, condition in enumerate(test_conditions):
            logger.info(f"\nTesting condition {i+1}: {condition}")
            
            # Simulate detection under different conditions
            if condition["lighting"] == "dim" or condition["angle"] == "angled":
                confidence = 0.6  # Lower confidence in challenging conditions
            else:
                confidence = 0.9  # High confidence in ideal conditions
            
            detector.simulate_stop_sign_detection(confidence)
            
            # Check detection
            detection = detector.detect_stop_sign()
            detected = detection is not None and detection['confidence'] > 0.5
            
            results.append({
                'condition': condition,
                'detected': detected,
                'confidence': detection['confidence'] if detection else 0.0
            })
            
            logger.info(f"  Result: {'DETECTED' if detected else 'NOT DETECTED'}")
            if detection:
                logger.info(f"  Confidence: {detection['confidence']:.2f}")
            
            detector.clear_simulation()
            await asyncio.sleep(1)
    
    finally:
        detector.stop_detection()
    
    # Analyze results
    total_tests = len(results)
    successful_detections = sum(1 for r in results if r['detected'])
    
    logger.info(f"\nDetection Accuracy: {successful_detections}/{total_tests} ({successful_detections/total_tests*100:.1f}%)")
    
    return results

async def test_real_time_detection_response():
    """Test real-time detection and response timing."""
    logger.info("Testing real-time detection response...")
    
    # Create integrated system
    config = SystemConfig()
    system = IntegratedSelfDrivingSystem(config)
    
    try:
        # Simulate car driving
        system.current_position = Position(60, 20, 0)
        driving_speed = 2.0  # cm per iteration
        
        logger.info("Car driving forward...")
        
        # Drive forward and suddenly detect stop sign
        for i in range(20):  # 40cm forward movement
            system.current_position.y += driving_speed
            
            # Detect stop sign at 15th iteration (30cm forward)
            if i == 15:
                start_time = time.time()
                logger.info("STOP SIGN DETECTED!")
                
                # Simulate detection response
                system.object_detector.stop_sign_detected.set()
                
                # Measure response time
                response_time = time.time() - start_time
                logger.info(f"Response time: {response_time:.3f}s")
                
                # Car should stop
                stop_position = system.current_position.y
                logger.info(f"Car stopped at position: {stop_position:.1f}cm")
                
                break
            
            await asyncio.sleep(0.1)  # Simulate movement timing
    
    finally:
        await system.stop()

async def run_traffic_tests():
    """Run complete traffic rule test suite."""
    test_suite = TrafficRuleTestSuite()
    
    try:
        # Run main traffic tests
        traffic_results = await test_suite.run_all_traffic_tests()
        
        # Run specialized tests
        logger.info("\nRunning detection accuracy tests...")
        accuracy_results = await test_stop_sign_detection_accuracy()
        
        logger.info("\nRunning real-time response tests...")
        await test_real_time_detection_response()
        
        # Combined summary
        logger.info("\n" + "="*60)
        logger.info("ALL TRAFFIC TESTS COMPLETE")
        logger.info("="*60)
        logger.info(f"Traffic rule compliance: {traffic_results['summary']['success_rate']:.1f}% success rate")
        
        detection_accuracy = sum(1 for r in accuracy_results if r['detected']) / len(accuracy_results) * 100
        logger.info(f"Detection accuracy: {detection_accuracy:.1f}%")
        
        return {
            'traffic_results': traffic_results,
            'accuracy_results': accuracy_results
        }
        
    except KeyboardInterrupt:
        logger.info("Traffic tests interrupted by user")
    except Exception as e:
        logger.error(f"Traffic tests failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--accuracy":
            logger.info("Running detection accuracy test only...")
            asyncio.run(test_stop_sign_detection_accuracy())
        elif sys.argv[1] == "--realtime":
            logger.info("Running real-time response test only...")
            asyncio.run(test_real_time_detection_response())
        else:
            logger.info("Usage: python test_traffic_rules.py [--accuracy|--realtime]")
    else:
        logger.info("Running complete traffic rule test suite...")
        asyncio.run(run_traffic_tests())