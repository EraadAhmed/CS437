#!/usr/bin/env python3
"""
Distance Calibration Test Suite
CS 437 Lab 1 Part B - Comprehensive Distance Testing

This module provides detailed distance calibration tests to pinpoint
accurate measurements and validate the 30cm camera range detection.
"""

import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Dict
import logging

# Hardware imports with fallback
try:
    from picarx import Picarx
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: Hardware not available. Running in simulation mode.")
    HW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistanceReading:
    """Single distance measurement result."""
    expected_distance: float
    measured_distance: float
    timestamp: float
    servo_angle: int
    measurement_id: int

class DistanceCalibrator:
    """Comprehensive distance calibration system."""
    
    def __init__(self):
        self.picar = None
        self.readings: List[DistanceReading] = []
        
        if HW_AVAILABLE:
            try:
                self.picar = Picarx(servo_pins=["P0", "P1", "P3"])
                logger.info("PiCar initialized for distance calibration")
            except Exception as e:
                logger.error(f"PiCar initialization failed: {e}")
    
    async def calibrate_sensor_accuracy(self) -> Dict:
        """Comprehensive sensor accuracy calibration."""
        logger.info("Starting comprehensive distance calibration...")
        
        # Test distances from 5cm to 100cm
        test_distances = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100]
        
        results = {
            'test_distances': test_distances,
            'measurements': {},
            'statistics': {},
            'camera_range_validation': {}
        }
        
        for distance in test_distances:
            logger.info(f"\n=== Testing at {distance}cm ===")
            print(f"Please place obstacle at {distance}cm and press Enter...")
            input()  # Wait for user to position obstacle
            
            measurements = await self._take_multiple_readings(distance)
            results['measurements'][distance] = measurements
            
            # Calculate statistics
            if measurements:
                stats = self._calculate_statistics(measurements)
                results['statistics'][distance] = stats
                
                logger.info(f"Distance {distance}cm results:")
                logger.info(f"  Mean: {stats['mean']:.2f}cm")
                logger.info(f"  Std:  {stats['std']:.2f}cm") 
                logger.info(f"  Error: {stats['error']:.2f}cm")
                logger.info(f"  Accuracy: {stats['accuracy']:.1f}%")
                
                # Special validation for camera range (30cm)
                if distance == 30:
                    results['camera_range_validation'] = {
                        'target': 30,
                        'measured_mean': stats['mean'],
                        'within_tolerance': abs(stats['error']) < 2,  # 2cm tolerance
                        'recommended_stop_distance': max(25, stats['mean'] - 5)
                    }
        
        await self._generate_calibration_report(results)
        return results
    
    async def _take_multiple_readings(self, expected_distance: float, 
                                    num_readings: int = 20) -> List[float]:
        """Take multiple readings at different servo angles."""
        readings = []
        servo_angles = [-15, -10, -5, 0, 5, 10, 15]  # Test multiple angles
        
        for angle in servo_angles:
            if self.picar:
                self.picar.set_cam_pan_angle(angle)
                await asyncio.sleep(0.3)  # Allow servo to settle
                
                # Take several readings at this angle
                for i in range(3):
                    distance = self.picar.ultrasonic.read()
                    if 0 < distance <= 200:  # Valid reading
                        readings.append(distance)
                        
                        # Store detailed reading
                        reading = DistanceReading(
                            expected_distance=expected_distance,
                            measured_distance=distance,
                            timestamp=time.time(),
                            servo_angle=angle,
                            measurement_id=len(self.readings)
                        )
                        self.readings.append(reading)
                    
                    await asyncio.sleep(0.1)
            else:
                # Simulation mode - add some noise to expected value
                for i in range(3):
                    noise = np.random.normal(0, 1.5)  # 1.5cm standard deviation
                    simulated_reading = expected_distance + noise
                    readings.append(max(0, simulated_reading))
                    await asyncio.sleep(0.1)
        
        # Return servo to center
        if self.picar:
            self.picar.set_cam_pan_angle(0)
            await asyncio.sleep(0.2)
        
        return readings
    
    def _calculate_statistics(self, readings: List[float]) -> Dict:
        """Calculate comprehensive statistics for readings."""
        if not readings:
            return {}
        
        readings_array = np.array(readings)
        expected = readings[0] if hasattr(readings[0], 'expected_distance') else 0
        
        stats = {
            'mean': float(np.mean(readings_array)),
            'std': float(np.std(readings_array)),
            'min': float(np.min(readings_array)),
            'max': float(np.max(readings_array)),
            'median': float(np.median(readings_array)),
            'count': len(readings),
            'error': 0,  # Will be calculated
            'accuracy': 0  # Will be calculated
        }
        
        # Calculate error if we know expected distance
        if hasattr(self.readings[-1], 'expected_distance'):
            expected = self.readings[-1].expected_distance
            stats['error'] = stats['mean'] - expected
            stats['accuracy'] = max(0, 100 - abs(stats['error']) / expected * 100)
        
        return stats
    
    async def test_camera_range_stopping(self) -> Dict:
        """Test optimal stopping distance for camera range detection."""
        logger.info("\n=== Camera Range Stopping Test ===")
        
        # Test stopping behavior at different distances
        test_distances = [20, 25, 30, 35, 40]
        results = {}
        
        for distance in test_distances:
            logger.info(f"Testing stop behavior at {distance}cm...")
            print(f"Place obstacle at {distance}cm. Car will approach and should stop at ~25cm.")
            print("Press Enter when ready...")
            input()
            
            # Simulate approach and measure actual stop distance
            stop_distance = await self._simulate_approach_and_stop(distance)
            
            results[distance] = {
                'obstacle_distance': distance,
                'stop_distance': stop_distance,
                'within_camera_range': stop_distance <= 30,
                'safe_margin': distance - stop_distance
            }
            
            logger.info(f"  Obstacle at {distance}cm: stopped at {stop_distance:.1f}cm")
        
        return results
    
    async def _simulate_approach_and_stop(self, obstacle_distance: float) -> float:
        """Simulate car approaching obstacle and measure stop distance."""
        if not self.picar:
            # Simulation mode
            return max(20, obstacle_distance - 8 + np.random.normal(0, 2))
        
        # Start from further back
        logger.info("Starting approach simulation...")
        
        # Move forward slowly while monitoring distance
        approach_power = 25  # Slow approach
        min_distance = float('inf')
        
        try:
            self.picar.forward(approach_power)
            
            start_time = time.time()
            while time.time() - start_time < 10:  # 10 second max
                current_distance = self.picar.ultrasonic.read()
                
                if 0 < current_distance < min_distance:
                    min_distance = current_distance
                
                # Stop if within camera range (30cm)
                if 0 < current_distance <= 25:  # 25cm stop threshold
                    self.picar.stop()
                    logger.info(f"Stopped at {current_distance}cm")
                    return current_distance
                
                await asyncio.sleep(0.1)
            
            # Timeout - stop anyway
            self.picar.stop()
            return min_distance
            
        except Exception as e:
            logger.error(f"Approach simulation error: {e}")
            if self.picar:
                self.picar.stop()
            return min_distance
    
    async def test_angular_accuracy(self) -> Dict:
        """Test sensor accuracy at different servo angles."""
        logger.info("\n=== Angular Accuracy Test ===")
        
        # Test at 30cm distance with different angles
        test_angles = [-30, -20, -10, 0, 10, 20, 30]
        results = {}
        
        print("Place obstacle at 30cm directly in front and press Enter...")
        input()
        
        for angle in test_angles:
            logger.info(f"Testing at servo angle {angle}°...")
            
            if self.picar:
                self.picar.set_cam_pan_angle(angle)
                await asyncio.sleep(0.5)  # Allow servo to settle
                
                readings = []
                for _ in range(10):
                    distance = self.picar.ultrasonic.read()
                    if 0 < distance <= 200:
                        readings.append(distance)
                    await asyncio.sleep(0.1)
                
                if readings:
                    mean_distance = np.mean(readings)
                    std_distance = np.std(readings)
                    
                    results[angle] = {
                        'angle': angle,
                        'mean_distance': mean_distance,
                        'std_distance': std_distance,
                        'readings_count': len(readings),
                        'expected_distance': 30,
                        'error': mean_distance - 30
                    }
                    
                    logger.info(f"  Angle {angle} degrees: {mean_distance:.1f}+/-{std_distance:.1f}cm")
            
            else:
                # Simulation mode
                # Simulate some angular error
                angle_error = abs(angle) * 0.1  # Small error increase with angle
                simulated_distance = 30 + np.random.normal(angle_error, 1.0)
                
                results[angle] = {
                    'angle': angle,
                    'mean_distance': simulated_distance,
                    'std_distance': 1.0,
                    'readings_count': 10,
                    'expected_distance': 30,
                    'error': simulated_distance - 30
                }
        
        # Return servo to center
        if self.picar:
            self.picar.set_cam_pan_angle(0)
        
        return results
    
    async def _generate_calibration_report(self, results: Dict):
        """Generate comprehensive calibration report."""
        logger.info("\n" + "="*60)
        logger.info("DISTANCE CALIBRATION REPORT")
        logger.info("="*60)
        
        # Save raw data to CSV
        csv_filename = f"distance_calibration_{int(time.time())}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Expected_Distance', 'Measured_Distance', 'Error', 
                           'Std_Deviation', 'Accuracy_%'])
            
            for distance, stats in results['statistics'].items():
                writer.writerow([
                    distance,
                    f"{stats['mean']:.2f}",
                    f"{stats['error']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['accuracy']:.1f}"
                ])
        
        logger.info(f"Raw data saved to {csv_filename}")
        
        # Generate accuracy plot
        self._generate_accuracy_plot(results)
        
        # Camera range validation
        if 'camera_range_validation' in results:
            cam_val = results['camera_range_validation']
            logger.info("\nCAMERA RANGE VALIDATION (30cm target):")
            logger.info(f"  Measured mean: {cam_val['measured_mean']:.2f}cm")
            logger.info(f"  Within tolerance: {cam_val['within_tolerance']}")
            logger.info(f"  Recommended stop distance: {cam_val['recommended_stop_distance']:.0f}cm")
        
        # Overall accuracy summary
        all_errors = [stats['error'] for stats in results['statistics'].values()]
        if all_errors:
            mean_abs_error = np.mean(np.abs(all_errors))
            logger.info(f"\nOVERALL SENSOR PERFORMANCE:")
            logger.info(f"  Mean absolute error: {mean_abs_error:.2f}cm")
            logger.info(f"  Suitable for navigation: {mean_abs_error < 5}")
    
    def _generate_accuracy_plot(self, results: Dict):
        """Generate accuracy visualization plot."""
        try:
            import matplotlib.pyplot as plt
            
            distances = list(results['statistics'].keys())
            measured = [results['statistics'][d]['mean'] for d in distances]
            errors = [results['statistics'][d]['std'] for d in distances]
            
            plt.figure(figsize=(12, 8))
            
            # Accuracy plot
            plt.subplot(2, 2, 1)
            plt.errorbar(distances, measured, yerr=errors, fmt='o-', capsize=5)
            plt.plot([0, 100], [0, 100], 'r--', alpha=0.7, label='Perfect accuracy')
            plt.xlabel('Expected Distance (cm)')
            plt.ylabel('Measured Distance (cm)')
            plt.title('Distance Measurement Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error plot
            plt.subplot(2, 2, 2)
            errors_list = [results['statistics'][d]['error'] for d in distances]
            plt.plot(distances, errors_list, 'ro-')
            plt.axhline(y=0, color='g', linestyle='--', alpha=0.7)
            plt.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='+/-2cm tolerance')
            plt.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Expected Distance (cm)')
            plt.ylabel('Error (cm)')
            plt.title('Measurement Error vs Distance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Standard deviation plot
            plt.subplot(2, 2, 3)
            std_devs = [results['statistics'][d]['std'] for d in distances]
            plt.plot(distances, std_devs, 'bo-')
            plt.xlabel('Expected Distance (cm)')
            plt.ylabel('Standard Deviation (cm)')
            plt.title('Measurement Precision')
            plt.grid(True, alpha=0.3)
            
            # Camera range focus
            plt.subplot(2, 2, 4)
            camera_range = [d for d in distances if 20 <= d <= 40]
            camera_measured = [measured[distances.index(d)] for d in camera_range]
            camera_errors = [errors[distances.index(d)] for d in camera_range]
            
            plt.errorbar(camera_range, camera_measured, yerr=camera_errors, 
                        fmt='go-', capsize=5, linewidth=2)
            plt.axhline(y=25, color='r', linestyle='--', label='Recommended stop distance')
            plt.axhline(y=30, color='orange', linestyle='--', label='Camera range limit')
            plt.xlabel('Expected Distance (cm)')
            plt.ylabel('Measured Distance (cm)')
            plt.title('Camera Range Accuracy (20-40cm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f"distance_calibration_plot_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Accuracy plot saved to {plot_filename}")
            
            # Show plot if in interactive mode
            # plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available - skipping plot generation")
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

async def run_distance_calibration():
    """Run complete distance calibration suite."""
    calibrator = DistanceCalibrator()
    
    try:
        # Main accuracy calibration
        accuracy_results = await calibrator.calibrate_sensor_accuracy()
        
        # Camera range stopping test
        stopping_results = await calibrator.test_camera_range_stopping()
        
        # Angular accuracy test
        angular_results = await calibrator.test_angular_accuracy()
        
        # Combined report
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE CALIBRATION COMPLETE")
        logger.info("="*60)
        
        # Save all results
        combined_results = {
            'timestamp': time.time(),
            'accuracy_test': accuracy_results,
            'stopping_test': stopping_results,
            'angular_test': angular_results
        }
        
        import json
        results_filename = f"complete_calibration_{int(time.time())}.json"
        with open(results_filename, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"Complete results saved to {results_filename}")
        
        return combined_results
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
    finally:
        if calibrator.picar:
            calibrator.picar.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        logger.info("Running quick distance test...")
        # Quick test with fewer measurements
    else:
        logger.info("Running comprehensive distance calibration...")
        asyncio.run(run_distance_calibration())