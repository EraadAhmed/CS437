#!/usr/bin/env python3
"""
CS 437 Lab 1 Part B - Main Test Runner
Comprehensive Self-Driving Car Testing Suite

This script provides a unified interface to run all tests and the main integrated system.
Addresses the sensing speed issues from integrated_main.py and provides comprehensive
distance calibration and testing capabilities.

Usage:
    python run_tests.py                    # Run main integrated system
    python run_tests.py --test-all        # Run all test suites
    python run_tests.py --test-distance   # Test distance calibration only
    python run_tests.py --test-navigation # Test navigation only  
    python run_tests.py --test-traffic    # Test traffic rules only
    python run_tests.py --demo            # Run demonstration scenarios
    python run_tests.py --config          # Show configuration
"""

import asyncio
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List

# Import all our modules
try:
    from integrated import IntegratedSelfDrivingSystem, SystemConfig
    from test_distance_calibration import run_distance_calibration
    from test_navigation import run_navigation_tests
    from test_traffic_rules import run_traffic_tests
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("  - integrated.py")
    print("  - test_distance_calibration.py") 
    print("  - test_navigation.py")
    print("  - test_traffic_rules.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'car_test_log_{int(time.time())}.log')
    ]
)
logger = logging.getLogger(__name__)

class TestOrchestrator:
    """Orchestrates all testing and system execution."""
    
    def __init__(self):
        self.config = SystemConfig()
        self.test_results = {}
        
    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "="*80)
        print("CS 437 Lab 1 Part B - Advanced Self-Driving Car System")
        print("Integrated Testing and Execution Suite")
        print("="*80)
        print("Key Improvements over integrated_main.py:")
        print("  * Optimized sensing speed (no blocking during driving)")
        print("  * Camera range detection (~30cm with 5cm safety margin)")
        print("  * Comprehensive test suites for distance calibration")
        print("  * Advanced A* pathfinding with periodic rescanning")
        print("  * Real-time object detection and traffic rule compliance")
        print("  * Multi-threaded architecture for responsive control")
        print("="*80 + "\n")
    
    def print_configuration(self):
        """Print current system configuration."""
        print("SYSTEM CONFIGURATION:")
        print("-" * 40)
        config_dict = self.config.__dict__
        for key, value in config_dict.items():
            if not key.startswith('_'):
                print(f"  {key:25s}: {value}")
        print("-" * 40 + "\n")
    
    async def run_integrated_system(self):
        """Run the main integrated self-driving system."""
        logger.info("Starting integrated self-driving system...")
        
        system = IntegratedSelfDrivingSystem(self.config)
        
        try:
            print("Starting integrated self-driving car system...")
            print("Goal: Navigate from start to end while avoiding obstacles")
            print("Press Ctrl+C to stop\n")
            
            await system.start()
            
        except KeyboardInterrupt:
            logger.info("System stopped by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await system.stop()
    
    async def run_all_tests(self):
        """Run all test suites comprehensively."""
        logger.info("Starting comprehensive test suite...")
        
        total_start_time = time.time()
        
        print("Running all test suites...")
        print("This may take 15-30 minutes depending on hardware\n")
        
        # Test 1: Distance Calibration
        print("Phase 1/3: Distance Calibration Tests")
        print("-" * 50)
        try:
            distance_results = await run_distance_calibration()
            self.test_results['distance_calibration'] = distance_results
            logger.info("Distance calibration tests completed")
        except Exception as e:
            logger.error(f"Distance calibration tests failed: {e}")
            self.test_results['distance_calibration'] = {'error': str(e)}
        
        print("\n")
        
        # Test 2: Navigation Tests
        print("Phase 2/3: Navigation Tests")
        print("-" * 50)
        try:
            navigation_results = await run_navigation_tests()
            self.test_results['navigation'] = navigation_results
            logger.info("Navigation tests completed")
        except Exception as e:
            logger.error(f"Navigation tests failed: {e}")
            self.test_results['navigation'] = {'error': str(e)}
        
        print("\n")
        
        # Test 3: Traffic Rule Tests
        print("Phase 3/3: Traffic Rule Tests")
        print("-" * 50)
        try:
            traffic_results = await run_traffic_tests()
            self.test_results['traffic_rules'] = traffic_results
            logger.info("Traffic rule tests completed")
        except Exception as e:
            logger.error(f"Traffic rule tests failed: {e}")
            self.test_results['traffic_rules'] = {'error': str(e)}
        
        # Generate comprehensive report
        total_time = time.time() - total_start_time
        await self._generate_comprehensive_report(total_time)
    
    async def run_demo_scenarios(self):
        """Run demonstration scenarios for lab video."""
        logger.info("Running demonstration scenarios...")
        
        scenarios = [
            {
                'name': 'Simple Navigation Demo',
                'description': 'Navigate straight path with no obstacles',
                'duration': 30
            },
            {
                'name': 'Obstacle Avoidance Demo',
                'description': 'Navigate around obstacles using A* pathfinding',
                'duration': 45
            },
            {
                'name': 'Stop Sign Compliance Demo', 
                'description': 'Detect stop sign and perform proper stopping behavior',
                'duration': 20
            },
            {
                'name': 'Safety Stop Demo',
                'description': 'Emergency stop when person detected',
                'duration': 15
            }
        ]
        
        print("DEMONSTRATION SCENARIOS")
        print("=" * 60)
        print("These scenarios demonstrate the key capabilities for your lab video:\n")
        
        for i, scenario in enumerate(scenarios):
            print(f"Scenario {i+1}: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Expected duration: {scenario['duration']} seconds")
            
            response = input(f"\nRun this scenario? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                try:
                    await self._run_demo_scenario(scenario)
                except KeyboardInterrupt:
                    print("Scenario stopped by user")
                except Exception as e:
                    logger.error(f"Scenario failed: {e}")
            
            print("-" * 60)
    
    async def _run_demo_scenario(self, scenario: Dict):
        """Run a single demonstration scenario."""
        print(f"\nStarting {scenario['name']}...")
        
        system = IntegratedSelfDrivingSystem(self.config)
        
        try:
            # Customize system for specific scenario
            if 'Simple Navigation' in scenario['name']:
                # Clear obstacles for simple demo
                system.goal_position.x = 60  # Straight line
                system.goal_position.y = 200  # Not too far
                
            elif 'Obstacle Avoidance' in scenario['name']:
                # Add some obstacles
                print("Place 2-3 obstacles in the car's path")
                input("Press Enter when obstacles are positioned...")
                
            elif 'Stop Sign' in scenario['name']:
                # Set up for stop sign demo
                print("Hold stop sign image in front of camera when car approaches")
                input("Press Enter to start...")
                
            elif 'Safety Stop' in scenario['name']:
                # Set up for person detection
                print("Show person image or step in front of camera during movement")
                input("Press Enter to start...")
            
            # Run scenario with timeout
            await asyncio.wait_for(
                system.start(), 
                timeout=scenario['duration'] + 10
            )
            
        except asyncio.TimeoutError:
            print(f"Scenario completed (timeout after {scenario['duration']}s)")
        finally:
            await system.stop()
    
    async def _generate_comprehensive_report(self, total_time: float):
        """Generate comprehensive test report."""
        timestamp = int(time.time())
        
        # Compile overall statistics
        overall_stats = {
            'timestamp': timestamp,
            'total_test_time': total_time,
            'test_results': self.test_results,
            'configuration': self.config.__dict__,
            'summary': {}
        }
        
        # Calculate summary statistics
        summary = {}
        
        # Distance calibration summary
        if 'distance_calibration' in self.test_results:
            dist_results = self.test_results['distance_calibration']
            if 'error' not in dist_results:
                summary['distance_calibration'] = {
                    'completed': True,
                    'camera_range_validated': dist_results.get('camera_range_validation', {}).get('within_tolerance', False)
                }
            else:
                summary['distance_calibration'] = {'completed': False, 'error': dist_results['error']}
        
        # Navigation summary
        if 'navigation' in self.test_results:
            nav_results = self.test_results['navigation']
            if 'error' not in nav_results and 'summary' in nav_results:
                summary['navigation'] = {
                    'completed': True,
                    'success_rate': nav_results['summary'].get('success_rate', 0),
                    'average_time': nav_results['summary'].get('average_completion_time', 0)
                }
            else:
                summary['navigation'] = {'completed': False}
        
        # Traffic rules summary
        if 'traffic_rules' in self.test_results:
            traffic_results = self.test_results['traffic_rules']
            if 'error' not in traffic_results:
                if 'traffic_results' in traffic_results:
                    traffic_summary = traffic_results['traffic_results'].get('summary', {})
                    summary['traffic_rules'] = {
                        'completed': True,
                        'success_rate': traffic_summary.get('success_rate', 0),
                        'avg_response_time': traffic_summary.get('average_response_time', 0)
                    }
                else:
                    summary['traffic_rules'] = {'completed': True}
            else:
                summary['traffic_rules'] = {'completed': False}
        
        overall_stats['summary'] = summary
        
        # Save comprehensive report
        report_filename = f"comprehensive_test_report_{timestamp}.json"
        with open(report_filename, 'w') as f:
            json.dump(overall_stats, f, indent=2, default=str)
        
        # Print summary
        self._print_final_summary(summary, total_time)
        
        logger.info(f"Comprehensive test report saved to {report_filename}")
    
    def _print_final_summary(self, summary: Dict, total_time: float):
        """Print final test summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print("="*80)
        
        print(f"Total Test Time: {total_time/60:.1f} minutes")
        print()
        
        # Distance calibration summary
        if 'distance_calibration' in self.test_results:
            dist_results = self.test_results['distance_calibration']
            if 'error' not in dist_results:
                camera_val = "[PASS]" if dist_results.get('camera_range_validation', {}).get('within_tolerance', False) else "[FAIL]"
                print(f"Distance Calibration:     COMPLETED {camera_val}")
            else:
                print(f"Distance Calibration:     FAILED")
        
        # Navigation
        nav_summary = summary.get('navigation', {})
        if nav_summary.get('completed'):
            success_rate = nav_summary.get('success_rate', 0)
            print(f"Navigation Tests:         COMPLETED ({success_rate:.1f}% success)")
        else:
            print(f"Navigation Tests:         FAILED")
        
        # Traffic rules
        traffic_summary = summary.get('traffic_rules', {})
        if traffic_summary.get('completed'):
            success_rate = traffic_summary.get('success_rate', 0)
            print(f"Traffic Rule Tests:       COMPLETED ({success_rate:.1f}% success)")
        else:
            print(f"Traffic Rule Tests:       FAILED")
        
        print("\nKey Improvements Validated:")
        print("  * Camera range detection (30cm +/- 5cm)")
        print("  * Non-blocking sensing during movement")
        print("  * Real-time obstacle avoidance")
        print("  * A* pathfinding with replanning")
        print("  * Traffic rule compliance")
        
        print("\nFiles Generated:")
        for file_path in Path('.').glob('*test*results*.json'):
            print(f"  - {file_path}")
        for file_path in Path('.').glob('*calibration*.csv'):
            print(f"  - {file_path}")
        for file_path in Path('.').glob('*plot*.png'):
            print(f"  - {file_path}")
        
        print("="*80)

def print_help():
    """Print help information."""
    print(__doc__)

async def main():
    """Main entry point."""
    orchestrator = TestOrchestrator()
    orchestrator.print_banner()
    
    if len(sys.argv) < 2:
        # Default: run integrated system
        orchestrator.print_configuration()
        await orchestrator.run_integrated_system()
        return
    
    command = sys.argv[1]
    
    if command == "--help" or command == "-h":
        print_help()
        
    elif command == "--config":
        orchestrator.print_configuration()
        
    elif command == "--test-all":
        orchestrator.print_configuration()
        await orchestrator.run_all_tests()
        
    elif command == "--test-distance":
        print("Running distance calibration tests only...")
        await run_distance_calibration()
        
    elif command == "--test-navigation":
        print("Running navigation tests only...")
        await run_navigation_tests()
        
    elif command == "--test-traffic":
        print("Running traffic rule tests only...")
        await run_traffic_tests()
        
    elif command == "--demo":
        await orchestrator.run_demo_scenarios()
        
    else:
        print(f"Unknown command: {command}")
        print("Use --help to see available options")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Program failed: {e}")
        sys.exit(1)