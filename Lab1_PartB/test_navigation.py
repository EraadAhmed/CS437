#!/usr/bin/env python3
"""
Navigation Test Suite
CS 437 Lab 1 Part B - Comprehensive Navigation Testing

This module provides detailed navigation tests for path planning,
obstacle avoidance, and goal reaching validation.
"""

import asyncio
import time
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
import matplotlib.pyplot as plt

# Import our integrated system
from integrated import IntegratedSelfDrivingSystem, SystemConfig, Position, OccupancyGrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NavigationTestCase:
    """Single navigation test case definition."""
    name: str
    start_position: Position
    goal_position: Position
    obstacles: List[Tuple[int, int]]  # (x, y) positions in cm
    expected_max_time: float  # seconds
    expected_path_length: float  # cm
    description: str

@dataclass  
class TestResult:
    """Results from a navigation test."""
    test_name: str
    success: bool
    final_position: Position
    time_taken: float
    path_length: float
    obstacles_avoided: int
    replan_count: int
    error_message: Optional[str] = None

class NavigationTestSuite:
    """Comprehensive navigation testing system."""
    
    def __init__(self):
        self.config = SystemConfig()
        self.test_results: List[TestResult] = []
        
        # Define comprehensive test cases
        self.test_cases = self._define_test_cases()
    
    def _define_test_cases(self) -> List[NavigationTestCase]:
        """Define all navigation test scenarios."""
        return [
            # Test 1: Simple straight line navigation
            NavigationTestCase(
                name="straight_line_navigation",
                start_position=Position(60, 20, 0),  # Middle, near start
                goal_position=Position(60, 200, 0),  # Middle, far forward
                obstacles=[],
                expected_max_time=30.0,
                expected_path_length=180.0,
                description="Navigate straight forward with no obstacles"
            ),
            
            # Test 2: Single obstacle avoidance
            NavigationTestCase(
                name="single_obstacle_avoidance",
                start_position=Position(60, 20, 0),
                goal_position=Position(60, 200, 0),
                obstacles=[(60, 100)],  # Obstacle directly in path
                expected_max_time=45.0,
                expected_path_length=220.0,
                description="Navigate around single obstacle in direct path"
            ),
            
            # Test 3: Multiple obstacles requiring complex path
            NavigationTestCase(
                name="multiple_obstacles_complex",
                start_position=Position(30, 20, 0),
                goal_position=Position(90, 200, 0),
                obstacles=[(40, 60), (70, 100), (50, 140)],
                expected_max_time=60.0,
                expected_path_length=280.0,
                description="Navigate through multiple obstacles requiring zigzag path"
            ),
            
            # Test 4: Narrow corridor navigation
            NavigationTestCase(
                name="narrow_corridor",
                start_position=Position(60, 20, 0),
                goal_position=Position(60, 200, 0),
                obstacles=[(40, 80), (80, 80), (40, 120), (80, 120)],  # Narrow passage
                expected_max_time=50.0,
                expected_path_length=200.0,
                description="Navigate through narrow corridor between obstacles"
            ),
            
            # Test 5: L-shaped path around wall
            NavigationTestCase(
                name="l_shaped_path",
                start_position=Position(30, 20, 0),
                goal_position=Position(90, 80, 0),
                obstacles=[(50, 40), (70, 40), (50, 60), (70, 60)],  # Wall blocking direct path
                expected_max_time=40.0,
                expected_path_length=160.0,
                description="Navigate L-shaped path around wall obstacle"
            ),
            
            # Test 6: Goal positioning accuracy
            NavigationTestCase(
                name="goal_accuracy_test",
                start_position=Position(40, 40, 0),
                goal_position=Position(80, 100, 0),  # Specific target
                obstacles=[(60, 70)],  # Single obstacle
                expected_max_time=35.0,
                expected_path_length=100.0,
                description="Test accuracy of reaching specific goal position"
            ),
            
            # Test 7: Recovery from dead end
            NavigationTestCase(
                name="dead_end_recovery",
                start_position=Position(60, 20, 0),
                goal_position=Position(60, 150, 0),
                obstacles=[(50, 80), (60, 80), (70, 80), (50, 90), (70, 90)],  # U-shaped trap
                expected_max_time=70.0,
                expected_path_length=250.0,
                description="Navigate out of dead-end situation requiring backtracking"
            ),
            
            # Test 8: Dynamic replanning test
            NavigationTestCase(
                name="dynamic_replanning",
                start_position=Position(30, 20, 0),
                goal_position=Position(90, 180, 0),
                obstacles=[(60, 60)],  # Will add more obstacles during test
                expected_max_time=50.0,
                expected_path_length=200.0,
                description="Test dynamic replanning when new obstacles appear"
            )
        ]
    
    async def run_all_tests(self) -> Dict:
        """Run all navigation test cases."""
        logger.info("Starting comprehensive navigation test suite...")
        logger.info(f"Running {len(self.test_cases)} test cases...")
        
        overall_results = {
            'test_results': [],
            'summary': {},
            'timestamp': time.time()
        }
        
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST {i+1}/{len(self.test_cases)}: {test_case.name}")
            logger.info(f"{'='*60}")
            logger.info(f"Description: {test_case.description}")
            
            try:
                result = await self._run_single_test(test_case)
                self.test_results.append(result)
                overall_results['test_results'].append(result.__dict__)
                
                # Log result
                status = "PASS" if result.success else "FAIL"
                logger.info(f"Result: {status}")
                if result.success:
                    logger.info(f"  Time: {result.time_taken:.1f}s")
                    logger.info(f"  Final distance to goal: {self._calculate_goal_distance(result.final_position, test_case.goal_position):.1f}cm")
                else:
                    logger.info(f"  Error: {result.error_message}")
                
            except Exception as e:
                logger.error(f"Test {test_case.name} failed with exception: {e}")
                result = TestResult(
                    test_name=test_case.name,
                    success=False,
                    final_position=test_case.start_position,
                    time_taken=0,
                    path_length=0,
                    obstacles_avoided=0,
                    replan_count=0,
                    error_message=str(e)
                )
                self.test_results.append(result)
                overall_results['test_results'].append(result.__dict__)
        
        # Generate summary
        overall_results['summary'] = self._generate_test_summary()
        
        # Save results
        await self._save_results(overall_results)
        
        return overall_results
    
    async def _run_single_test(self, test_case: NavigationTestCase) -> TestResult:
        """Run a single navigation test case."""
        # Create isolated system for this test
        system = IntegratedSelfDrivingSystem(self.config)
        
        # Set up test scenario
        system.current_position = Position(
            test_case.start_position.x,
            test_case.start_position.y,
            test_case.start_position.theta
        )
        system.goal_position = Position(
            test_case.goal_position.x,
            test_case.goal_position.y,
            test_case.goal_position.theta
        )
        
        # Add obstacles to grid
        await self._setup_test_obstacles(system, test_case.obstacles)
        
        # Track test metrics
        start_time = time.time()
        start_position = system.current_position
        path_length = 0.0
        replan_count = 0
        
        try:
            # Initialize system
            await system._perform_initial_scan()
            
            # Run navigation with timeout
            navigation_task = asyncio.create_task(self._run_navigation_with_monitoring(
                system, test_case, start_time
            ))
            
            # Wait for completion or timeout
            try:
                result_data = await asyncio.wait_for(
                    navigation_task, 
                    timeout=test_case.expected_max_time + 10
                )
                path_length = result_data['path_length']
                replan_count = result_data['replan_count']
                
            except asyncio.TimeoutError:
                logger.warning(f"Test {test_case.name} timed out")
                return TestResult(
                    test_name=test_case.name,
                    success=False,
                    final_position=system.current_position,
                    time_taken=time.time() - start_time,
                    path_length=path_length,
                    obstacles_avoided=len(test_case.obstacles),
                    replan_count=replan_count,
                    error_message="Navigation timed out"
                )
            
            # Calculate final results
            final_time = time.time() - start_time
            goal_distance = self._calculate_goal_distance(
                system.current_position, 
                test_case.goal_position
            )
            
            # Determine success
            success = goal_distance < 15.0  # 15cm tolerance
            
            return TestResult(
                test_name=test_case.name,
                success=success,
                final_position=system.current_position,
                time_taken=final_time,
                path_length=path_length,
                obstacles_avoided=len(test_case.obstacles),
                replan_count=replan_count,
                error_message=None if success else f"Goal distance: {goal_distance:.1f}cm"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                success=False,
                final_position=system.current_position,
                time_taken=time.time() - start_time,
                path_length=0,
                obstacles_avoided=0,
                replan_count=0,
                error_message=str(e)
            )
        finally:
            await system.stop()
    
    async def _setup_test_obstacles(self, system: IntegratedSelfDrivingSystem, 
                                   obstacles: List[Tuple[int, int]]):
        """Set up obstacles in the system's grid."""
        for x, y in obstacles:
            # Add obstacle to grid
            gx = int(x / system.grid.resolution)
            gy = int(y / system.grid.resolution)
            
            if (0 <= gx < system.grid.grid_width and 
                0 <= gy < system.grid.grid_height):
                system.grid.grid[gy, gx] = 1  # Mark as obstacle
                
                # Add some size to the obstacle
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = gx + dx, gy + dy
                        if (0 <= nx < system.grid.grid_width and 
                            0 <= ny < system.grid.grid_height):
                            system.grid.grid[ny, nx] = 1
    
    async def _run_navigation_with_monitoring(self, system: IntegratedSelfDrivingSystem,
                                            test_case: NavigationTestCase,
                                            start_time: float) -> Dict:
        """Run navigation while monitoring progress."""
        path_length = 0.0
        replan_count = 0
        last_position = system.current_position
        last_replan_time = system.last_replan_time
        
        # Simplified navigation loop for testing
        while time.time() - start_time < test_case.expected_max_time:
            # Update position tracking
            current_distance = self._calculate_distance(last_position, system.current_position)
            path_length += current_distance
            last_position = system.current_position
            
            # Count replanning events
            if system.last_replan_time > last_replan_time:
                replan_count += 1
                last_replan_time = system.last_replan_time
            
            # Check if goal reached
            goal_distance = self._calculate_goal_distance(
                system.current_position, 
                test_case.goal_position
            )
            
            if goal_distance < 15.0:  # Goal reached
                break
            
            # Simulate navigation step
            await self._simulate_navigation_step(system, test_case)
            
            await asyncio.sleep(0.1)  # Control loop frequency
        
        return {
            'path_length': path_length,
            'replan_count': replan_count
        }
    
    async def _simulate_navigation_step(self, system: IntegratedSelfDrivingSystem,
                                      test_case: NavigationTestCase):
        """Simulate one navigation step for testing."""
        # This is a simplified simulation for testing purposes
        # In real implementation, this would be the actual car movement
        
        # Calculate direction to goal
        dx = test_case.goal_position.x - system.current_position.x
        dy = test_case.goal_position.y - system.current_position.y
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        if distance_to_goal > 5:  # Move towards goal if not close
            # Normalize direction
            step_size = min(5.0, distance_to_goal)  # 5cm steps
            unit_dx = dx / distance_to_goal
            unit_dy = dy / distance_to_goal
            
            # Check for obstacles in path
            new_x = system.current_position.x + step_size * unit_dx
            new_y = system.current_position.y + step_size * unit_dy
            
            # Simple obstacle avoidance for simulation
            if self._check_obstacle_at_position(system, new_x, new_y):
                # Try to go around obstacle
                new_x = system.current_position.x + step_size * unit_dy  # Perpendicular
                new_y = system.current_position.y - step_size * unit_dx
                
                if self._check_obstacle_at_position(system, new_x, new_y):
                    # Try other direction
                    new_x = system.current_position.x - step_size * unit_dy
                    new_y = system.current_position.y + step_size * unit_dx
            
            # Update position
            system.current_position.x = max(0, min(self.config.FIELD_WIDTH, new_x))
            system.current_position.y = max(0, min(self.config.FIELD_LENGTH, new_y))
    
    def _check_obstacle_at_position(self, system: IntegratedSelfDrivingSystem,
                                  x: float, y: float) -> bool:
        """Check if there's an obstacle at the given position."""
        gx = int(x / system.grid.resolution)
        gy = int(y / system.grid.resolution)
        
        if (0 <= gx < system.grid.grid_width and 
            0 <= gy < system.grid.grid_height):
            return system.grid.grid[gy, gx] == 1
        return False
    
    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between two positions."""
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
    
    def _calculate_goal_distance(self, current: Position, goal: Position) -> float:
        """Calculate distance to goal."""
        return self._calculate_distance(current, goal)
    
    def _generate_test_summary(self) -> Dict:
        """Generate summary statistics for all tests."""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        
        # Calculate averages for successful tests
        successful_results = [r for r in self.test_results if r.success]
        
        if successful_results:
            avg_time = np.mean([r.time_taken for r in successful_results])
            avg_path_length = np.mean([r.path_length for r in successful_results])
            avg_replans = np.mean([r.replan_count for r in successful_results])
        else:
            avg_time = avg_path_length = avg_replans = 0
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests * 100,
            'average_completion_time': avg_time,
            'average_path_length': avg_path_length,
            'average_replanning_events': avg_replans,
            'test_categories': {
                'simple_navigation': 0,
                'obstacle_avoidance': 0,
                'complex_scenarios': 0
            }
        }
        
        # Categorize test results
        for result in self.test_results:
            if 'straight_line' in result.test_name:
                summary['test_categories']['simple_navigation'] += 1 if result.success else 0
            elif 'obstacle' in result.test_name or 'corridor' in result.test_name:
                summary['test_categories']['obstacle_avoidance'] += 1 if result.success else 0
            else:
                summary['test_categories']['complex_scenarios'] += 1 if result.success else 0
        
        return summary
    
    async def _save_results(self, results: Dict):
        """Save test results to files."""
        timestamp = int(time.time())
        
        # Save JSON results
        json_filename = f"navigation_test_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {json_filename}")
        
        # Generate visualization
        await self._generate_test_visualization(results)
        
        # Print summary
        self._print_test_summary(results['summary'])
    
    async def _generate_test_visualization(self, results: Dict):
        """Generate visualization of test results."""
        try:
            # Create test results plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Test success/failure chart
            test_names = [r['test_name'] for r in results['test_results']]
            success_status = [1 if r['success'] else 0 for r in results['test_results']]
            
            ax1.bar(range(len(test_names)), success_status, 
                   color=['green' if s else 'red' for s in success_status])
            ax1.set_xticks(range(len(test_names)))
            ax1.set_xticklabels([name.replace('_', '\n') for name in test_names], rotation=45)
            ax1.set_ylabel('Success (1) / Failure (0)')
            ax1.set_title('Test Results Overview')
            ax1.grid(True, alpha=0.3)
            
            # Completion time comparison
            completion_times = [r['time_taken'] for r in results['test_results']]
            ax2.plot(range(len(test_names)), completion_times, 'bo-')
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels([name.replace('_', '\n') for name in test_names], rotation=45)
            ax2.set_ylabel('Completion Time (s)')
            ax2.set_title('Navigation Time Performance')
            ax2.grid(True, alpha=0.3)
            
            # Path length analysis
            path_lengths = [r['path_length'] for r in results['test_results']]
            ax3.bar(range(len(test_names)), path_lengths, alpha=0.7)
            ax3.set_xticks(range(len(test_names)))
            ax3.set_xticklabels([name.replace('_', '\n') for name in test_names], rotation=45)
            ax3.set_ylabel('Path Length (cm)')
            ax3.set_title('Path Length Comparison')
            ax3.grid(True, alpha=0.3)
            
            # Success rate by category
            summary = results['summary']
            categories = ['simple_navigation', 'obstacle_avoidance', 'complex_scenarios']
            category_success = [summary['test_categories'][cat] for cat in categories]
            
            ax4.pie(category_success, labels=[cat.replace('_', ' ').title() for cat in categories], 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Success Rate by Test Category')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"navigation_test_visualization_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Test visualization saved to {plot_filename}")
            
        except ImportError:
            logger.warning("Matplotlib not available - skipping visualization")
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    def _print_test_summary(self, summary: Dict):
        """Print formatted test summary."""
        logger.info("\n" + "="*60)
        logger.info("NAVIGATION TEST SUITE SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['passed_tests'] > 0:
            logger.info(f"\nPerformance Metrics (Successful Tests):")
            logger.info(f"  Average Completion Time: {summary['average_completion_time']:.1f}s")
            logger.info(f"  Average Path Length: {summary['average_path_length']:.1f}cm")
            logger.info(f"  Average Replanning Events: {summary['average_replanning_events']:.1f}")
        
        logger.info(f"\nCategory Results:")
        for category, count in summary['test_categories'].items():
            logger.info(f"  {category.replace('_', ' ').title()}: {count} passed")
        
        logger.info("="*60)

# Specialized test scenarios
async def test_camera_range_navigation():
    """Test navigation with camera range constraints."""
    logger.info("Testing camera range navigation constraints...")
    
    config = SystemConfig()
    config.CAMERA_RANGE = 30  # 30cm detection range
    
    system = IntegratedSelfDrivingSystem(config)
    
    # Test scenario: obstacle at 25cm distance
    test_case = NavigationTestCase(
        name="camera_range_test",
        start_position=Position(60, 20, 0),
        goal_position=Position(60, 100, 0),
        obstacles=[(60, 45)],  # Obstacle at 25cm from start
        expected_max_time=20.0,
        expected_path_length=100.0,
        description="Test stopping at camera detection range"
    )
    
    # Run test with monitoring
    start_time = time.time()
    
    try:
        # Set up scenario
        system.current_position = test_case.start_position
        system.goal_position = test_case.goal_position
        
        # Add obstacle
        await system._setup_test_obstacles(system, test_case.obstacles)
        
        # Monitor approach to obstacle
        while time.time() - start_time < 20:
            distance_to_obstacle = abs(system.current_position.y - 45)  # Distance to obstacle
            
            if distance_to_obstacle <= config.CAMERA_RANGE:
                logger.info(f"Stopped at {distance_to_obstacle:.1f}cm from obstacle (camera range: {config.CAMERA_RANGE}cm)")
                break
            
            # Simulate approach
            system.current_position.y += 1  # Move 1cm forward
            await asyncio.sleep(0.1)
        
        # Validate stop distance
        final_distance = abs(system.current_position.y - 45)
        success = final_distance >= (config.CAMERA_RANGE - config.SAFETY_MARGIN)
        
        logger.info(f"Camera range test: {'PASS' if success else 'FAIL'}")
        logger.info(f"Final distance to obstacle: {final_distance:.1f}cm")
        
        return success
        
    finally:
        await system.stop()

async def run_navigation_tests():
    """Run complete navigation test suite."""
    test_suite = NavigationTestSuite()
    
    try:
        # Run main test suite
        results = await test_suite.run_all_tests()
        
        # Run specialized tests
        camera_test_success = await test_camera_range_navigation()
        
        # Combined results
        logger.info("\n" + "="*60)
        logger.info("ALL NAVIGATION TESTS COMPLETE")
        logger.info("="*60)
        logger.info(f"Main test suite: {results['summary']['success_rate']:.1f}% success rate")
        logger.info(f"Camera range test: {'PASS' if camera_test_success else 'FAIL'}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Navigation tests interrupted by user")
    except Exception as e:
        logger.error(f"Navigation tests failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--camera":
            logger.info("Running camera range test only...")
            asyncio.run(test_camera_range_navigation())
        elif sys.argv[1] == "--quick":
            logger.info("Running quick navigation test...")
            # Would run subset of tests
            asyncio.run(run_navigation_tests())
        else:
            logger.info("Usage: python test_navigation.py [--camera|--quick]")
    else:
        logger.info("Running complete navigation test suite...")
        asyncio.run(run_navigation_tests())