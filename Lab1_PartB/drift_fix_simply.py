# simple_drift_fix.py
# CS 4        # HARDWARE FIXED: Pin configuration corrected!
        # No servo offset needed anymore
        self.SERVO_OFFSET = 0  # Hardware drift resolvedb 1: Simple but Effective Drift Fix
# Based on your actual test results showing severe rightward drift

import asyncio
import time
import logging
from threading import Event

# Hardware imports
try:
    from picarx import Picarx
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: PiCarx not available")
    HW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDriftFix:
    """
    Hardware has been FIXED! Pin configuration corrected.
    No servo offset needed - car should drive straight now.
    This test verifies the fix is working correctly.
    """
    
    def __init__(self):
        # Based on your actual drift: 60cm right over 175cm forward
        # Drift angle = arctan(60/175) ≈ 19 degrees
        
        # MUCH MORE AGGRESSIVE servo offset
        self.SERVO_OFFSET = 0   # Hardware fixed: no offset needed
        
        # Movement parameters
        self.DRIVE_POWER = 30
        self.DRIVE_SPEED = 20.0  # cm/s
        self.STEP_SIZE = 20      # cm per step
        
        # Target
        self.TARGET_Y = 375      # End of hallway
        self.current_y = 0
        
        # Hardware
        self.picarx = None
        self.stop_event = Event()
        
        logger.info(f"Simple drift fix initialized - HARDWARE FIXED: SERVO_OFFSET = {self.SERVO_OFFSET} (no offset needed)")

    async def initialize(self):
        """Initialize with aggressive servo offset."""
        if not HW_AVAILABLE:
            logger.error("Hardware not available")
            return False
            
        try:
            self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
            # FIRST: Reset direction angle to 0 (center)
            logger.info("Resetting direction servo to center (0°)")
            self.picarx.set_dir_servo_angle(0)
            await asyncio.sleep(1.0)  # Let it settle at center
            
            # THEN: Apply the aggressive servo offset
            logger.info(f"Applying servo offset: {self.SERVO_OFFSET}°")
            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
            await asyncio.sleep(1.0)  # Let it settle at offset
            
            logger.info("PiCar initialized with aggressive drift correction")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def move_forward(self, distance):
        """Move forward with the aggressive servo offset."""
        if not self.picarx:
            return 0
        
        logger.info(f"Moving forward {distance}cm with servo offset {self.SERVO_OFFSET}")
        
        try:
            # FIRST: Reset servo to center, THEN apply offset
            self.picarx.set_dir_servo_angle(0)
            await asyncio.sleep(0.1)
            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
            await asyncio.sleep(0.1)
            
            # Calculate movement time
            move_time = distance / self.DRIVE_SPEED
            
            # Move forward
            self.picarx.forward(self.DRIVE_POWER)
            
            # Monitor for obstacles during movement
            elapsed = 0
            check_interval = 0.2
            
            while elapsed < move_time and not self.stop_event.is_set():
                # Check for obstacles
                try:
                    obstacle_distance = self.picarx.ultrasonic.read()
                    if 0 < obstacle_distance <= 20:
                        logger.warning(f"Obstacle detected at {obstacle_distance}cm - stopping")
                        break
                except:
                    pass  # Continue if ultrasonic fails
                
                await asyncio.sleep(check_interval)
                elapsed += check_interval
            
            # Stop
            self.picarx.stop()
            
            # Calculate actual distance moved
            actual_distance = (elapsed / move_time) * distance
            self.current_y += actual_distance
            
            logger.info(f"Moved {actual_distance:.1f}cm, total distance: {self.current_y:.1f}cm")
            return actual_distance
            
        except Exception as e:
            logger.error(f"Movement failed: {e}")
            if self.picarx:
                self.picarx.stop()
            return 0

    async def run_navigation(self):
        """Simple navigation - just go straight with drift correction."""
        logger.info("Starting navigation with drift correction...")
        
        step_count = 0
        
        try:
            while not self.stop_event.is_set():
                step_count += 1
                
                # Check if we've reached the goal
                if self.current_y >= self.TARGET_Y:
                    logger.info("GOAL REACHED!")
                    break
                
                # Calculate remaining distance
                remaining = self.TARGET_Y - self.current_y
                step_distance = min(self.STEP_SIZE, remaining)
                
                logger.info(f"Step {step_count}: Moving {step_distance:.1f}cm (remaining: {remaining:.1f}cm)")
                
                # Move forward
                moved = await self.move_forward(step_distance)
                
                if moved < 2:  # Barely moved - probably hit obstacle
                    logger.warning("Movement blocked - stopping navigation")
                    break
                
                # Progress report
                progress = (self.current_y / self.TARGET_Y) * 100
                logger.info(f"Progress: {progress:.1f}% complete")
                
                # Small delay between steps
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("Navigation interrupted")
        except Exception as e:
            logger.error(f"Navigation error: {e}")
        finally:
            if self.picarx:
                self.picarx.stop()

    async def run(self):
        """Main run method."""
        try:
            if await self.initialize():
                await self.run_navigation()
            else:
                logger.error("Failed to initialize")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.stop_event.set()
        if self.picarx:
            self.picarx.stop()
        logger.info("Shutdown complete")


# Interactive calibration to find the right offset
async def find_servo_offset():
    """
    Interactive calibration based on your drift pattern.
    Your car went 60cm right in 175cm forward, so we need to find
    the servo offset that counteracts this drift.
    """
    if not HW_AVAILABLE:
        print("Hardware not available for calibration")
        return
    
    print("\n" + "="*50)
    print("SERVO OFFSET CALIBRATION")
    print("="*50)
    print("Hardware has been FIXED! Pin configuration corrected.")
    print("Testing small offsets around 0 to verify no drift.")
    print("")
    
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    # Hardware fixed: test around 0 offset
    test_offsets = [-2, -1, 0, 1, 2]
    
    print("Instructions:")
    print("1. Place car at starting position")
    print("2. For each test, the car will drive forward for 3 seconds")
    print("3. Observe if it goes straight, drifts left, or drifts right")
    print("4. With hardware fixed, it should go straight at offset 0")
    print("")
    
    for offset in test_offsets:
        print(f"\n--- Testing servo offset: {offset}° ---")
        input("Press Enter when ready to test (make sure car is in starting position)...")
        
        try:
            # FIRST: Reset servo to center (0°)
            logger.info("Resetting servo to center (0°)")
            picarx.set_dir_servo_angle(0)
            await asyncio.sleep(1.0)
            
            # THEN: Set the test servo offset
            logger.info(f"Setting servo offset to {offset}°")
            picarx.set_dir_servo_angle(offset)
            await asyncio.sleep(1.0)
            
            print(f"Testing with offset {offset}° for 3 seconds...")
            
            # Drive forward
            picarx.forward(25)
            await asyncio.sleep(3.0)
            picarx.stop()
            
            print(f"With offset {offset}°:")
            print("  - Did it go straight? (ideal)")
            print("  - Did it drift right? (need more negative offset)")
            print("  - Did it drift left? (offset too negative)")
            
            result = input("Result (s=straight, r=right, l=left, q=quit): ").lower()
            
            if result == 's':
                print(f"\n🎉 SUCCESS! Use SERVO_OFFSET = {offset} in your code")
                print(f"Update this line in your code:")
                print(f"self.SERVO_OFFSET = {offset}")
                break
            elif result == 'r':
                print("Still drifting right - need more negative offset")
            elif result == 'l':
                print("Now drifting left - previous offset was closer")
                if test_offsets.index(offset) > 0:
                    prev_offset = test_offsets[test_offsets.index(offset) - 1]
                    print(f"Try using SERVO_OFFSET = {prev_offset}")
                break
            elif result == 'q':
                break
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    picarx.stop()
    print("\nCalibration complete!")
    print("Remember to update your code with the optimal SERVO_OFFSET value.")


# Quick test with current offset
async def test_current_offset():
    """Test the current servo offset setting."""
    system = SimpleDriftFix()
    
    if await system.initialize():
        print(f"\nTesting current SERVO_OFFSET = {system.SERVO_OFFSET}")
        print("Car will move forward 50cm...")
        input("Press Enter when ready...")
        
        await system.move_forward(50)
        
        print("Test complete!")
        print("Did the car go straight? If not, run calibration:")
        print("python simple_drift_fix.py --calibrate")
    
    await system.shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            print("Starting servo offset calibration...")
            asyncio.run(find_servo_offset())
        elif sys.argv[1] == "--test":
            print("Testing current servo offset...")
            asyncio.run(test_current_offset())
        else:
            print("Usage:")
            print("  python simple_drift_fix.py              # Run navigation")
            print("  python simple_drift_fix.py --calibrate  # Find optimal servo offset")
            print("  python simple_drift_fix.py --test       # Test current offset")
    else:
        # Run main navigation
        system = SimpleDriftFix()
        asyncio.run(system.run())