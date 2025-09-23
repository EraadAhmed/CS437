#!/usr/bin/env python3
"""
Power Tuning Script for CS 437 Self-Driving Car
Allows real-time tuning of drive and turn power settings
"""

import time
import sys
import threading
from dataclasses import dataclass

# Hardware imports with error handling
try:
    from picarx import Picarx
    HW_AVAILABLE = True
    print("+ PiCarX hardware available")
except ImportError:
    print("x PiCarX not available - simulation mode only")
    HW_AVAILABLE = False

@dataclass
class TuningConfig:
    """Current tuning parameters"""
    drive_power: int = 35
    turn_power: int = 35
    servo_offset: float = -2.2
    move_duration: float = 5.0
    turn_duration: float = 0.5

class PowerTuner:
    """Interactive power tuning system"""
    
    def __init__(self):
        self.config = TuningConfig()
        self.picar = None
        self.running = True
        
        if HW_AVAILABLE:
            try:
                self.picar = Picarx(servo_pins=["P0", "P1", "P3"])
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                print("+ PiCarX initialized successfully")
            except Exception as e:
                print(f"x PiCarX initialization failed: {e}")
                self.picar = None
        
        self.setup_input_thread()
    
    def setup_input_thread(self):
        """Setup background input handling"""
        self.input_thread = threading.Thread(target=self.input_handler, daemon=True)
        self.input_thread.start()
    
    def input_handler(self):
        """Handle user input in background"""
        while self.running:
            try:
                command = input().strip().lower()
                self.process_command(command)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
    
    def process_command(self, command):
        """Process user commands"""
        try:
            if command == 'q' or command == 'quit':
                self.running = False
                return
            
            elif command == 'h' or command == 'help':
                self.show_help()
            
            elif command == 'status' or command == 'status':
                self.show_status()
            
            elif command == 'w':  # Forward
                self.test_forward()
            
            elif command == 's':  # Backward  
                self.test_backward()
            
            elif command == 'a':  # Turn left
                self.test_turn_left()
            
            elif command == 'd':  # Turn right
                self.test_turn_right()
            
            elif command == 'x':  # Stop
                self.stop_car()
            
            elif command.startswith('dp '):  # Drive power
                try:
                    power = int(command.split()[1])
                    if 0 <= power <= 100:
                        self.config.drive_power = power
                        print(f"+ Drive power set to {power}")
                    else:
                        print("x Drive power must be 0-100")
                except (ValueError, IndexError):
                    print("x Usage: dp <power> (0-100)")

            elif command.startswith('tp '):  # Turn power
                try:
                    power = int(command.split()[1])
                    if 0 <= power <= 100:
                        self.config.turn_power = power
                        print(f"+ Turn power set to {power}")
                    else:
                        print("x Turn power must be 0-100")
                except (ValueError, IndexError):
                    print("x Usage: tp <power> (0-100)")

            elif command.startswith('so '):  # Servo offset
                try:
                    offset = int(command.split()[1])
                    if -30 <= offset <= 30:
                        self.config.servo_offset = offset
                        if self.picar:
                            self.picar.set_dir_servo_angle(offset)
                        print(f"+ Servo offset set to {offset}")
                    else:
                        print("x Servo offset must be -30 to 30")
                except (ValueError, IndexError):
                    print("x Usage: so <offset> (-30 to 30)")

            elif command.startswith('mt '):  # Move time
                try:
                    duration = float(command.split()[1])
                    if 0.1 <= duration <= 5.0:
                        self.config.move_duration = duration
                        print(f"+ Move duration set to {duration}s")
                    else:
                        print("x Move duration must be 0.1-5.0 seconds")
                except (ValueError, IndexError):
                    print("x Usage: mt <seconds> (0.1-5.0)")

            elif command.startswith('tt '):  # Turn time
                try:
                    duration = float(command.split()[1])
                    if 0.1 <= duration <= 3.0:
                        self.config.turn_duration = duration
                        print(f"+ Turn duration set to {duration}s")
                    else:
                        print("x Turn duration must be 0.1-3.0 seconds")
                except (ValueError, IndexError):
                    print("x Usage: tt <seconds> (0.1-3.0)")

            elif command == 'test':
                self.run_test_sequence()
            
            elif command == 'save':
                self.save_config()
            
            else:
                print(f"x Unknown command: {command}. Type 'h' for help")
        
        except Exception as e:
            print(f"x Command error: {e}")

    def test_forward(self):
        """Test forward movement"""
        print(f"Testing forward: power={self.config.drive_power}, duration={self.config.move_duration}s")
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                self.picar.forward(self.config.drive_power)
                time.sleep(self.config.move_duration)
                self.picar.stop()
                print("+ Forward test complete")
            except Exception as e:
                print(f"x Forward test failed: {e}")
        else:
            print("+ Forward test (simulation)")

    def test_backward(self):
        """Test backward movement"""
        print(f"Testing backward: power={self.config.drive_power}, duration={self.config.move_duration}s")
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                self.picar.backward(self.config.drive_power)
                time.sleep(self.config.move_duration)
                self.picar.stop()
                print("+ Backward test complete")
            except Exception as e:
                print(f"x Backward test failed: {e}")
        else:
            print("+ Backward test (simulation)")

    def test_turn_left(self):
        """Test left turn"""
        print(f"Testing left turn: power={self.config.turn_power}, duration={self.config.turn_duration}s")
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(-25 + self.config.servo_offset)
                self.picar.forward(self.config.turn_power)
                time.sleep(self.config.turn_duration)
                self.picar.stop()
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                print("+ Left turn test complete")
            except Exception as e:
                print(f"x Left turn test failed: {e}")
        else:
            print("+ Left turn test (simulation)")

    def test_turn_right(self):
        """Test right turn"""
        print(f"Testing right turn: power={self.config.turn_power}, duration={self.config.turn_duration}s")
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(25 + self.config.servo_offset)
                self.picar.forward(self.config.turn_power)
                time.sleep(self.config.turn_duration)
                self.picar.stop()
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                print("+ Right turn test complete")
            except Exception as e:
                print(f"x Right turn test failed: {e}")
        else:
            print("+ Right turn test (simulation)")

    def stop_car(self):
        """Emergency stop"""
        print(" EMERGENCY STOP")
        if self.picar:
            try:
                self.picar.stop()
                self.picar.set_dir_servo_angle(self.config.servo_offset)
                print("+ Car stopped")
            except Exception as e:
                print(f"x Stop failed: {e}")
        else:
            print("+ Stop (simulation)")
    
    def run_test_sequence(self):
        """Run a complete test sequence"""
        print("🧪 Running complete test sequence...")
        print("Forward -> Right -> Backward -> Left -> Stop")
        
        try:
            self.test_forward()
            time.sleep(0.5)
            
            self.test_turn_right()
            time.sleep(0.5)
            
            self.test_backward()
            time.sleep(0.5)
            
            self.test_turn_left()
            time.sleep(0.5)
            
            self.stop_car()
            print("+ Test sequence complete")
            
        except Exception as e:
            print(f"x Test sequence failed: {e}")
            self.stop_car()
    
    def save_config(self):
        """Save current configuration to file"""
        config_text = f"""# Power Tuning Configuration
# Generated by tune_power_settings.py

# Drive settings
DRIVE_POWER = {self.config.drive_power}
TURN_POWER = {self.config.turn_power}
SERVO_OFFSET = {self.config.servo_offset}

# Timing settings
MOVE_DURATION = {self.config.move_duration}
TURN_DURATION = {self.config.turn_duration}

# Usage in your main code:
# config = SystemConfig()
# config.DRIVE_POWER = {self.config.drive_power}
# config.TURN_POWER = {self.config.turn_power}
# config.SERVO_OFFSET = {self.config.servo_offset}
"""
        
        try:
            with open('tuned_power_config.py', 'w') as f:
                f.write(config_text)
            print("+ Configuration saved to 'tuned_power_config.py'")
        except Exception as e:
            print(f"x Save failed: {e}")
    
    def show_status(self):
        """Show current configuration"""
        print("\n" + "="*50)
        print(" CURRENT POWER SETTINGS")
        print("="*50)
        print(f"Drive Power:    {self.config.drive_power}")
        print(f"Turn Power:     {self.config.turn_power}")
        print(f"Servo Offset:   {self.config.servo_offset}")
        print(f"Move Duration:  {self.config.move_duration}s")
        print(f"Turn Duration:  {self.config.turn_duration}s")
        print(f"Hardware:       {'Connected' if self.picar else 'Simulation'}")
        print("="*50 + "\n")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print(" POWER TUNING CONTROLS")
        print("="*60)
        print("MOVEMENT CONTROLS:")
        print("  w          - Test forward movement")
        print("  s          - Test backward movement") 
        print("  a          - Test left turn")
        print("  d          - Test right turn")
        print("  x          - Emergency stop")
        print()
        print("POWER SETTINGS:")
        print("  dp <power> - Set drive power (0-100)")
        print("  tp <power> - Set turn power (0-100)")
        print("  so <offset>- Set servo offset (-30 to 30)")
        print()
        print("TIMING SETTINGS:")
        print("  mt <time>  - Set move duration (0.1-5.0s)")
        print("  tt <time>  - Set turn duration (0.1-3.0s)")
        print()
        print("UTILITIES:")
        print("  test       - Run complete test sequence")
        print("  status     - Show current settings")
        print("  save       - Save config to file")
        print("  help/h     - Show this help")
        print("  quit/q     - Exit program")
        print()
        print("TIPS:")
        print(" - Start with low power (20-30) and increase gradually")
        print(" - Adjust servo offset if car doesn't drive straight")
        print(" - Use shorter durations for precise movements")
        print(" - Save settings when you find good values")
        print("="*60 + "\n")
    
    def run(self):
        """Main run loop"""
        print("\n PiCarX Power Tuning System")
        print("Type 'help' for commands, 'quit' to exit")
        self.show_status()
        
        try:
            while self.running:
                time.sleep(0.1)  # Prevent busy loop
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n Cleaning up...")
        self.running = False
        
        if self.picar:
            try:
                self.picar.stop()
                print("+ Car stopped")
            except:
                pass
        
        print("+ Cleanup complete")


def main():
    """Main entry point"""
    tuner = PowerTuner()
    tuner.run()


if __name__ == "__main__":
    main()