#!/usr/bin/env python3
"""
Test stop sign detection specifically
"""

import asyncio
from object_detection import ObjectDetector

async def test_stop_sign_detection():
    """Test if the system can detect stop signs"""
    
    print("=== Stop Sign Detection Test ===")
    print("Point the camera at a stop sign and press Ctrl+C when done")
    print()
    
    detector = ObjectDetector(
        model_path='efficientdet_lite0.tflite',
        confidence_threshold=0.3  # Lower threshold to catch more detections
    )
    
    try:
        detector.start_detection()
        await asyncio.sleep(2)  # Let it initialize
        
        detection_count = 0
        stop_sign_count = 0
        
        while detection_count < 50:  # Check 50 detection cycles
            result = detector.get_latest_detection()
            if result and 'detections' in result:
                detection_count += 1
                print(f"Detection {detection_count}:")
                
                for detection in result['detections']:
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    print(f"  {class_name}: {confidence:.3f}")
                    
                    if 'stop_sign' in class_name.lower():
                        stop_sign_count += 1
                        print(f"  *** STOP SIGN DETECTED! (#{stop_sign_count}) ***")
                
                print()
            
            await asyncio.sleep(0.5)  # Check every 0.5 seconds
        
        print(f"=== Results ===")
        print(f"Total detections checked: {detection_count}")
        print(f"Stop signs detected: {stop_sign_count}")
        
        if stop_sign_count > 0:
            print("✅ Stop sign detection is working!")
        else:
            print("❌ No stop signs detected - check positioning or lighting")
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    asyncio.run(test_stop_sign_detection())