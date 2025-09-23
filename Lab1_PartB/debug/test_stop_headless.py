# test_stop_headless.py
# Headless test for stop sign detection (no GUI display)

import cv2
import time
import asyncio
from object_detection import ObjectDetector

async def test_stop_sign_detection_headless():
    """
    Test stop sign detection without GUI display.
    Instructions: Hold phone with stop sign image ~12 inches from camera.
    """
    print("=== CS 437 Step 7: Stop Sign Detection Test (Headless) ===")
    print("Instructions:")
    print("1. Hold phone with stop sign image ~12 inches from camera")
    print("2. System will detect and print stop sign results")  
    print("3. Press Ctrl+C to quit")
    print("4. Images will be saved automatically when stop signs detected")
    print()
    
    # Initialize detector
    detector = ObjectDetector(
        model_path='efficientdet_lite0.tflite',
        confidence_threshold=0.3,
        max_results=10
    )
    
    try:
        detector.start_detection()
        print("Object detector started...")
        
        capture_count = 0
        last_detection_time = time.time()
        
        while True:
            # Get latest detection
            result = detector.get_latest_detection()
            
            if result:
                frame = result['frame']
                detections = result['detections']
                halt_needed = result['halt_needed']
                
                # Look for stop signs specifically
                stop_signs = [d for d in detections 
                             if 'stop' in d['class_name'].lower()]
                
                # Create annotated frame for saving
                annotated_frame = frame.copy()
                
                for detection in detections:
                    bbox = detection['bbox']
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # Color code: red for stop signs, green for others
                    color = (0, 0, 255) if 'stop' in class_name.lower() else (0, 255, 0)
                    thickness = 3 if 'stop' in class_name.lower() else 2
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(annotated_frame, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add performance info
                stats = detector.get_performance_stats()
                info_text = [
                    f"FPS: {stats['fps']:.1f}",
                    f"Inference: {stats['inference_time_ms']:.1f}ms",
                    f"Detections: {len(detections)}",
                    f"Stop Signs: {len(stop_signs)}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Highlight if HALT condition
                if halt_needed:
                    cv2.rectangle(annotated_frame, (0, 0), (320, 50), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "HALT DETECTED!", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Print detection results every 2 seconds to avoid spam
                current_time = time.time()
                if current_time - last_detection_time > 2.0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Detection Results:")
                    print(f"  FPS: {stats['fps']:.1f} | Inference: {stats['inference_time_ms']:.1f}ms")
                    
                    if stop_signs:
                        print(f"  🛑 STOP SIGNS DETECTED ({len(stop_signs)}):")
                        for i, stop_sign in enumerate(stop_signs):
                            print(f"    #{i+1}: Confidence: {stop_sign['confidence']:.3f}")
                            print(f"         Bounding box: {stop_sign['bbox']}")
                            if stop_sign['confidence'] > 0.5:
                                print(f"         ✅ HIGH CONFIDENCE - HALT TRIGGERED!")
                        
                        # Auto-save when stop signs detected
                        capture_filename = f"stop_detection_{capture_count:03d}.jpg"
                        cv2.imwrite(capture_filename, annotated_frame)
                        print(f"  📸 Auto-saved: {capture_filename}")
                        capture_count += 1
                        
                    elif detections:
                        detected_classes = [d['class_name'] for d in detections]
                        print(f"  Other objects: {', '.join(set(detected_classes))}")
                    else:
                        print(f"  No objects detected")
                    
                    last_detection_time = current_time
            
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        detector.stop_detection()
        
        # Print summary
        print("\n=== Test Summary ===")
        print("Expected behavior:")
        print("- Stop signs should be detected with red bounding boxes")
        print("- High confidence (>0.5) should trigger HALT condition") 
        print("- System should achieve ~1 FPS detection rate")
        print("- Low latency (~100ms) for real-time response")
        print("\nFor lab report:")
        print("- Note detection accuracy at different distances")
        print("- Record FPS and inference times achieved")
        print("- Test with different lighting conditions")
        print("- Try various angles and stop sign sizes")
        print(f"- {capture_count} images captured for analysis")

def test_model_loading():
    """Test if TensorFlow Lite model loads correctly."""
    print("=== Model Loading Test ===")
    
    try:
        detector = ObjectDetector()
        print("✅ Model loaded successfully")
        
        stats = detector.get_performance_stats()
        print(f"Initial stats: {stats}")
        
        detector.stop_detection()
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if efficientdet_lite0.tflite exists")
        print("2. Verify tflite-support or tensorflow-lite is installed")
        print("3. Run setup script: bash setup_step7.sh")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-model":
        test_model_loading()
    else:
        print("Running headless mode - no GUI display")
        print("Images will be saved automatically when stop signs are detected")
        asyncio.run(test_stop_sign_detection_headless())