"""
Utility functions for object detection visualization and testing
Based on TensorFlow examples for Raspberry Pi object detection
"""

import cv2
import numpy as np
from typing import List, Tuple

def visualize(
    image: np.ndarray,
    detection_result,
    score_threshold: float = 0.5
) -> np.ndarray:
    """Draws bounding boxes and labels on the input image and return it.
    
    Args:
        image: The input RGB image as a numpy array of shape [height, width, 3].
        detection_result: The list of all "Detection" entities to be visualized.
        score_threshold: The minimum confidence score for detections to be shown.
        
    Returns:
        Image with bounding boxes and labels drawn on it.
    """
    # Color palette for different object classes
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for detection in detection_result.detections:
        # Get the top category with highest score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        
        # Only show detections above threshold
        if probability < score_threshold:
            continue
        
        # Get bounding box coordinates
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        
        # Choose color based on hash of category name
        color_idx = hash(category_name) % len(COLORS)
        color = COLORS[color_idx]
        
        # Draw bounding box
        cv2.rectangle(image, start_point, end_point, color, 2)
        
        # Create label text
        label = f'{category_name}: {probability}'
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background rectangle for text
        label_start = (start_point[0], start_point[1] - text_height - baseline)
        label_end = (start_point[0] + text_width, start_point[1])
        cv2.rectangle(image, label_start, label_end, color, -1)
        
        # Draw text label
        cv2.putText(
            image, 
            label, 
            (start_point[0], start_point[1] - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    return image


def draw_map(map_array: np.ndarray, current_pos: Tuple[int, int], 
             goal_pos: Tuple[int, int], path: List = None) -> np.ndarray:
    """Draw a visual representation of the map with obstacles, position, and path.
    
    Args:
        map_array: 2D numpy array where 1=obstacle, 0=free space
        current_pos: Current position (x, y)
        goal_pos: Goal position (x, y) 
        path: Optional list of waypoints [(x, y), ...]
        
    Returns:
        RGB image representation of the map
    """
    height, width = map_array.shape
    
    # Create RGB image (scale up for visibility)
    scale = 10
    img = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
    
    # Draw map
    for i in range(height):
        for j in range(width):
            y_start, y_end = i * scale, (i + 1) * scale
            x_start, x_end = j * scale, (j + 1) * scale
            
            if map_array[i, j] == 1:
                # Obstacle - red
                img[y_start:y_end, x_start:x_end] = [0, 0, 255]
            else:
                # Free space - white
                img[y_start:y_end, x_start:x_end] = [255, 255, 255]
    
    # Draw path if provided
    if path:
        for i in range(len(path) - 1):
            start = (int(path[i][1] * scale + scale/2), int(path[i][0] * scale + scale/2))
            end = (int(path[i+1][1] * scale + scale/2), int(path[i+1][0] * scale + scale/2))
            cv2.line(img, start, end, (0, 255, 0), 2)  # Green path
    
    # Draw current position - blue circle
    center = (int(current_pos[1] * scale + scale/2), int(current_pos[0] * scale + scale/2))
    cv2.circle(img, center, scale//3, (255, 0, 0), -1)
    
    # Draw goal position - yellow circle
    goal_center = (int(goal_pos[1] * scale + scale/2), int(goal_pos[0] * scale + scale/2))
    cv2.circle(img, goal_center, scale//3, (0, 255, 255), -1)
    
    return img


def create_test_image_with_stop_sign() -> np.ndarray:
    """Create a test image with a simple stop sign for testing"""
    img = np.ones((240, 320, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw a simple red octagon as stop sign
    center = (160, 120)
    radius = 40
    
    # Octagon points
    angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 points
    points = []
    for angle in angles:
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append([x, y])
    
    points = np.array(points, np.int32)
    cv2.fillPoly(img, [points], (0, 0, 255))  # Red stop sign
    
    # Add "STOP" text
    cv2.putText(img, "STOP", (center[0]-25, center[1]+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img


def save_detection_result(image: np.ndarray, detections, filename: str):
    """Save detection result with visualizations to file"""
    try:
        # Add detection info as text overlay
        info_text = f"Detections: {len(detections)}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save image
        cv2.imwrite(filename, image)
        print(f"Detection result saved to {filename}")
    except Exception as e:
        print(f"Failed to save detection result: {e}")


def print_detection_info(detections):
    """Print detailed information about detections"""
    print(f"Found {len(detections)} detections:")
    for i, detection in enumerate(detections):
        if detection.categories:
            category = detection.categories[0]
            name = getattr(category, "category_name", "unknown")
            score = getattr(category, "score", 0.0)
            bbox = detection.bounding_box
            print(f"  {i+1}. {name} ({score:.3f}) at "
                  f"[{bbox.origin_x}, {bbox.origin_y}, {bbox.width}, {bbox.height}]")


def benchmark_detection_speed(detector_func, test_image: np.ndarray, num_iterations: int = 10):
    """Benchmark detection speed"""
    import time
    
    print(f"Benchmarking detection speed over {num_iterations} iterations...")
    
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        detections = detector_func(test_image)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Iteration {i+1}: {times[-1]:.3f}s, {len(detections)} detections")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\nBenchmark Results:")
    print(f"Average time per detection: {avg_time:.3f}s")
    print(f"Estimated FPS: {fps:.2f}")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")
    
    return avg_time, fps


# Test functions for development
def test_visualization():
    """Test visualization functions"""
    print("Testing visualization functions...")
    
    # Create test map
    test_map = np.zeros((20, 30))
    test_map[5:8, 10:15] = 1  # Obstacle
    test_map[15:18, 20:25] = 1  # Another obstacle
    
    current_pos = (2, 5)
    goal_pos = (18, 25)
    test_path = [(2, 5), (5, 5), (8, 8), (12, 12), (18, 25)]
    
    map_img = draw_map(test_map, current_pos, goal_pos, test_path)
    
    try:
        cv2.imwrite("test_map.png", map_img)
        print("Test map saved to test_map.png")
    except:
        print("Could not save test map (cv2 not available)")
    
    # Test stop sign image
    stop_sign_img = create_test_image_with_stop_sign()
    try:
        cv2.imwrite("test_stop_sign.png", stop_sign_img)
        print("Test stop sign saved to test_stop_sign.png")
    except:
        print("Could not save test stop sign image (cv2 not available)")


if __name__ == "__main__":
    test_visualization()