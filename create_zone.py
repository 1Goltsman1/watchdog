#!/usr/bin/env python3
"""
Interactive Zone Selector for Hailo YOLOX-S Detection
Click points to create a polygon zone, press 's' to save, 'c' to clear, 'q' to quit
"""
import cv2
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                             InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)
import os
import json

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RTSP_URL = "rtsp://Dair:A123456a%21@80.178.150.137:554/unicast/c1/s1/live"
HEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolox_s_leaky_hailo8.hef")
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to see more detections

# RTSP optimization
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;10485760|max_delay;5000000|"
    "reorder_queue_size;10|stimeout;60000000|timeout;0"
)

# Global variables for zone selection
zone_points = []
frame_width = 0
frame_height = 0


def postprocess_yolox_nms(output, conf_threshold=0.3, img_width=1920, img_height=1080):
    """Postprocess YOLOX NMS output - Class 0 = person"""
    detections = []
    
    if not output or len(output) == 0:
        return detections
    
    try:
        if isinstance(output, (list, np.ndarray)) and len(output) > 0:
            person_detections = output[0]  # Class 0 = person in COCO
            
            if hasattr(person_detections, '__iter__'):
                for det in person_detections:
                    if len(det) >= 5:
                        y_min, x_min, y_max, x_max, score = det[:5]
                        
                        if score > conf_threshold:
                            x1 = int(x_min * img_width)
                            y1 = int(y_min * img_height)
                            x2 = int(x_max * img_width)
                            y2 = int(y_max * img_height)
                            
                            x1 = max(0, min(x1, img_width))
                            y1 = max(0, min(y1, img_height))
                            x2 = max(0, min(x2, img_width))
                            y2 = max(0, min(y2, img_height))
                            
                            detections.append([x1, y1, x2, y2, score, 0])
    except Exception as e:
        print(f"[WARNING] Postprocessing error: {e}")
    
    return detections


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to create zone polygon"""
    global zone_points, frame_width, frame_height
    
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append([x, y])
        print(f"Added point {len(zone_points)}: ({x}, {y})")
        
        # Convert to normalized coordinates
        norm_x = x / frame_width
        norm_y = y / frame_height
        print(f"  Normalized: ({norm_x:.3f}, {norm_y:.3f})")


def save_zone():
    """Save zone polygon to file in normalized coordinates"""
    global zone_points, frame_width, frame_height
    
    if len(zone_points) < 3:
        print("❌ Need at least 3 points to create a zone!")
        return False
    
    # Convert to normalized coordinates (0-1 range)
    normalized_polygon = []
    for point in zone_points:
        norm_x = point[0] / frame_width
        norm_y = point[1] / frame_height
        normalized_polygon.append([norm_x, norm_y])
    
    # Save to file
    zone_data = {
        "zone_polygon": normalized_polygon,
        "absolute_coords": zone_points,
        "frame_size": {
            "width": frame_width,
            "height": frame_height
        }
    }
    
    with open('bike_zone.json', 'w') as f:
        json.dump(zone_data, f, indent=2)
    
    # Generate Python code
    print("\n" + "="*70)
    print("✅ Zone saved to bike_zone.json")
    print("="*70)
    print("\nCopy this into your hailo_theft_prevention.py:\n")
    print("ZONE_POLYGON = np.array([", end="")
    for i, point in enumerate(normalized_polygon):
        if i > 0:
            print(", ", end="")
        print(f"[{point[0]:.3f}, {point[1]:.3f}]", end="")
    print("])")
    print("\n" + "="*70)
    
    return True


def main():
    global zone_points, frame_width, frame_height
    
    print("="*70)
    print("🎯 INTERACTIVE ZONE SELECTOR - HAILO YOLOX-S")
    print("="*70)
    print("\nInstructions:")
    print("  • Left-click to add zone boundary points")
    print("  • Press 's' to SAVE the zone")
    print("  • Press 'c' to CLEAR and start over")
    print("  • Press 'q' to QUIT")
    print("\nTip: Click points in order around your bike zone")
    print("="*70 + "\n")
    
    with VDevice() as target:
        hef = HEF(HEF_MODEL_PATH)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # YOLOX-S uses UINT8 input (640x640)
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        with network_group.activate(network_group_params):
            print("Opening video stream...")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print("❌ Error: Could not open video stream.")
                return
            
            # Wait for stream to stabilize
            for _ in range(30):
                cap.read()
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✓ Stream opened: {frame_width}x{frame_height}")
            print(f"✓ Model: YOLOX-S (640x640 UINT8 input)")
            print(f"✓ Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print("\nWindow opened - Start clicking points!\n")
            
            # Create window and set mouse callback
            cv2.namedWindow("Zone Selector", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Zone Selector", mouse_callback)
            
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                input_name = list(input_vstreams_params.keys())[0]
                output_name = list(output_vstreams_params.keys())[0]
                
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠ Stream lost, retrying...")
                        cap.release()
                        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        continue
                    
                    frame_count += 1
                    display_frame = frame.copy()
                    
                    # Run detection every 5 frames
                    if frame_count % 5 == 0:
                        # Preprocess for YOLOX-S (640x640 UINT8)
                        input_data = cv2.resize(frame, (640, 640))
                        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                        input_data = input_data.astype(np.uint8)
                        input_data = np.expand_dims(input_data, axis=0)
                        
                        # Inference
                        try:
                            input_dict = {input_name: input_data}
                            output_dict = infer_pipeline.infer(input_dict)
                            output_data = output_dict[output_name]
                            
                            # Postprocess
                            detections = postprocess_yolox_nms(output_data, CONFIDENCE_THRESHOLD,
                                                               frame_width, frame_height)
                            
                            # Draw detections (GREEN boxes)
                            for det in detections:
                                x1, y1, x2, y2, conf, _ = det
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"{conf:.2f}", (x1, y1-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Inference error: {e}")
                    
                    # Draw current zone polygon (BLUE)
                    if len(zone_points) > 0:
                        pts = np.array(zone_points, dtype=np.int32)
                        
                        # Draw lines between points
                        for i in range(len(zone_points)):
                            cv2.circle(display_frame, tuple(zone_points[i]), 5, (255, 0, 0), -1)
                            if i > 0:
                                cv2.line(display_frame, tuple(zone_points[i-1]), 
                                        tuple(zone_points[i]), (255, 0, 0), 2)
                        
                        # Close the polygon if we have 3+ points
                        if len(zone_points) >= 3:
                            cv2.line(display_frame, tuple(zone_points[-1]), 
                                    tuple(zone_points[0]), (255, 0, 0), 2)
                            cv2.fillPoly(display_frame, [pts], (255, 0, 0), lineType=cv2.LINE_AA)
                            # Make semi-transparent
                            display_frame = cv2.addWeighted(frame, 0.7, display_frame, 0.3, 0)
                    
                    # Add instructions overlay
                    instructions = [
                        f"Points: {len(zone_points)}",
                        "Left-click: Add point",
                        "S: Save zone",
                        "C: Clear",
                        "Q: Quit"
                    ]
                    
                    y_offset = 30
                    for text in instructions:
                        cv2.putText(display_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        y_offset += 30
                    
                    cv2.imshow("Zone Selector", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        if save_zone():
                            print("\n✅ Zone saved! You can continue selecting or press 'q' to quit.")
                    elif key == ord('c'):
                        zone_points = []
                        print("\n🔄 Zone cleared. Start over by clicking points.")
            
            cap.release()
            cv2.destroyAllWindows()
    
    print("\n✓ Zone selector closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
