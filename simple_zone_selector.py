#!/usr/bin/env python3
"""
Simple Zone Selector - No AI required
Just click points on your camera feed to create a zone polygon
"""
import cv2
import numpy as np
import json
import os

# Configuration
RTSP_URL = "rtsp://Dair:A123456a%21@80.178.150.137:554/unicast/c1/s1/live"

# RTSP optimization
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;10485760|max_delay;5000000|"
    "reorder_queue_size;10|stimeout;60000000|timeout;0"
)

# Global variables
zone_points = []
frame_width = 0
frame_height = 0


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to create zone polygon"""
    global zone_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append([x, y])
        print(f"✓ Added point {len(zone_points)}: ({x}, {y})")
        
        # Convert to normalized coordinates
        norm_x = x / frame_width
        norm_y = y / frame_height
        print(f"  Normalized: ({norm_x:.3f}, {norm_y:.3f})")


def save_zone():
    """Save zone polygon to file in normalized coordinates"""
    global zone_points, frame_width, frame_height
    
    if len(zone_points) < 3:
        print("\n❌ Need at least 3 points to create a zone!")
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
    print("\n📋 Copy this into your hailo_theft_prevention.py (line 60):\n")
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
    print("🎯 SIMPLE ZONE SELECTOR")
    print("="*70)
    print("\n⚠️  IMPORTANT: Stop your main detection script first!")
    print("   (Press Ctrl+C in the other terminal)\n")
    print("\nInstructions:")
    print("  • Left-click to add zone boundary points")
    print("  • Press 's' to SAVE the zone")
    print("  • Press 'c' to CLEAR and start over")
    print("  • Press 'q' to QUIT")
    print("\nTip: Click points in order around your bike zone")
    print("="*70 + "\n")
    
    print("Opening video stream...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        print("\nTroubleshooting:")
        print("  1. Check if another script is using the camera")
        print("  2. Verify the RTSP URL is correct")
        print("  3. Check network connection")
        return
    
    # Wait for stream to stabilize
    print("Waiting for stream to stabilize...")
    for _ in range(30):
        cap.read()
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Stream opened: {frame_width}x{frame_height}")
    print("\n🖱️  Window opened - Start clicking points to draw your zone!\n")
    
    # Create window and set mouse callback
    cv2.namedWindow("Zone Selector - Click points around bike", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Zone Selector - Click points around bike", mouse_callback)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Stream lost, retrying...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Draw current zone polygon (BLUE)
        if len(zone_points) > 0:
            pts = np.array(zone_points, dtype=np.int32)
            
            # Draw points
            for i, point in enumerate(zone_points):
                # Draw circle at each point
                cv2.circle(display_frame, tuple(point), 8, (255, 0, 0), -1)
                # Draw point number
                cv2.putText(display_frame, str(i+1), 
                           (point[0]+15, point[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw lines between consecutive points
                if i > 0:
                    cv2.line(display_frame, tuple(zone_points[i-1]), 
                            tuple(point), (255, 0, 0), 3)
            
            # Close the polygon if we have 3+ points
            if len(zone_points) >= 3:
                cv2.line(display_frame, tuple(zone_points[-1]), 
                        tuple(zone_points[0]), (255, 0, 0), 3)
                
                # Fill semi-transparent
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (255, 100, 100))
                display_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                
                # Redraw the border on top
                cv2.polylines(display_frame, [pts], True, (255, 0, 0), 3)
                
                # Redraw points on top
                for i, point in enumerate(zone_points):
                    cv2.circle(display_frame, tuple(point), 8, (255, 0, 0), -1)
                    cv2.circle(display_frame, tuple(point), 10, (255, 255, 255), 2)
                    cv2.putText(display_frame, str(i+1), 
                               (point[0]+15, point[1]-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions overlay
        instructions = [
            f"Points: {len(zone_points)}/3 minimum",
            "",
            "Controls:",
            "Left-click: Add point",
            "S: Save zone",
            "C: Clear all",
            "Q: Quit"
        ]
        
        # Dark background for text
        overlay_height = len(instructions) * 30 + 20
        cv2.rectangle(display_frame, (5, 5), (280, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (5, 5), (280, overlay_height), (255, 255, 255), 2)
        
        y_offset = 30
        for text in instructions:
            if text == "":
                y_offset += 15
                continue
            cv2.putText(display_frame, text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Add footer
        if len(zone_points) >= 3:
            footer = "Zone ready! Press 'S' to save"
            cv2.putText(display_frame, footer, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        # Update window title with status
        if len(zone_points) >= 3:
            window_title = f"Zone Selector - {len(zone_points)} points - READY TO SAVE (press S)"
        else:
            window_title = f"Zone Selector - {len(zone_points)}/3 points - Click to add more"
        
        cv2.imshow(window_title, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('s') or key == ord('S'):
            print(f"\n🔑 'S' key pressed! (You have {len(zone_points)} points)")
            if len(zone_points) < 3:
                print(f"❌ Need at least 3 points. You have {len(zone_points)}.")
                print("   Click more points on the video, then press 's' again.\n")
            else:
                if save_zone():
                    print("\n✅ Zone saved! You can continue adjusting or press 'q' to quit.\n")
        elif key == ord('c') or key == ord('C'):
            print(f"\n🔑 'C' key pressed! Clearing {len(zone_points)} points...")
            zone_points = []
            print("🔄 Zone cleared. Start over by clicking points.\n")
    
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
