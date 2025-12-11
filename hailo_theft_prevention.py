import cv2
import time
import sys
import signal
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                             InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)

# Load .env file BEFORE importing alert_service
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

# Import alert service
from alert_service import send_push_notification, make_phone_call, send_telegram_alert

# Import web dashboard
import threading
try:
    from web_dashboard import app, update_frame, update_stats, is_monitoring_enabled
    WEB_DASHBOARD_ENABLED = True
except ImportError:
    print("[WARNING] Flask not installed - web dashboard disabled")
    WEB_DASHBOARD_ENABLED = False
    # Dummy function if web dashboard is not available
    def is_monitoring_enabled():
        return True

# --- CONFIGURATION ---
# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Uniview NVR path: /unicast/c1/s1/live (Channel 1, Substream - better for Pi decoding)
RTSP_URL = "rtsp://Dair:A123456a%21@80.178.150.137:554/unicast/c1/s1/live"
HEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolox_s_leaky_hailo8.hef")  # YOLOX-S for Hailo-8 (Apache 2.0)
CONFIDENCE_THRESHOLD = 0.45  # Lowered for dark environment (was 0.5)
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame to reduce load
HEADLESS_MODE = True  # Disable display window (for SSH/testing)
WEB_DASHBOARD_PORT = 5000  # Web dashboard port (access at http://PI_IP:5000)

# Image enhancement for dark environment
BRIGHTNESS_BOOST = 30  # Add brightness (0-100, higher = brighter)
CONTRAST_BOOST = 1.2   # Multiply contrast (1.0 = no change, >1.0 = more contrast)

# Optimize RTSP for stability (TCP with increased timeout and buffering)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "buffer_size;10485760|"      # 10MB buffer
    "max_delay;5000000|"         # 5s max delay for smoothing
    "reorder_queue_size;10|"     # Allow reordering for lost packets
    "stimeout;60000000|"         # 60s socket timeout (prevents 30s disconnect)
    "timeout;0"                  # Infinite read timeout
)

# Fix Qt wayland warning - use offscreen for headless
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# Define the "Bike Zone" polygon (Normalized coordinates 0-1: [x, y])
# Bike-specific detection zone (restored)
ZONE_POLYGON = np.array([[0.237, 0.861], [0.412, 0.628], [0.597, 0.729], [0.596, 0.849], [0.550, 0.979], [0.500, 0.970]])

# Alert Thresholds (seconds)
ALERT_TIME_THRESHOLD = 20  # Telegram alert after 20 seconds
CALL_TIME_THRESHOLD = 40   # Phone call after 40 seconds
CALL_COOLDOWN = 25  # Seconds between calls

# --- STATE TRACKING ---
zone_entry_times = {}  # tracker_id -> first entry timestamp
last_seen_times = {}   # tracker_id -> last detection timestamp
alert_state = defaultdict(lambda: {"alert_sent": False, "call_sent": False})

# Grace period: keep person in zone for 5s after last detection (prevents flicker)
GRACE_PERIOD = 5.0  # seconds
last_call_time = 0

# Global variables for shutdown handler
shutdown_stats = {
    'start_time': None,
    'reconnect_count': 0,
    'total_detections': 0,
    'total_alerts': 0
}

# Graceful shutdown handler
def signal_handler(sig, frame):
    print('\n[INFO] Shutting down gracefully...')
    if shutdown_stats['start_time']:
        uptime = time.time() - shutdown_stats['start_time']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"Session Statistics:")
        print(f"  Uptime: {hours}h {minutes}m {seconds}s")
        print(f"  Reconnections: {shutdown_stats['reconnect_count']}")
        print(f"  Total detections: {shutdown_stats['total_detections']}")
        print(f"  Total alerts: {shutdown_stats['total_alerts']}")

        # Send shutdown alert to Telegram
        shutdown_message = (
            f"🔴 <b>Bike Monitor Disarmed</b>\n\n"
            f"⏹ System has been stopped\n"
            f"⏱ Uptime: {hours}h {minutes}m {seconds}s\n"
            f"👥 Total detections: {shutdown_stats['total_detections']}\n"
            f"🚨 Total alerts: {shutdown_stats['total_alerts']}\n"
            f"🔄 Reconnections: {shutdown_stats['reconnect_count']}\n"
            f"🕒 Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_alert(shutdown_message)

    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Simple ByteTrack-like tracker (Hailo doesn't have built-in tracker)
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = 30
        
    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, conf, class_id]
        returns: list of [x1, y1, x2, y2, conf, class_id, tracker_id]
        """
        if len(detections) == 0:
            # Increment disappeared counter for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        tracked = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Find closest existing track
            min_dist = float('inf')
            matched_id = None
            for track_id, track in self.tracks.items():
                dist = np.sqrt((cx - track['cx'])**2 + (cy - track['cy'])**2)
                if dist < min_dist and dist < 100:  # Max distance threshold
                    min_dist = dist
                    matched_id = track_id
            
            if matched_id is not None:
                # Update existing track
                self.tracks[matched_id] = {'cx': cx, 'cy': cy, 'disappeared': 0}
                tracked.append([x1, y1, x2, y2, conf, class_id, matched_id])
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'cx': cx, 'cy': cy, 'disappeared': 0}
                tracked.append([x1, y1, x2, y2, conf, class_id, new_id])
        
        return tracked

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def postprocess_yolox_nms(output, conf_threshold=0.5, img_width=1920, img_height=1080):
    """
    Convert Hailo NMS output to detections for YOLOX-S

    Output format: HAILO NMS BY CLASS(number of classes: 80, maximum bounding boxes per class: 100)
    YOLOX-S uses COCO classes, class 0 = person
    """
    detections = []

    if not output or len(output) == 0:
        return detections

    # DEBUG: Print output structure on first call
    if not hasattr(postprocess_yolox_nms, 'call_count'):
        postprocess_yolox_nms.call_count = 0
    postprocess_yolox_nms.call_count += 1
    if postprocess_yolox_nms.call_count == 1:
        print(f"[DEBUG] === YOLOX-S Output Structure ===")
        print(f"[DEBUG] Output type: {type(output)}, len: {len(output) if hasattr(output, '__len__') else 'N/A'}")

    try:
        # YOLOX NMS output: output[0][class_id] contains detections for that class
        # For COCO: class 0 = person
        if isinstance(output, (list, np.ndarray)) and len(output) > 0:
            if hasattr(output[0], '__len__') and len(output[0]) > 0:
                person_detections = output[0][0]  # Class 0 = person in COCO

                if hasattr(person_detections, '__iter__'):
                    for det in person_detections:
                        try:
                            if not hasattr(det, '__len__') or len(det) < 5:
                                continue

                            # NMS format: [y_min, x_min, y_max, x_max, score]
                            det_arr = np.array(det).flatten()
                            if len(det_arr) < 5:
                                continue

                            y_min, x_min, y_max, x_max, score = det_arr[:5]

                            if score > conf_threshold:
                                # Convert normalized coords to pixels
                                x1 = int(x_min * img_width)
                                y1 = int(y_min * img_height)
                                x2 = int(x_max * img_width)
                                y2 = int(y_max * img_height)

                                # Clamp to image bounds
                                x1 = max(0, min(x1, img_width))
                                y1 = max(0, min(y1, img_height))
                                x2 = max(0, min(x2, img_width))
                                y2 = max(0, min(y2, img_height))

                                detections.append([x1, y1, x2, y2, float(score), 0])  # class_id=0 for person
                        except Exception as e:
                            pass  # Skip malformed detections

    except Exception as e:
        print(f"[WARNING] YOLOX postprocessing error: {e}")

    return detections


def postprocess_ssd_nms(output, conf_threshold=0.5, img_width=1920, img_height=1080):
    """
    Convert Hailo NMS output to detections for SSD MobileNet V1

    Output format: HAILO NMS BY CLASS(number of classes: 90, maximum bounding boxes per class: 20)
    SSD MobileNet V1 uses COCO classes, class 1 = person
    """
    detections = []

    if not output or len(output) == 0:
        return detections

    # DEBUG: Print output structure once every 100 calls
    if not hasattr(postprocess_ssd_nms, 'call_count'):
        postprocess_ssd_nms.call_count = 0
    postprocess_ssd_nms.call_count += 1
    if postprocess_ssd_nms.call_count == 1:
        print(f"[DEBUG] === SSD MobileNet V1 Output Structure ===")
        print(f"[DEBUG] Output type: {type(output)}, len: {len(output) if hasattr(output, '__len__') else 'N/A'}")
        if isinstance(output, (list, np.ndarray)) and len(output) > 0:
            print(f"[DEBUG] Classes available: {len(output[0]) if hasattr(output[0], '__len__') else 'N/A'}")
            if hasattr(output[0], '__len__'):
                # Check class 0
                if len(output[0]) > 0:
                    class0_dets = len(output[0][0]) if hasattr(output[0][0], '__len__') else 0
                    print(f"[DEBUG] Class 0 detections: {class0_dets}")
                    if class0_dets > 0:
                        print(f"[DEBUG] Class 0 sample: {output[0][0][0] if hasattr(output[0][0][0], '__len__') else output[0][0][0]}")
                # Check class 1
                if len(output[0]) > 1:
                    class1_dets = len(output[0][1]) if hasattr(output[0][1], '__len__') else 0
                    print(f"[DEBUG] Class 1 detections: {class1_dets}")
                    if class1_dets > 0:
                        print(f"[DEBUG] Class 1 sample: {output[0][1][0] if hasattr(output[0][1][0], '__len__') else output[0][1][0]}")

    try:
        # Hailo NMS output is organized by class
        # output is a list/array where each element corresponds to a class
        # For COCO, class 0 = background, class 1 = person

        # Try to access class 1 (person) detections
        # The exact structure depends on how the model was compiled
        # Common format: output[0][class_id] contains detections for that class

        if isinstance(output, (list, np.ndarray)):
            # Try different possible structures
            # NOTE: Hailo COCO labels might use class 0 for person (from research)
            # Standard COCO uses class 1 for person
            # We'll check BOTH to be safe

            person_class_ids = [1, 0]  # Try class 1 first (standard), then class 0 (Hailo variant)

            for person_class_id in person_class_ids:
                # Try to get person detections from this class ID
                if len(output) > 0 and hasattr(output[0], '__len__') and len(output[0]) > person_class_id:
                    person_detections = output[0][person_class_id]

                    if hasattr(person_detections, '__iter__') and len(person_detections) > 0:
                        for i, det in enumerate(person_detections):
                            try:
                                if not hasattr(det, '__len__') or len(det) < 5:
                                    continue

                                # Common NMS format: [y_min, x_min, y_max, x_max, score]
                                # Handle various numpy array types
                                det_arr = np.array(det).flatten()
                                if len(det_arr) < 5:
                                    continue

                                y_min, x_min, y_max, x_max, score = det_arr[:5]

                                if score > conf_threshold:
                                    # Convert normalized coords to pixels
                                    x1 = int(x_min * img_width)
                                    y1 = int(y_min * img_height)
                                    x2 = int(x_max * img_width)
                                    y2 = int(y_max * img_height)

                                    # Clamp to image bounds
                                    x1 = max(0, min(x1, img_width))
                                    y1 = max(0, min(y1, img_height))
                                    x2 = max(0, min(x2, img_width))
                                    y2 = max(0, min(y2, img_height))

                                    detections.append([x1, y1, x2, y2, float(score), person_class_id])
                            except Exception as e:
                                # Only print first error to avoid spam
                                if i == 0 and person_class_id == 1:
                                    print(f"[DEBUG] Det parsing error: {e}")

                        # If we found detections, don't check other class IDs
                        if len(detections) > 0:
                            break
            
            # Option 2: Flattened format with class IDs in the data
            if len(detections) == 0:
                for item in output:
                    if isinstance(item, (list, np.ndarray)) and len(item) >= 6:
                        # Format: [class_id, score, y_min, x_min, y_max, x_max] or similar
                        class_id = int(item[0])
                        if class_id == 1:  # Person class
                            score = float(item[1])
                            if score > conf_threshold:
                                y_min, x_min, y_max, x_max = item[2:6]
                                x1 = int(x_min * img_width)
                                y1 = int(y_min * img_height)
                                x2 = int(x_max * img_width)
                                y2 = int(y_max * img_height)
                                
                                x1 = max(0, min(x1, img_width))
                                y1 = max(0, min(y1, img_height))
                                x2 = max(0, min(x2, img_width))
                                y2 = max(0, min(y2, img_height))
                                
                                detections.append([x1, y1, x2, y2, score, 1])
    
    except Exception as e:
        print(f"[WARNING] Postprocessing error: {e}")
        pass

    return detections

def start_web_dashboard():
    """Start Flask web dashboard in background thread"""
    if WEB_DASHBOARD_ENABLED:
        print(f"🌐 Starting web dashboard on port {WEB_DASHBOARD_PORT}...")
        print(f"   Access at: http://<PI_IP>:{WEB_DASHBOARD_PORT}")

        def run_flask():
            app.run(host='0.0.0.0', port=WEB_DASHBOARD_PORT, debug=False, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        time.sleep(1)  # Give Flask time to start
        print("✓ Web dashboard started")
    else:
        print("[INFO] Web dashboard disabled (Flask not installed)")


def main():
    # Start web dashboard first
    start_web_dashboard()

    # System Diagnostics
    print("="*60)
    print("Bike Theft Prevention System - Starting Up")
    print(f"System started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Check NumPy version
    print(f"NumPy version: {np.__version__}")

    # Check temperature
    try:
        temp_output = os.popen("vcgencmd measure_temp").read()
        print(f"CPU Temperature: {temp_output.strip()}")
    except:
        print("Temperature monitoring unavailable")

    # Check Hailo device
    hailo_model = "Hailo device"
    try:
        device_check = os.popen("hailortcli fw-control identify").read()
        if "Board Name: Hailo-8L" in device_check:
            hailo_model = "Hailo-8L (13 TOPS)"
        elif "Board Name: Hailo-8" in device_check:
            hailo_model = "Hailo-8 (26 TOPS)"
        print(f"✓ {hailo_model} detected")
    except:
        print("Could not check Hailo device")

    print("="*60)

    # 1. Initialize Hailo Device
    print(f"Initializing {hailo_model}...")
    print(f"🔧 FIXED: Using correct input size 300x300 and UINT8 format for SSD MobileNet V1")

    # Use default params (scheduling algorithm is set automatically)
    with VDevice() as target:
        hef = HEF(HEF_MODEL_PATH)
        
        # Configure network
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # ✅ FIXED: Use UINT8 format (quantized=True) for SSD MobileNet V1
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        # Activate network group
        with network_group.activate(network_group_params):
            # 2. Setup Video Capture
            print(f"Opening video stream...")
            print(f"  URL: {RTSP_URL[:40]}...")
            print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print(f"  Processing every {PROCESS_EVERY_N_FRAMES} frames")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

            # Optimize buffer for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 25)
            
            if not cap.isOpened():
                print("Error: Could not open video stream.")
                return
            
            # Wait for stream to stabilize
            print("Waiting for stream to stabilize...")
            for _ in range(30):
                cap.read()
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 3. Setup Zone
            abs_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
            print(f"[ZONE_INFO] Frame size: {frame_width}x{frame_height}")
            print(f"[ZONE_INFO] Zone polygon (pixels): {abs_polygon.tolist()}")

            # 4. Initialize Tracker
            tracker = SimpleTracker()

            # 5. Performance counters
            frame_count = 0
            fps_count = 0
            fps_start = time.time()
            last_temp_check = time.time()
            reconnect_count = 0
            start_time = time.time()

            # Set global shutdown stats
            shutdown_stats['start_time'] = start_time

            print("System armed. Press Ctrl+C to quit.")
            print(f"Configuration: Threshold={CONFIDENCE_THRESHOLD}, Frame skip={PROCESS_EVERY_N_FRAMES}")

            # Send startup alert to Telegram
            startup_message = (
                f"✅ <b>Bike Monitor Armed</b>\n\n"
                f"🟢 System is now monitoring your bike\n"
                f"📹 Camera: Connected\n"
                f"🤖 AI: {hailo_model}\n"
                f"⚙️ Confidence: {CONFIDENCE_THRESHOLD}\n"
                f"🔧 Model: SSD MobileNet V1 (300x300, UINT8)\n"
                f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram_alert(startup_message)

            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                # Get stream names from dictionary keys
                input_name = list(input_vstreams_params.keys())[0]
                output_name = list(output_vstreams_params.keys())[0]
                
                while True:
                    try:
                        ret, frame = cap.read()
                    except Exception as e:
                        print(f"[ERROR] Exception reading frame: {e}")
                        ret = False

                    if not ret:
                        reconnect_count += 1
                        shutdown_stats['reconnect_count'] = reconnect_count
                        uptime = time.time() - start_time
                        print(f"[WARNING] Stream lost after {uptime:.0f}s (reconnect #{reconnect_count})")
                        cap.release()
                        time.sleep(2)
                        print("[INFO] Reconnecting to stream...")
                        try:
                            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            frame_count = 0  # Reset frame counter
                            print("[INFO] Reconnection successful")
                        except Exception as e:
                            print(f"[ERROR] Reconnection failed: {e}")
                            time.sleep(5)
                        continue

                    frame_count += 1

                    # Temperature monitoring every 30 seconds
                    current_time = time.time()
                    if current_time - last_temp_check > 30:
                        try:
                            temp_output = os.popen("vcgencmd measure_temp").read()
                            temp = temp_output.replace("temp=", "").replace("'C\n", "")
                            print(f"[TEMP] CPU: {temp}°C")
                            last_temp_check = current_time
                        except:
                            pass

                    # Enhance brightness/contrast for dark environment (BEFORE detection)
                    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_BOOST, beta=BRIGHTNESS_BOOST)

                    # Check if monitoring is enabled
                    monitoring_active = is_monitoring_enabled()

                    # Skip frames to reduce processing load or if monitoring is disabled
                    if frame_count % PROCESS_EVERY_N_FRAMES != 0 or not monitoring_active:
                        # Still show frame even if not processing
                        if not HEADLESS_MODE:
                            try:
                                cv2.polylines(frame, [abs_polygon], True, (0, 0, 255), 2)
                                cv2.imshow("Hailo Theft Prevention", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            except:
                                pass

                        # Update web dashboard even when monitoring is disabled
                        if WEB_DASHBOARD_ENABLED and not monitoring_active:
                            uptime = int(time.time() - start_time)
                            update_frame('camera1', frame)
                            update_stats(
                                active_tracking=0,
                                alerts_sent=shutdown_stats['total_alerts'],
                                fps=0,
                                uptime=uptime,
                                status="System Paused"
                            )
                        continue

                    # Prepare input: YOLOX-S expects 640x640 UINT8
                    input_data = cv2.resize(frame, (640, 640))  # YOLOX-S expects 640x640
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                    input_data = input_data.astype(np.uint8)  # Keep as UINT8, no normalization
                    
                    # Add batch dimension
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # Run inference
                    try:
                        input_dict = {input_name: input_data}
                        output_dict = infer_pipeline.infer(input_dict)
                        output_data = output_dict[output_name]
                    except Exception as e:
                        print(f"[ERROR] Inference failed: {e}")
                        continue

                    # Use YOLOX NMS postprocessing (80 classes, person = class 0)
                    detections = postprocess_yolox_nms(output_data, CONFIDENCE_THRESHOLD, frame_width, frame_height)

                    # FPS counter
                    fps_count += 1
                    current_fps = 0.0
                    if fps_count % 30 == 0:
                        fps_elapsed = time.time() - fps_start
                        current_fps = 30 / fps_elapsed
                        print(f"[INFO] FPS: {current_fps:.1f} (Processing every {PROCESS_EVERY_N_FRAMES} frames)")

                        # DEBUG: Periodic detection check
                        print(f"[DEBUG] Detections after filtering (conf>{CONFIDENCE_THRESHOLD}): {len(detections)}")

                        fps_count = 0
                        fps_start = time.time()

                    # Detection logging
                    if len(detections) > 0:
                        shutdown_stats['total_detections'] += len(detections)
                        print(f"[DETECT] Found {len(detections)} person(s)")

                    # Track
                    tracked_detections = tracker.update(detections)

                    if len(tracked_detections) > 0:
                        print(f"[TRACK] Tracking {len(tracked_detections)} person(s)")

                    current_time = time.time()
                    active_ids = set()
                    
                    # Check zone
                    for det in tracked_detections:
                        x1, y1, x2, y2, conf, class_id, tracker_id = det
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        # DEBUG: Print zone check details (reduce spam)
                        in_zone = point_in_polygon((cx, cy), abs_polygon)
                        if tracker_id not in last_seen_times or (current_time - last_seen_times.get(tracker_id, 0)) > 2:
                            print(f"[ZONE_DEBUG] ID:{tracker_id} Center:({int(cx)},{int(cy)}) InZone:{in_zone}")

                        if in_zone:
                            active_ids.add(tracker_id)
                            last_seen_times[tracker_id] = current_time  # Update last seen time

                            if tracker_id not in zone_entry_times:
                                zone_entry_times[tracker_id] = current_time
                                print(f"✓ Person {tracker_id} entered the Bike Zone.")
                            
                            duration = current_time - zone_entry_times[tracker_id]
                            
                            # Check Thresholds
                            if duration > ALERT_TIME_THRESHOLD and not alert_state[tracker_id]["alert_sent"]:
                                # Save current frame for Telegram alert
                                alert_image_path = "/tmp/bike_alert.jpg"
                                try:
                                    cv2.imwrite(alert_image_path, frame)
                                except Exception as e:
                                    print(f"[WARNING] Failed to save alert image: {e}")
                                    alert_image_path = None

                                # Send alerts
                                send_push_notification(tracker_id, duration)

                                # Send Telegram alert with image
                                telegram_message = (
                                    f"🚨 <b>Bike Alert!</b>\n\n"
                                    f"👤 Person ID: {tracker_id}\n"
                                    f"⏱ Duration in zone: {duration:.1f}s\n"
                                    f"📍 Status: In bike zone\n"
                                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                send_telegram_alert(telegram_message, alert_image_path)

                                alert_state[tracker_id]["alert_sent"] = True
                                shutdown_stats['total_alerts'] += 1

                            if duration > CALL_TIME_THRESHOLD and not alert_state[tracker_id]["call_sent"]:
                                # Save current frame for second Telegram alert
                                alert_image_path_45s = "/tmp/bike_alert_45s.jpg"
                                try:
                                    cv2.imwrite(alert_image_path_45s, frame)
                                except Exception as e:
                                    print(f"[WARNING] Failed to save 45s alert image: {e}")
                                    alert_image_path_45s = None

                                # Make phone call
                                if make_phone_call(tracker_id, duration):
                                    # Send second Telegram alert with updated photo
                                    telegram_message_urgent = (
                                        f"🚨🚨 <b>URGENT: Phone Call Initiated!</b>\n\n"
                                        f"☎️ Calling you now!\n"
                                        f"👤 Person ID: {tracker_id}\n"
                                        f"⏱ Duration in zone: {duration:.1f}s\n"
                                        f"⚠️ STILL in bike zone - CHECK NOW!\n"
                                        f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    send_telegram_alert(telegram_message_urgent, alert_image_path_45s)

                                    alert_state[tracker_id]["call_sent"] = True
                                    shutdown_stats['total_alerts'] += 1
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID: {tracker_id} | {int(current_time - zone_entry_times.get(tracker_id, current_time))}s"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Cleanup with grace period (5 seconds)
                    for tracker_id in list(zone_entry_times.keys()):
                        if tracker_id not in active_ids:
                            # Check how long since we last saw this person
                            time_since_last_seen = current_time - last_seen_times.get(tracker_id, 0)

                            # Only remove if they've been gone longer than grace period
                            if time_since_last_seen > GRACE_PERIOD:
                                print(f"✗ Person {tracker_id} left the Bike Zone (gone for {time_since_last_seen:.1f}s)")
                                del zone_entry_times[tracker_id]
                                if tracker_id in alert_state:
                                    del alert_state[tracker_id]
                                if tracker_id in last_seen_times:
                                    del last_seen_times[tracker_id]
                            # else: Keep them in zone, timer continues!
                    
                    # Draw polygon
                    cv2.polylines(frame, [abs_polygon], True, (0, 0, 255), 2)

                    # Update web dashboard
                    if WEB_DASHBOARD_ENABLED:
                        uptime = int(time.time() - start_time)
                        update_frame('camera1', frame)  # Update frame for camera1
                        update_stats(
                            active_tracking=len(tracked_detections),
                            alerts_sent=shutdown_stats['total_alerts'],
                            fps=current_fps if current_fps > 0 else 0,
                            uptime=uptime,
                            status="System Active - Monitoring"
                        )

                    # Display (Headless friendly)
                    if not HEADLESS_MODE:
                        try:
                            cv2.imshow("Hailo Theft Prevention", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except Exception:
                            # If no display available, just continue (headless mode)
                            pass
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
