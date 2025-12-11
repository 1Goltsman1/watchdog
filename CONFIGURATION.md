# Configuration Guide

Comprehensive configuration options for Watchdog security monitoring system.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Detection Parameters](#detection-parameters)
- [Alert Configuration](#alert-configuration)
- [Camera Settings](#camera-settings)
- [Zone Configuration](#zone-configuration)
- [Web Dashboard](#web-dashboard)
- [Multi-Camera Setup](#multi-camera-setup)
- [Advanced Options](#advanced-options)

## Environment Variables

Configuration file: `.env`

### Telegram Configuration

```env
# Telegram Bot Token (required for Telegram alerts)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Telegram Chat ID (your personal chat ID)
TELEGRAM_CHAT_ID=123456789
```

**How to get Telegram credentials:**

1. **Create Bot:**
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot`
   - Follow prompts to create bot
   - Save the token provided

2. **Get Chat ID:**
   - Message [@userinfobot](https://t.me/userinfobot)
   - Your chat ID will be displayed
   - Or use: `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Twilio Configuration (Phone Calls)

```env
# Twilio Account SID
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Twilio Auth Token
TWILIO_AUTH_TOKEN=your_auth_token_here

# Twilio Phone Number (from)
TWILIO_FROM_NUMBER=+15551234567

# Your Phone Number (to)
TWILIO_TO_NUMBER=+15557654321
```

**How to get Twilio credentials:**

1. Sign up at [twilio.com](https://www.twilio.com/try-twilio)
2. Navigate to Console Dashboard
3. Copy Account SID and Auth Token
4. Purchase a phone number with voice capabilities

### Optional: Pushbullet (Alternative Notification)

```env
# Pushbullet Access Token
PUSHBULLET_ACCESS_TOKEN=o.xxxxxxxxxxxxxxxxxxxxx
```

## Detection Parameters

Configuration file: `hailo_theft_prevention.py`

### Core Detection Settings

```python
# Confidence threshold for person detection (0.0 - 1.0)
# Lower = more detections (higher false positives)
# Higher = fewer detections (might miss some)
CONFIDENCE_THRESHOLD = 0.45  # Default: 0.45

# Process every Nth frame (reduce for higher accuracy, increase for performance)
PROCESS_EVERY_N_FRAMES = 3  # Default: 3 (process 1 in 3 frames)

# Grace period in seconds (prevents flickering exit/entry)
GRACE_PERIOD = 5.0  # Default: 5 seconds
```

### Image Enhancement

```python
# Brightness boost (0-100, higher = brighter)
BRIGHTNESS_BOOST = 30  # Default: 30

# Contrast multiplier (1.0 = no change, >1.0 = more contrast)
CONTRAST_BOOST = 1.2  # Default: 1.2
```

### Tracking Parameters

```python
# Maximum frames a track can disappear before being deleted
max_disappeared = 30  # Default: 30 (at 10 FPS = 3 seconds)

# Maximum pixel distance for track matching
distance_threshold = 100  # Default: 100 pixels
```

## Alert Configuration

Configuration file: `hailo_theft_prevention.py`

### Alert Thresholds

```python
# Telegram alert delay (seconds in zone before alert)
ALERT_TIME_THRESHOLD = 20  # Default: 20 seconds

# Phone call delay (seconds in zone before calling)
CALL_TIME_THRESHOLD = 40  # Default: 40 seconds

# Cooldown between phone calls (prevents spam)
CALL_COOLDOWN = 25  # Default: 25 seconds
```

### Alert Behavior

**Disable Phone Calls (Telegram only):**

```python
# In hailo_theft_prevention.py, comment out the phone call section:
# if duration > CALL_TIME_THRESHOLD and not alert_state[tracker_id]["call_sent"]:
#     make_phone_call(tracker_id, duration)
```

**Custom Alert Messages:**

Edit templates in `hailo_theft_prevention.py`:

```python
# Telegram startup message
startup_message = (
    "System Armed\n\n"
    f"Monitoring active\n"
    f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Detection alert
telegram_message = (
    "Alert Detected\n\n"
    f"Person ID: {tracker_id}\n"
    f"Duration in zone: {duration:.1f}s\n"
    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
```

## Camera Settings

Configuration file: `hailo_theft_prevention.py`

### Camera Source

**USB Camera:**
```python
RTSP_URL = "/dev/video0"  # or just: 0
```

**RTSP Network Camera:**
```python
RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream"
```

**Special characters in password:**
```python
# URL encode special characters
# @ becomes %40
# ! becomes %21
# Example: password "pass@123!" becomes "pass%40123%21"
RTSP_URL = "rtsp://admin:pass%40123%21@192.168.1.100:554/stream"
```

### RTSP Optimization

```python
# OpenCV FFmpeg capture options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"          # TCP (more reliable) or udp (lower latency)
    "buffer_size;10485760|"        # 10MB buffer (adjust for network)
    "max_delay;5000000|"           # 5s max delay (smoothing)
    "stimeout;60000000|"           # 60s socket timeout
    "timeout;0"                     # Infinite read timeout
)
```

### Camera Resolution

```python
# Set in hailo_theft_prevention.py after cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)
```

## Zone Configuration

Configuration file: `bike_zone.json`

### Manual Zone Editing

```json
{
  "polygon": [
    [0.237, 0.861],
    [0.412, 0.628],
    [0.597, 0.729],
    [0.596, 0.849],
    [0.550, 0.979],
    [0.500, 0.970]
  ]
}
```

Coordinates are normalized (0.0 to 1.0):
- `[0.0, 0.0]` = top-left corner
- `[1.0, 1.0]` = bottom-right corner

### Using Zone Selector Tool

```bash
# Interactive zone creation
python3 simple_zone_selector.py

# Controls:
# - Left click: Add point to polygon
# - Right click: Remove last point
# - 's': Save zone to bike_zone.json
# - 'c': Clear all points
# - 'q': Quit
```

### Multiple Zones

Edit code to check multiple polygons:

```python
# Define multiple zones
ZONE_POLYGONS = [
    np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.7], [0.5, 0.9]]),  # Zone 1
    np.array([[0.6, 0.5], [0.8, 0.5], [0.8, 0.7], [0.6, 0.7]])   # Zone 2
]

# Check if point is in any zone
def in_any_zone(point):
    return any(point_in_polygon(point, zone) for zone in ZONE_POLYGONS)
```

## Web Dashboard

Configuration file: `web_dashboard.py`

### Port Configuration

```python
# Change web dashboard port
WEB_DASHBOARD_PORT = 5000  # Default: 5000

# Access dashboard at: http://[PI_IP]:5000
```

### Disable Web Dashboard

```python
# Set to False in hailo_theft_prevention.py
WEB_DASHBOARD_ENABLED = False
```

### Dashboard Performance

```python
# Adjust update frequency (milliseconds)
# In templates/index.html:
setInterval(updateStatus, 2000);  # Default: 2000ms (2 seconds)
```

## Multi-Camera Setup

Configuration file: `cameras.json`

### Camera Configuration

```json
{
  "cameras": [
    {
      "id": "camera1",
      "name": "Front Entrance",
      "url": "rtsp://admin:password@192.168.1.100:554/stream",
      "zone_file": "zones/front_zone.json",
      "enabled": true
    },
    {
      "id": "camera2",
      "name": "Back Yard",
      "url": "rtsp://admin:password@192.168.1.101:554/stream",
      "zone_file": "zones/back_zone.json",
      "enabled": true
    }
  ]
}
```

### Enable Multi-Camera Mode

Currently single-camera. For multi-camera support, modify `hailo_theft_prevention.py` to:

1. Load multiple camera streams
2. Process each stream in separate thread
3. Update dashboard with per-camera stats

## Advanced Options

### Headless Mode

```python
# Disable display window (for SSH-only operation)
HEADLESS_MODE = True  # Default: True
```

### Model Selection

```python
# YOLOX-S (default, balanced)
HEF_MODEL_PATH = "yolox_s_leaky_hailo8.hef"

# Alternative models (download separately):
# - yolox_tiny (faster, less accurate)
# - yolox_m (slower, more accurate)
```

### Logging Configuration

```python
import logging

# Configure logging level
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watchdog.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Tuning

```python
# CPU affinity (bind to specific cores)
import os
os.sched_setaffinity(0, {2, 3})  # Use cores 2 and 3

# Thread priority
import psutil
p = psutil.Process()
p.nice(-10)  # Higher priority (requires sudo)
```

### Network Retry Logic

```python
# Reconnection attempts
MAX_RECONNECT_ATTEMPTS = 10  # Default: unlimited (None)
RECONNECT_DELAY = 2  # Seconds between attempts

# Connection timeout
STREAM_TIMEOUT = 30  # Seconds before giving up
```

## Configuration Examples

### High Security (Sensitive)

```python
CONFIDENCE_THRESHOLD = 0.35        # More sensitive
ALERT_TIME_THRESHOLD = 10          # Quick alerts (10s)
CALL_TIME_THRESHOLD = 20           # Quick escalation (20s)
PROCESS_EVERY_N_FRAMES = 1         # Process every frame
GRACE_PERIOD = 2.0                 # Short grace period
```

### Balanced (Default)

```python
CONFIDENCE_THRESHOLD = 0.45
ALERT_TIME_THRESHOLD = 20
CALL_TIME_THRESHOLD = 40
PROCESS_EVERY_N_FRAMES = 3
GRACE_PERIOD = 5.0
```

### Low False Alerts (Relaxed)

```python
CONFIDENCE_THRESHOLD = 0.60        # Less sensitive
ALERT_TIME_THRESHOLD = 30          # Slower alerts (30s)
CALL_TIME_THRESHOLD = 60           # Delayed escalation (60s)
PROCESS_EVERY_N_FRAMES = 5         # Process fewer frames
GRACE_PERIOD = 10.0                # Long grace period
```

## Validation

### Test Configuration

```bash
# Validate environment variables
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('TELEGRAM_BOT_TOKEN:', 'SET' if os.getenv('TELEGRAM_BOT_TOKEN') else 'MISSING')"

# Test camera connection
python3 -c "import cv2; cap = cv2.VideoCapture('YOUR_RTSP_URL'); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"

# Verify Hailo device
hailortcli fw-control identify
```

### Backup Configuration

```bash
# Backup current configuration
tar -czf watchdog-config-$(date +%Y%m%d).tar.gz .env bike_zone.json cameras.json hailo_theft_prevention.py

# Restore configuration
tar -xzf watchdog-config-YYYYMMDD.tar.gz
```

## Next Steps

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [README](README.md) - Main documentation
- [Contributing](CONTRIBUTING.md) - Improve the project
