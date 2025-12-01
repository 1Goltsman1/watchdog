# Bike Monitor - AI-Powered Theft Prevention System

An intelligent bike monitoring system using Raspberry Pi 5 with Hailo-8 AI accelerator for real-time person detection and automated alerts.

## Features

- **Real-time Person Detection**: Uses YOLOX-S object detection model optimized for Hailo-8
- **Custom Zone Monitoring**: Define specific areas to monitor (e.g., bike parking area)
- **Multi-Channel Alerts**:
  - Telegram notifications with snapshots
  - Phone call alerts via Twilio
- **Live Web Dashboard**: View camera feed and detection status in real-time
- **Smart Tracking**: Tracks individuals with grace period to prevent false alerts
- **Dark Environment Enhancement**: Automatic brightness/contrast adjustment

## Hardware Requirements

- Raspberry Pi 5 (4GB+ recommended)
- Hailo-8 or Hailo-8L AI accelerator
- USB/CSI camera
- Stable internet connection

## Software Requirements

- Raspberry Pi OS (64-bit)
- Python 3.9+
- HailoRT SDK
- OpenCV
- Flask (for web dashboard)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd BikeMonitor
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Download the YOLOX-S model:
```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolox_s_leaky.hef -O yolox_s_leaky_hailo8.hef
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Set up your monitoring zone:
```bash
python3 simple_zone_selector.py
```

## Configuration

Create a `.env` file with the following:

```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890
```

## Usage

### Run the main monitoring system:
```bash
python3 hailo_theft_prevention.py
```

### Access the web dashboard:
Navigate to `http://<raspberry-pi-ip>:5000`

### Create/adjust monitoring zones:
```bash
python3 create_zone.py
```

## Project Structure

- `hailo_theft_prevention.py` - Main detection and alert system
- `alert_service.py` - Telegram and Twilio integration
- `web_dashboard.py` - Live video streaming dashboard
- `simple_zone_selector.py` - GUI tool for zone configuration
- `create_zone.py` - Zone creation utility
- `bike_zone.json` - Stored zone coordinates
- `cameras.json` - Camera configuration

## Alert Thresholds

- **Telegram Alert**: 20 seconds in zone
- **Phone Call**: 40 seconds in zone
- **Grace Period**: 5 seconds (prevents false resets)

## License

This project uses YOLOX-S model which is licensed under Apache 2.0, making it suitable for commercial use.

## Troubleshooting

### No detections:
- Check camera connection
- Verify model file exists (yolox_s_leaky_hailo8.hef)
- Check Hailo device: `hailortcli fw-control identify`

### False alerts:
- Adjust confidence threshold in `hailo_theft_prevention.py`
- Refine monitoring zone using `simple_zone_selector.py`
- Increase alert time thresholds

### Poor performance in low light:
- Adjust `BRIGHTNESS_BOOST` and `CONTRAST_BOOST` values
- Consider adding external lighting

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## Acknowledgments

- Hailo for the AI accelerator and model zoo
- YOLOX team for the excellent detection model
- Raspberry Pi Foundation
