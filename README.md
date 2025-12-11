# Watchdog - AI-Powered Security Monitoring System

A real-time security monitoring system powered by Raspberry Pi 5 and Hailo-8 AI accelerator. Features intelligent person detection, customizable monitoring zones, and multi-channel alerting capabilities.

## Overview

Watchdog leverages edge AI processing to provide fast, privacy-focused security monitoring without relying on cloud services. Perfect for monitoring valuable assets, property entrances, or any area requiring intelligent surveillance.

## Key Features

- **Real-Time AI Detection**: YOLOX-S object detection optimized for Hailo-8 (26 TOPS)
- **Custom Zone Monitoring**: Define specific areas to monitor with polygon-based detection zones
- **Multi-Camera Support**: Monitor multiple camera feeds simultaneously
- **Live Web Dashboard**: Browser-based interface with live video streaming and system metrics
- **Multi-Channel Alerts**:
  - Instant Telegram notifications with image snapshots
  - Phone call escalation via Twilio
- **Intelligent Tracking**: Persistent person tracking with grace period to reduce false alerts
- **Low-Light Enhancement**: Automatic brightness and contrast adjustment for dark environments
- **Remote Control**: Enable/disable monitoring via web interface
- **Headless Operation**: Runs efficiently without display requirements

## Hardware Requirements

- Raspberry Pi 5 (4GB RAM minimum, 8GB recommended)
- Hailo-8 or Hailo-8L AI accelerator module
- USB camera or Raspberry Pi Camera Module v3
- MicroSD card (32GB minimum)
- Stable internet connection (for alerts)
- Optional: UPS for continuous operation

## Software Requirements

- Raspberry Pi OS (64-bit) Bookworm or later
- Python 3.9 or higher
- HailoRT SDK 4.17.0+
- OpenCV 4.5+
- Flask 2.0+ (for web dashboard)

## Quick Start

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

```bash
# Clone repository
git clone https://github.com/yourusername/watchdog.git
cd watchdog

# Install dependencies
pip3 install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Download AI model
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolox_s_leaky.hef -O yolox_s_leaky_hailo8.hef

# Set up monitoring zone
python3 simple_zone_selector.py

# Run the system
python3 hailo_theft_prevention.py
```

## Documentation

- [Installation Guide](INSTALLATION.md) - Complete setup instructions
- [Configuration Guide](CONFIGURATION.md) - Customize your deployment
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

## Project Structure

```
watchdog/
├── hailo_theft_prevention.py    # Main detection and monitoring system
├── alert_service.py              # Alert notification handlers
├── web_dashboard.py              # Web interface backend
├── simple_zone_selector.py      # Zone configuration tool
├── create_zone.py                # Alternative zone setup utility
├── templates/
│   └── index.html                # Web dashboard frontend
├── bike_zone.json                # Monitoring zone coordinates
├── cameras.json                  # Camera configuration
├── bike-monitor.service          # Systemd service file
└── requirements.txt              # Python dependencies
```

## Alert Configuration

Default thresholds (configurable in `hailo_theft_prevention.py`):

- **Telegram Alert**: Triggered after 20 seconds of presence in monitored zone
- **Phone Call**: Escalation after 40 seconds of continued presence
- **Grace Period**: 5-second window to prevent false exits

## Performance

- **Detection Speed**: ~8-10 FPS on Raspberry Pi 5 with Hailo-8
- **Latency**: <100ms per frame
- **Power Consumption**: ~15W (Pi 5 + Hailo-8)
- **Memory Usage**: ~500MB RAM

## Use Cases

- Vehicle and motorcycle theft prevention
- Property entrance monitoring
- Warehouse security
- Construction site surveillance
- Package delivery monitoring
- Restricted area access control

## Security & Privacy

- **Local Processing**: All AI inference runs on-device
- **No Cloud Dependencies**: Video never leaves your network
- **Encrypted Communications**: TLS for all alert transmissions
- **Credential Protection**: Environment-based configuration

## Troubleshooting

### No Detections
- Verify camera connection: `v4l2-ctl --list-devices`
- Check Hailo device: `hailortcli fw-control identify`
- Ensure model file exists: `ls -lh yolox_s_leaky_hailo8.hef`
- Review confidence threshold settings

### False Alerts
- Adjust `CONFIDENCE_THRESHOLD` (default: 0.45)
- Refine monitoring zones using `simple_zone_selector.py`
- Increase `ALERT_TIME_THRESHOLD` for less sensitivity

### Performance Issues
- Reduce `PROCESS_EVERY_N_FRAMES` (process more frames)
- Lower camera resolution in RTSP settings
- Monitor CPU temperature: `vcgencmd measure_temp`

### Connection Errors
- Verify RTSP URL format and credentials
- Test network connectivity to camera
- Check firewall rules for port 554 (RTSP)

## License

This project uses the YOLOX-S model under Apache 2.0 License, making it suitable for both personal and commercial use.

See [LICENSE](LICENSE) for complete terms.

## Acknowledgments

- [Hailo](https://hailo.ai/) for the AI accelerator platform and model zoo
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) team for the detection model
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for the hardware platform

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This system is designed for legitimate security purposes. Ensure compliance with local privacy and surveillance laws in your jurisdiction.
