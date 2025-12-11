# Installation Guide

Complete setup instructions for Watchdog security monitoring system on Raspberry Pi 5.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Setup](#hardware-setup)
- [Operating System Installation](#operating-system-installation)
- [Hailo SDK Installation](#hailo-sdk-installation)
- [Watchdog Installation](#watchdog-installation)
- [Camera Configuration](#camera-configuration)
- [Service Setup](#service-setup)
- [Verification](#verification)

## Prerequisites

### Hardware Checklist

- Raspberry Pi 5 (4GB or 8GB)
- Hailo-8 or Hailo-8L AI accelerator
- MicroSD card (32GB minimum, Class 10 or UHS-I)
- USB camera or Raspberry Pi Camera Module v3
- Power supply (5V 5A USB-C)
- Ethernet cable (recommended) or WiFi
- Monitor and keyboard (for initial setup)

### Software Requirements

- Raspberry Pi Imager
- SSH client (optional, for remote access)
- Web browser (for accessing dashboard)

## Hardware Setup

### 1. Install Hailo Module

```bash
# Power off Raspberry Pi
sudo poweroff

# Install Hailo-8 module onto PCIe slot
# Secure with provided standoffs
# Connect power if using Hailo-8

# Power on Raspberry Pi
```

### 2. Connect Camera

**For USB Camera:**
```bash
# Connect USB camera to USB 3.0 port (blue)
# Verify detection
v4l2-ctl --list-devices
```

**For Pi Camera Module:**
```bash
# Connect ribbon cable to CSI port
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
```

## Operating System Installation

### 1. Flash Raspberry Pi OS

```bash
# Download Raspberry Pi Imager
# Select OS: Raspberry Pi OS (64-bit)
# Configure settings:
#   - Set hostname: watchdog
#   - Enable SSH
#   - Set username and password
#   - Configure WiFi (optional)
# Write to microSD card
```

### 2. First Boot Configuration

```bash
# SSH into Pi (or use local terminal)
ssh pi@watchdog.local

# Update system
sudo apt update
sudo apt upgrade -y

# Install essential tools
sudo apt install -y git python3-pip python3-venv
```

## Hailo SDK Installation

### 1. Install HailoRT

```bash
# Add Hailo repository
echo "deb https://hailo-ai.github.io/ppa stable main" | sudo tee /etc/apt/sources.list.d/hailo.list
wget -qO - https://hailo-ai.github.io/ppa/hailo.gpg.key | sudo apt-key add -

# Update and install
sudo apt update
sudo apt install -y hailort

# Verify installation
hailortcli fw-control identify

# Expected output: Board Name: Hailo-8 (or Hailo-8L)
```

### 2. Configure Permissions

```bash
# Add user to hailo group
sudo usermod -a -G hailo $USER

# Reload groups (or logout/login)
newgrp hailo

# Verify permissions
groups | grep hailo
```

## Watchdog Installation

### 1. Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone Watchdog
git clone https://github.com/yourusername/watchdog.git
cd watchdog
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip3 install -r requirements.txt

# Verify OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"
```

### 3. Download AI Model

```bash
# Download YOLOX-S model (Apache 2.0 license)
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolox_s_leaky.hef \
     -O yolox_s_leaky_hailo8.hef

# Verify download
ls -lh yolox_s_leaky_hailo8.hef
# Expected size: ~15MB
```

## Camera Configuration

### 1. Test Camera Feed

**For USB Camera:**
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera (replace /dev/video0 with your device)
ffplay /dev/video0
```

**For RTSP Camera:**
```bash
# Test RTSP stream
ffplay rtsp://username:password@camera_ip:554/stream
```

### 2. Configure Camera Source

Edit `hailo_theft_prevention.py`:

```python
# For USB camera
RTSP_URL = "/dev/video0"  # or use camera index: 0

# For RTSP camera
RTSP_URL = "rtsp://username:password@camera_ip:554/stream"
```

### 3. Set Up Monitoring Zone

```bash
# Run zone selector tool
python3 simple_zone_selector.py

# Instructions:
# 1. Click to add polygon points
# 2. Press 's' to save
# 3. Press 'q' to quit

# Zone saved to: bike_zone.json
```

## Service Setup

### 1. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options.

### 2. Install Systemd Service

```bash
# Copy service file
sudo cp bike-monitor.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/bike-monitor.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable bike-monitor.service

# Start service
sudo systemctl start bike-monitor.service

# Check status
sudo systemctl status bike-monitor.service
```

### 3. View Logs

```bash
# Follow logs in real-time
sudo journalctl -u bike-monitor.service -f

# View recent logs
sudo journalctl -u bike-monitor.service -n 100

# Check for errors
sudo journalctl -u bike-monitor.service -p err
```

## Verification

### 1. Test Detection

```bash
# Stop service if running
sudo systemctl stop bike-monitor.service

# Run manually to see output
python3 hailo_theft_prevention.py

# Expected output:
# - Hailo device detected
# - Camera connected
# - System armed
# - Detection logs

# Press Ctrl+C to stop
```

### 2. Access Web Dashboard

```bash
# Find Pi's IP address
hostname -I

# Open browser to:
# http://[PI_IP]:5000

# You should see:
# - Live camera feed
# - Detection statistics
# - System status
# - Control button
```

### 3. Test Alerts

```bash
# Trigger test alert
python3 -c "from alert_service import send_telegram_alert; send_telegram_alert('Test alert from Watchdog')"

# Check Telegram for message
```

## Post-Installation

### Enable Automatic Updates

```bash
# Create update script
cat > ~/update_watchdog.sh << 'EOF'
#!/bin/bash
cd ~/watchdog
git pull
pip3 install -r requirements.txt --upgrade
sudo systemctl restart bike-monitor.service
EOF

chmod +x ~/update_watchdog.sh

# Run weekly via cron
(crontab -l 2>/dev/null; echo "0 3 * * 0 ~/update_watchdog.sh") | crontab -
```

### Configure Firewall (Optional)

```bash
# Install UFW
sudo apt install -y ufw

# Allow SSH
sudo ufw allow 22/tcp

# Allow web dashboard
sudo ufw allow 5000/tcp

# Enable firewall
sudo ufw enable
```

### Optimize Performance

```bash
# Increase GPU memory (helps with video processing)
sudo raspi-config
# Performance Options > GPU Memory > 256

# Overclock (optional, reduces thermal throttling)
# Edit /boot/config.txt
sudo nano /boot/config.txt

# Add:
# over_voltage=6
# arm_freq=2400

# Reboot
sudo reboot
```

## Troubleshooting

### Hailo Not Detected

```bash
# Check PCIe detection
lspci | grep Hailo

# Reinstall HailoRT
sudo apt install --reinstall hailort

# Check kernel module
lsmod | grep hailo
```

### Camera Issues

```bash
# USB camera not found
sudo apt install v4l-utils
v4l2-ctl --list-devices

# Permissions issue
sudo usermod -a -G video $USER
newgrp video

# RTSP timeout
# Increase timeout in hailo_theft_prevention.py:
# stimeout;120000000
```

### Python Dependencies

```bash
# OpenCV import error
sudo apt install -y python3-opencv
pip3 install opencv-python --upgrade

# Flask not found
pip3 install flask --upgrade
```

### Service Won't Start

```bash
# Check service logs
sudo journalctl -u bike-monitor.service -n 50

# Verify paths in service file
sudo nano /etc/systemd/system/bike-monitor.service

# Check permissions
ls -la ~/watchdog
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md) - Customize your deployment
- [Contributing Guidelines](CONTRIBUTING.md) - Improve the project
- [README](README.md) - Return to main documentation
