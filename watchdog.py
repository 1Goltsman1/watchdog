#!/usr/bin/env python3
"""
Watchdog - Monitors if bike monitor system is running
Sends alert if system is down for more than 30 seconds
Run this in background to monitor the main system
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Load .env file BEFORE importing alert_service
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

from alert_service import send_telegram_alert

# Configuration
CHECK_INTERVAL = 10  # Check every 10 seconds
DOWN_THRESHOLD = 30  # Alert if down for 30 seconds
PROCESS_NAME = "hailo_theft_prevention.py"

def is_system_running():
    """Check if the bike monitor is running"""
    try:
        # Check if process is running
        result = subprocess.run(
            ['pgrep', '-f', PROCESS_NAME],
            capture_output=True,
            text=True
        )
        return result.returncode == 0  # 0 means process found
    except Exception as e:
        print(f"Error checking process: {e}")
        return False

def main():
    print("="*60)
    print("Bike Monitor Watchdog")
    print("="*60)
    print(f"Monitoring: {PROCESS_NAME}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Alert threshold: {DOWN_THRESHOLD}s")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    down_since = None
    alert_sent = False
    last_status = None

    while True:
        try:
            is_running = is_system_running()
            current_time = time.time()

            if is_running:
                if last_status == False:
                    # System just came back up
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ System is back online")

                    if alert_sent:
                        # Send recovery message
                        recovery_message = (
                            f"✅ <b>Bike Monitor Back Online</b>\n\n"
                            f"🟢 System has recovered\n"
                            f"⏱ Downtime: {int(current_time - down_since)}s\n"
                            f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        send_telegram_alert(recovery_message)

                # Reset down tracking
                down_since = None
                alert_sent = False

            else:
                # System is down
                if down_since is None:
                    down_since = current_time
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  System is down")

                downtime = int(current_time - down_since)

                if downtime >= DOWN_THRESHOLD and not alert_sent:
                    # Send alert
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 Sending down alert!")

                    alert_message = (
                        f"🚨 <b>Bike Monitor is DOWN!</b>\n\n"
                        f"🔴 System has been offline for {downtime} seconds\n"
                        f"⚠️ Your bike is NOT being monitored!\n"
                        f"🕒 Down since: {datetime.fromtimestamp(down_since).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"<b>Please check the system immediately!</b>"
                    )
                    send_telegram_alert(alert_message)
                    alert_sent = True

                elif downtime < DOWN_THRESHOLD:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  System down for {downtime}s (threshold: {DOWN_THRESHOLD}s)")

            last_status = is_running
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n[INFO] Watchdog stopped by user")
            break
        except Exception as e:
            print(f"[ERROR] Watchdog error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
