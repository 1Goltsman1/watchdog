#!/usr/bin/env python3
"""
Send current camera view to Telegram
Helps verify bike is in frame and zone is correct
"""

import os
import sys
import glob

# Load .env file BEFORE importing alert_service
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

from alert_service import send_telegram_alert

def main():
    print("="*60)
    print("Send Camera View to Telegram")
    print("="*60)

    # Find the most recent camera view image
    view_images = glob.glob("/tmp/camera_view_*.jpg")

    if not view_images:
        print("\n❌ No camera view image found!")
        print("\nPlease run first:")
        print("  python3 check_camera_view.py")
        return

    # Get most recent
    latest_image = max(view_images, key=os.path.getctime)
    print(f"\n✓ Found image: {latest_image}")
    print(f"  Size: {os.path.getsize(latest_image) / 1024:.1f} KB")

    # Send to Telegram
    print("\nSending to Telegram...")
    message = (
        "📹 <b>Camera View Check</b>\n\n"
        "🔴 Red polygon = Bike zone\n"
        "✅ Your bike should be inside the red area\n\n"
        "If zone doesn't cover your bike:\n"
        "1. Run: python3 zone_selector.py\n"
        "2. Click around bike to create new zone\n"
        "3. Update ZONE_POLYGON in script"
    )

    success = send_telegram_alert(message, latest_image)

    if success:
        print("\n✅ Camera view sent to Telegram!")
        print("\nCheck your Telegram to see:")
        print("  - Current camera view")
        print("  - Red polygon showing bike zone")
        print("\nIs your bike inside the red area?")
    else:
        print("\n❌ Failed to send to Telegram")
        print("Check your Telegram credentials in .env file")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
