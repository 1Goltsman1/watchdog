#!/usr/bin/env python3
"""
Test script for Telegram alerts
Run this to verify your Telegram credentials are working
"""

import os
import sys

# Load .env file BEFORE importing alert_service
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")
    print("   To use .env file, install: pip3 install python-dotenv")

import cv2
import numpy as np
from alert_service import send_telegram_alert

def create_test_image():
    """Create a simple test image"""
    # Create a 640x480 test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add blue background
    img[:, :] = (100, 50, 0)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'BIKE MONITOR TEST', (50, 240), font, 1.5, (255, 255, 255), 3)

    # Add a rectangle (simulating detection box)
    cv2.rectangle(img, (200, 150), (440, 350), (0, 255, 0), 3)
    cv2.putText(img, 'Person ID: 0', (210, 140), font, 0.7, (0, 255, 0), 2)

    # Save test image
    test_image_path = '/tmp/test_telegram.jpg'
    cv2.imwrite(test_image_path, img)
    print(f"✓ Test image created: {test_image_path}")

    return test_image_path

def main():
    print("="*60)
    print("Telegram Alert Test")
    print("="*60)

    # Check environment variables
    print("\n1. Checking configuration...")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        print("\nTo set it up:")
        print("1. Create a .env file in this directory")
        print("2. Add: TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("\nOr export it:")
        print("export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
        return
    else:
        print(f"✓ TELEGRAM_BOT_TOKEN: {bot_token[:10]}...{bot_token[-5:]}")

    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID not set!")
        print("\nTo set it up:")
        print("1. Add to .env file: TELEGRAM_CHAT_ID=your_chat_id")
        print("\nOr export it:")
        print("export TELEGRAM_CHAT_ID='your_chat_id'")
        return
    else:
        print(f"✓ TELEGRAM_CHAT_ID: {chat_id}")

    # Test 1: Text-only message
    print("\n2. Testing text-only message...")
    test_message = (
        "🚨 <b>Telegram Test - Text Only</b>\n\n"
        "This is a test message from your Bike Monitor system.\n"
        "If you receive this, text alerts are working! ✅"
    )
    success = send_telegram_alert(test_message)

    if not success:
        print("\n❌ Text message test failed!")
        print("Please check:")
        print("  - Bot token is correct")
        print("  - Chat ID is correct")
        print("  - You have started a conversation with the bot")
        print("  - Internet connection is working")
        return

    # Test 2: Message with image
    print("\n3. Testing message with image...")
    test_image_path = create_test_image()

    test_message_with_image = (
        "🚨 <b>Telegram Test - With Image</b>\n\n"
        "👤 Person ID: 0 (test)\n"
        "⏱ Duration: 25.0s\n"
        "📍 Status: In bike zone\n\n"
        "If you see this image, photo alerts are working! ✅"
    )
    success = send_telegram_alert(test_message_with_image, test_image_path)

    if success:
        print("\n✅ All tests passed!")
        print("\nYour Telegram alerts are configured correctly.")
        print("You should have received:")
        print("  1. A text-only message")
        print("  2. A message with a test image")
    else:
        print("\n⚠️ Image test failed (but text worked)")
        print("Check that the image file exists and is readable")

    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"\n✓ Cleaned up test image")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    main()
