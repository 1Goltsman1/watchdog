import time
import os
from twilio.rest import Client
import requests

# --- CONFIGURATION ---
# Get these from environment variables or .env file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")  # Your Twilio number
YOUR_PHONE_NUMBER = os.getenv("YOUR_PHONE_NUMBER", "")  # Your personal number

# Pushover (Free alternative to Twilio for notifications)
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN", "")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Global state for rate limiting
last_call_time = 0
CALL_COOLDOWN = 25  # Seconds between calls

def send_push_notification(tracker_id, duration):
    """Send push notification via Pushover"""
    print(f"\n[ALERT] 🚨 Person {tracker_id} has been near the bike for {duration:.1f}s! Sending Push Notification...")
    
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        print("[SKIPPED] Pushover not configured. Set PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN.")
        return False
    
    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": "⚠️ Bike Alert",
            "message": f"Person {tracker_id} near bike for {duration:.1f}s!",
            "priority": 1,
            "sound": "siren"
        })
        
        if response.status_code == 200:
            print("✅ Push notification sent successfully!")
            return True
        else:
            print(f"❌ Push notification failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Push notification error: {e}")
        return False

def make_phone_call(tracker_id, duration):
    """Trigger Twilio voice call"""
    global last_call_time
    current_time = time.time()
    
    if current_time - last_call_time < CALL_COOLDOWN:
        print(f"[SKIPPED] ⏳ Call cooldown active. Skipping call for Person {tracker_id}.")
        return False
    
    print(f"\n[URGENT] ☎️ Person {tracker_id} has been near the bike for {duration:.1f}s! INITIATING PHONE CALL...")
    
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("[SKIPPED] Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.")
        print("(Would have called you here)")
        last_call_time = current_time
        return True  # Pretend it worked for testing
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        call = client.calls.create(
            twiml=f'<Response><Say voice="alice">Alert! Someone has been near your bike for {int(duration)} seconds. Please check your security camera.</Say></Response>',
            to=YOUR_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"✅ Call initiated! Call SID: {call.sid}")
        last_call_time = current_time
        return True
    except Exception as e:
        print(f"❌ Call failed: {e}")
        return False

def send_telegram_alert(message, image_path=None):
    """Send Telegram alert with optional image

    Args:
        message (str): Alert message to send
        image_path (str, optional): Path to image file to attach

    Returns:
        bool: True if successful, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[SKIPPED] Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False

    try:
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

        if image_path and os.path.exists(image_path):
            # Send photo with caption
            url = f"{base_url}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': TELEGRAM_CHAT_ID,
                    'caption': message,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, data=data, files=files, timeout=10)
        else:
            # Send text message only
            url = f"{base_url}/sendMessage"
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=10)

        if response.status_code == 200:
            print("✅ Telegram alert sent successfully!")
            return True
        else:
            print(f"❌ Telegram alert failed: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Telegram alert error: Image file not found: {image_path}")
        return False
    except requests.exceptions.Timeout:
        print("❌ Telegram alert error: Request timeout (network issue)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Telegram alert error: Connection failed (network down)")
        return False
    except Exception as e:
        print(f"❌ Telegram alert error: {e}")
        return False

# Standalone test
if __name__ == "__main__":
    print("Testing alert service...")
    print("\n1. Testing Push Notification...")
    send_push_notification(123, 25.5)

    print("\n2. Testing Telegram (text only)...")
    send_telegram_alert("🚨 Test Alert: Person detected near bike!")

    print("\n3. Testing Phone Call...")
    make_phone_call(123, 45.0)

    print("\n4. Testing Cooldown...")
    make_phone_call(124, 50.0)  # Should be skipped
