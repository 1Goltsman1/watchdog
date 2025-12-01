#!/usr/bin/env python3
"""
Check Twilio account status and verify phone number
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from twilio.rest import Client

# Get credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
your_phone = os.getenv('YOUR_PHONE_NUMBER')

print("="*60)
print("Twilio Account Verification")
print("="*60)

try:
    client = Client(account_sid, auth_token)

    # Get account info
    account = client.api.accounts(account_sid).fetch()
    print(f"\n✓ Connected to Twilio!")
    print(f"  Account Status: {account.status}")
    print(f"  Account Type: {account.type}")

    # Check if trial
    if account.type == 'Trial':
        print("\n⚠️  This is a TRIAL account")
        print("   You can only call VERIFIED phone numbers")
        print("")

        # List verified numbers
        print("Checking verified phone numbers...")
        try:
            verified = client.outgoing_caller_ids.list(limit=20)

            if verified:
                print(f"\nVerified Numbers ({len(verified)}):")
                for number in verified:
                    print(f"  ✓ {number.phone_number}")

                # Check if user's phone is verified
                user_phone_verified = any(number.phone_number == your_phone for number in verified)

                if user_phone_verified:
                    print(f"\n✅ Your phone {your_phone} is verified!")
                    print("   Phone calls should work.")
                else:
                    print(f"\n❌ Your phone {your_phone} is NOT verified!")
                    print("\nTo verify your phone number:")
                    print("1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
                    print(f"2. Click 'Add a new number'")
                    print(f"3. Enter: {your_phone}")
                    print(f"4. Choose 'Call' verification")
                    print(f"5. Answer the call and enter the code")
                    print(f"\nAfter verification, test again!")
            else:
                print(f"\n⚠️  No verified numbers found")
                print(f"\nVerify your phone {your_phone}:")
                print("https://console.twilio.com/us1/develop/phone-numbers/manage/verified")

        except Exception as e:
            print(f"\nCouldn't list verified numbers: {e}")
            print("\nManually verify your phone at:")
            print("https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
    else:
        print(f"\n✅ Full account - no restrictions")
        print(f"   You can call any number")

    print("\n" + "="*60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPossible issues:")
    print("  - Account SID incorrect")
    print("  - Auth Token incorrect")
    print("  - Account suspended")
    print("  - Network issue")

