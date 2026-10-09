"""
deploy_ngrok.py -- Public Deployment Script for BEVFusion Dashboard
====================================================================
This script:
  1. Starts the Flask backend (which also serves the frontend)
  2. Opens an ngrok tunnel and prints the PUBLIC URL

HOW TO USE:
    python deploy_ngrok.py

Everyone with the printed link can access your project!
"""

import subprocess
import sys
import os
import threading
import time

# ---- Setup pyngrok to use local ngrok.exe ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NGROK_EXE = os.path.join(PROJECT_ROOT, 'ngrok.exe')

try:
    from pyngrok import ngrok, conf
except ImportError:
    print("[INFO] Installing pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

# Point pyngrok to local ngrok.exe (avoids SSL download issues)
if os.path.exists(NGROK_EXE):
    pyngrok_config = conf.PyngrokConfig(ngrok_path=NGROK_EXE)
    conf.set_default(pyngrok_config)
    print(f"[OK] Using local ngrok binary: {NGROK_EXE}")
else:
    print(f"[WARN] ngrok.exe not found at {NGROK_EXE}")
    print("      Will attempt to download ngrok automatically...")

# ---- Start Flask in background ----
BACKEND_PORT = 5000

def run_flask():
    sys.path.insert(0, PROJECT_ROOT)
    from backend.app import app
    app.run(debug=False, host='0.0.0.0', port=BACKEND_PORT, use_reloader=False, threaded=True)

print("\n[INFO] Starting Flask backend on port 5000...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print("[INFO] Waiting for backend to initialize (loading nuScenes dataset)...")
print("       This may take 10-15 seconds on first run...")
time.sleep(15)

# ---- Open ngrok tunnel ----
print("\n[INFO] Opening public tunnel via ngrok...")
print("       NOTE: For a stable/branded URL, add your free ngrok authtoken:")
print("       https://dashboard.ngrok.com/get-started/your-authtoken")
print("       Then run:  ngrok config add-authtoken YOUR_TOKEN")

try:
    tunnel = ngrok.connect(BACKEND_PORT, "http")
    public_url = tunnel.public_url

    print("\n" + "=" * 65)
    print("  DEPLOYMENT SUCCESSFUL!")
    print("=" * 65)
    print(f"\n  PUBLIC LINK:  {public_url}")
    print(f"\n  Share this link with ANYONE!")
    print(f"  They can open the BEVFusion dashboard from anywhere.")
    print(f"\n  IMPORTANT: This link works ONLY while this window is open.")
    print(f"             Keep this terminal running. Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    # Keep alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        print("[INFO] Goodbye!")

except Exception as e:
    print(f"\n[ERROR] Could not open ngrok tunnel: {e}")
    print("\nFIX: You need a free ngrok account.")
    print("  1. Go to: https://ngrok.com  -> Sign up (free)")
    print("  2. Copy your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("  3. Run this command ONCE:  ngrok config add-authtoken YOUR_TOKEN")
    print("  4. Then run this script again: python deploy_ngrok.py")
    sys.exit(1)
