#!/usr/bin/env python3
"""
Web Dashboard for Bike Monitor
Provides live video feed and system status via Flask web interface
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import os

app = Flask(__name__)

# Global variables (thread-safe with lock)
# Multi-camera support: camera_id -> frame
frame_buffers = {}
frame_lock = threading.Lock()

# Current selected camera for display
current_camera = 'camera1'
camera_lock = threading.Lock()

# Statistics (updated by main script)
# Per-camera stats
camera_stats = {}
stats_lock = threading.Lock()

# Global stats
global_stats = {
    'cpu_temp': 0.0,
    'fps': 0.0,
    'uptime': 0,
    'status': 'Starting...'
}
global_stats_lock = threading.Lock()


def get_cpu_temperature():
    """Get Raspberry Pi CPU temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
            return round(temp, 1)
    except:
        return 0.0


def update_frame(camera_id, frame):
    """Update the frame buffer for a specific camera (called from main script)"""
    global frame_buffers
    with frame_lock:
        frame_buffers[camera_id] = frame.copy()


def update_camera_stats(camera_id, active_tracking=None, alerts_sent=None):
    """Update statistics for a specific camera"""
    global camera_stats
    with stats_lock:
        if camera_id not in camera_stats:
            camera_stats[camera_id] = {'active_tracking': 0, 'alerts_sent': 0}

        if active_tracking is not None:
            camera_stats[camera_id]['active_tracking'] = active_tracking
        if alerts_sent is not None:
            camera_stats[camera_id]['alerts_sent'] = alerts_sent


def update_global_stats(fps=None, uptime=None, status=None):
    """Update global system statistics"""
    global global_stats
    with global_stats_lock:
        if fps is not None:
            global_stats['fps'] = round(fps, 1)
        if uptime is not None:
            global_stats['uptime'] = uptime
        if status is not None:
            global_stats['status'] = status
        global_stats['cpu_temp'] = get_cpu_temperature()


# Backward compatibility with single-camera code
def update_stats(active_tracking=None, alerts_sent=None, fps=None, uptime=None, status=None):
    """Legacy function for single camera (backward compatible)"""
    update_camera_stats('camera1', active_tracking, alerts_sent)
    update_global_stats(fps, uptime, status)


def generate_video_stream(camera_id='camera1'):
    """Generator for MJPEG video stream for specific camera"""
    import numpy as np

    while True:
        with frame_lock:
            if camera_id not in frame_buffers or frame_buffers[camera_id] is None:
                # Create blank frame with message
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for {camera_id}...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = jpeg.tobytes()
                else:
                    time.sleep(0.1)
                    continue
            else:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame_buffers[camera_id], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    time.sleep(0.1)
                    continue
                frame_bytes = jpeg.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS max


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """JSON status endpoint"""
    with stats_lock, global_stats_lock:
        # Calculate totals across all cameras
        total_tracking = sum(cam.get('active_tracking', 0) for cam in camera_stats.values())
        total_alerts = sum(cam.get('alerts_sent', 0) for cam in camera_stats.values())

        return jsonify({
            'active_tracking': total_tracking,
            'alerts_sent': total_alerts,
            'cpu_temp': global_stats['cpu_temp'],
            'fps': global_stats['fps'],
            'uptime': global_stats['uptime'],
            'status': global_stats['status'],
            'cameras': camera_stats
        })


if __name__ == '__main__':
    # For testing only
    print("Web Dashboard Test Mode")
    print("Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
