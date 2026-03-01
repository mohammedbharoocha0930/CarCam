"""
ROADWATCH — Python Backend Server
==================================
Live NYC traffic camera feed with:
  - License plate blurring (OpenCV)
  - Speed estimation (optical flow)
  - Vehicle detection (Google Gemini — FREE)
  - REST API for the frontend dashboard

Install dependencies:
    pip install flask flask-cors requests opencv-python numpy Pillow google-generativeai python-dotenv

Get a FREE Gemini API key (no credit card needed):
    1. Go to aistudio.google.com
    2. Sign in with Google
    3. Click Get API Key → Create API Key
    4. Paste it in your .env file as GEMINI_API_KEY=your-key-here

Run:
    python roadwatch_server.py

Then open roadwatch_dashboard.html in your browser.
"""

import os
import io
import time
import json
import base64
import threading
import requests
import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")   # free at aistudio.google.com
NYC_DOT_API     = "https://webcams.nyctmc.org/api/cameras"
CAM_REFRESH_SEC = 3      # how often to pull a new frame per active camera
SPEED_SAMPLE_SEC = 2     # optical flow interval
PORT            = 3001

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # fast + free tier
else:
    gemini_model = None

# ─────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow the HTML frontend to call us

# In-memory state
camera_list   = []          # all cameras from NYC DOT
frame_cache   = {}          # cam_id -> latest raw JPEG bytes
flow_cache    = {}          # cam_id -> (prev_gray, timestamp)
speed_cache   = {}          # cam_id -> estimated mph
detection_cache = {}        # cam_id -> latest AI detection result


# ─────────────────────────────────────────────────────────────
# 1.  LOAD CAMERA LIST FROM NYC DOT
# ─────────────────────────────────────────────────────────────
def load_cameras():
    """Fetch full camera list from NYC DOT (no API key needed)."""
    global camera_list
    print("Fetching camera list from NYC DOT...")
    try:
        res = requests.get(NYC_DOT_API, timeout=10)
        res.raise_for_status()
        data = res.json()
        # Keep only online cameras
        camera_list = [c for c in data if c.get("isOnline") in (True, "true")]
        print(f"  Loaded {len(camera_list)} online cameras.")
    except Exception as e:
        print(f"  ERROR loading cameras: {e}")
        camera_list = []


# ─────────────────────────────────────────────────────────────
# 2.  FETCH A SINGLE FRAME
# ─────────────────────────────────────────────────────────────
def fetch_frame(cam: dict) -> bytes | None:
    """Download the latest JPEG snapshot for a camera."""
    url = cam.get("imageUrl", "")
    if not url:
        return None
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200 and res.content:
            return res.content
    except Exception as e:
        print(f"  Frame fetch error ({cam['id'][:8]}): {e}")
    return None


# ─────────────────────────────────────────────────────────────
# 3.  LICENSE PLATE BLUR (OpenCV)
# ─────────────────────────────────────────────────────────────
def blur_license_plates(jpeg_bytes: bytes) -> bytes:
    """
    Blur the lower 35% of the frame where license plates typically appear.

    For production: replace the zone blur with a proper plate detector.
    Options:
      - OpenCV HAAR cascade for plates
      - Plate Recognizer API (platerecognizer.com) — has a blur endpoint
      - YOLOv8 fine-tuned on license plates (Roboflow has free pre-trained models)
    """
    # Decode JPEG → numpy array
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jpeg_bytes

    h, w = img.shape[:2]

    # ── Option A: Blanket zone blur (bottom 35%) ──────────────
    plate_zone_top = int(h * 0.62)
    region = img[plate_zone_top:h, 0:w]

    # Strong Gaussian blur on that band
    blurred_region = cv2.GaussianBlur(region, (51, 51), 20)
    img[plate_zone_top:h, 0:w] = blurred_region

    # ── Option B (advanced): OpenCV plate detection ───────────
    # Uncomment and supply a haarcascade XML for precise blurring:
    #
    # plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
    # for (x, y, pw, ph) in plates:
    #     img[y:y+ph, x:x+pw] = cv2.GaussianBlur(
    #         img[y:y+ph, x:x+pw], (51, 51), 20
    #     )

    # Draw privacy indicator line
    cv2.line(img, (0, plate_zone_top), (w, plate_zone_top), (0, 229, 255), 1)
    cv2.putText(
        img, "PLATE ZONE BLURRED",
        (w - 200, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (0, 229, 255), 1, cv2.LINE_AA
    )

    # Re-encode to JPEG
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────
# 4.  SPEED ESTIMATION (Optical Flow)
# ─────────────────────────────────────────────────────────────
def estimate_speed(cam_id: str, new_frame_bytes: bytes) -> float | None:
    """
    Compare current frame to the previous frame using Farneback optical flow.

    Calibration:
        pixel_to_meters = real_road_width_meters / road_width_in_pixels
        Typical NYC lane = 3.7m wide.  Measure it in the image once per camera.

    Returns estimated speed in mph, or None if not enough data.
    """
    PIXEL_TO_METERS = 0.05   # adjust per camera — 1 pixel ≈ 5 cm
    FPS_EQUIVALENT  = 1.0 / SPEED_SAMPLE_SEC  # frames per second

    nparr = np.frombuffer(new_frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if frame is None:
        return None

    now = time.time()
    prev_data = flow_cache.get(cam_id)

    if prev_data is None:
        flow_cache[cam_id] = (frame, now)
        return None

    prev_gray, prev_time = prev_data
    elapsed = now - prev_time

    if elapsed < SPEED_SAMPLE_SEC * 0.8:
        return None  # wait for next interval

    # Resize both to same size if needed
    if frame.shape != prev_gray.shape:
        frame = cv2.resize(frame, (prev_gray.shape[1], prev_gray.shape[0]))

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, frame,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Use 90th percentile to ignore background noise
    moving_pixels = magnitude[magnitude > 1.5]
    if len(moving_pixels) < 50:
        flow_cache[cam_id] = (frame, now)
        return 0.0  # no significant movement

    avg_pixel_displacement = float(np.percentile(moving_pixels, 90))

    # pixels/frame → meters/second → mph
    meters_per_second = avg_pixel_displacement * PIXEL_TO_METERS * FPS_EQUIVALENT
    mph = meters_per_second * 2.23694

    flow_cache[cam_id] = (frame, now)
    return round(min(mph, 120), 1)  # cap at 120mph


# ─────────────────────────────────────────────────────────────
# 5.  AI VEHICLE DETECTION (Claude Vision)
# ─────────────────────────────────────────────────────────────
def analyze_frame_with_gemini(jpeg_bytes: bytes) -> dict:
    """
    Send a camera frame to Google Gemini (FREE) and get back vehicle detections.
    Returns a dict with vehicles list, scene description, etc.

    Free tier: 15 requests/minute, 1 million tokens/day — more than enough.
    Get your free key at: aistudio.google.com
    """
    if not GEMINI_API_KEY or not gemini_model:
        return {"error": "No GEMINI_API_KEY set. Get a free key at aistudio.google.com"}

    try:
        # Convert bytes to PIL Image for Gemini
        img = Image.open(io.BytesIO(jpeg_bytes))

        prompt = (
            "You are analyzing a live NYC traffic camera frame. "
            "Return ONLY a JSON object with no markdown, no explanation:\n"
            "{\n"
            '  "vehicles": [\n'
            '    {\n'
            '      "type": "sedan|suv|truck|van|bus|motorcycle|taxi",\n'
            '      "make": "best guess or Unknown",\n'
            '      "model": "best guess or Unknown",\n'
            '      "color": "color name",\n'
            '      "direction": "approaching|departing|stopped"\n'
            "    }\n"
            "  ],\n"
            '  "scene": "one sentence traffic conditions summary",\n'
            '  "vehicle_count": <number>,\n'
            '  "congestion": "light|moderate|heavy"\n'
            "}"
        )

        response = gemini_model.generate_content([prompt, img])
        raw = response.text.strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# 6.  BACKGROUND FRAME POLLER
# ─────────────────────────────────────────────────────────────
active_cam_id = None   # the camera the frontend is currently watching

def frame_poller():
    """Background thread: continuously refresh the active camera's frame."""
    while True:
        cam_id = active_cam_id
        if cam_id:
            cam = next((c for c in camera_list if c["id"] == cam_id), None)
            if cam:
                raw = fetch_frame(cam)
                if raw:
                    # Estimate speed BEFORE blurring (optical flow on original)
                    speed = estimate_speed(cam_id, raw)
                    if speed is not None:
                        speed_cache[cam_id] = speed

                    # Blur plates
                    processed = blur_license_plates(raw)
                    frame_cache[cam_id] = processed

        time.sleep(CAM_REFRESH_SEC)


# ─────────────────────────────────────────────────────────────
# 7.  REST API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """Return list of all online cameras."""
    return jsonify(camera_list)


@app.route("/api/cameras/random", methods=["GET"])
def get_random_camera():
    """Return a random online camera."""
    import random
    if not camera_list:
        return jsonify({"error": "No cameras loaded"}), 503
    cam = random.choice(camera_list)
    return jsonify(cam)


@app.route("/api/cameras/<cam_id>/activate", methods=["POST"])
def activate_camera(cam_id):
    """Tell the server which camera the frontend is watching."""
    global active_cam_id
    active_cam_id = cam_id

    # Pre-fetch the first frame immediately
    cam = next((c for c in camera_list if c["id"] == cam_id), None)
    if cam:
        raw = fetch_frame(cam)
        if raw:
            processed = blur_license_plates(raw)
            frame_cache[cam_id] = processed
            return jsonify({"status": "ok", "cam": cam})

    return jsonify({"error": "Camera not found"}), 404


@app.route("/api/cameras/<cam_id>/frame", methods=["GET"])
def get_frame(cam_id):
    """
    Return the latest processed (plate-blurred) JPEG frame.
    The frontend polls this every 3 seconds to refresh the image.
    """
    frame = frame_cache.get(cam_id)

    if not frame:
        # Try to fetch on-demand
        cam = next((c for c in camera_list if c["id"] == cam_id), None)
        if cam:
            raw = fetch_frame(cam)
            if raw:
                frame = blur_license_plates(raw)
                frame_cache[cam_id] = frame

    if not frame:
        return Response("Frame unavailable", status=503)

    return Response(
        frame,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
    )


@app.route("/api/cameras/<cam_id>/speed", methods=["GET"])
def get_speed(cam_id):
    """Return the latest optical-flow speed estimate for a camera."""
    speed = speed_cache.get(cam_id)
    return jsonify({
        "cam_id": cam_id,
        "speed_mph": speed,
        "timestamp": time.time()
    })


@app.route("/api/cameras/<cam_id>/analyze", methods=["POST"])
def analyze_camera(cam_id):
    """
    Run Claude AI analysis on the current frame.
    Returns vehicle detections, scene description, congestion level.
    """
    frame = frame_cache.get(cam_id)
    if not frame:
        return jsonify({"error": "No frame available for this camera"}), 404

    # Check if we have a fresh detection (avoid hammering the API)
    last = detection_cache.get(cam_id)
    if last and (time.time() - last.get("timestamp", 0)) < 5:
        return jsonify(last)

    result = analyze_frame_with_gemini(frame)
    result["timestamp"] = time.time()
    result["cam_id"] = cam_id

    # Attach speed estimate
    result["speed_mph"] = speed_cache.get(cam_id)

    detection_cache[cam_id] = result
    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    """Health check — confirms server is running."""
    return jsonify({
        "status": "ok",
        "cameras_loaded": len(camera_list),
        "active_cam": active_cam_id,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "port": PORT
    })


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load cameras on startup
    load_cameras()

    # Start background frame poller thread
    poller = threading.Thread(target=frame_poller, daemon=True)
    poller.start()
    print(f"Frame poller started.")

    print(f"\n{'='*50}")
    print(f"  ROADWATCH Server running on http://localhost:{PORT}")
    print(f"  Cameras loaded: {len(camera_list)}")
    print(f"  Gemini key:     {'SET ✓' if GEMINI_API_KEY else 'NOT SET — get free key at aistudio.google.com'}")
    print(f"{'='*50}\n")
    print("  Endpoints:")
    print(f"    GET  /api/cameras               — list all cameras")
    print(f"    GET  /api/cameras/random         — get a random camera")
    print(f"    POST /api/cameras/<id>/activate  — set active camera")
    print(f"    GET  /api/cameras/<id>/frame     — get latest frame (JPEG)")
    print(f"    GET  /api/cameras/<id>/speed     — get speed estimate")
    print(f"    POST /api/cameras/<id>/analyze   — run Claude AI analysis")
    print(f"    GET  /api/status                 — health check\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
