🚦 ROADWATCH — Live Traffic Intelligence Dashboard
A real-time traffic camera dashboard that pulls live feeds from NYC's public camera network, blurs license plates for privacy, estimates vehicle speeds using optical flow analysis, and uses Google Gemini AI to identify vehicle make, model, color, and direction — all displayed on a sleek live dashboard.

📸 What It Does
Every 3 seconds, the server fetches a live JPEG snapshot from a real NYC DOT traffic camera. Before the image ever reaches your browser, OpenCV automatically blurs the lower portion of the frame where license plates appear. The frame is then analyzed by Google Gemini which returns structured data about every vehicle in the shot — type, make, model, color, and direction of travel. Simultaneously, optical flow analysis compares consecutive frames to estimate how fast vehicles are moving in MPH. All of this feeds into a live dashboard that tracks a 30-second rolling window of vehicle activity.

✨ Features

🎥 900+ Live Cameras — Real NYC DOT traffic cameras across all 5 boroughs, updating every 2-3 seconds, completely free with no API key required
🔒 License Plate Privacy — Every frame is processed server-side with OpenCV Gaussian blur on the plate zone before being sent to the browser
📏 Speed Estimation — Farneback optical flow compares consecutive frames and converts pixel displacement to MPH
🤖 AI Vehicle Detection — Google Gemini Vision identifies vehicle type, make, model, color, and direction of travel
📊 30-Second Rolling Stats — Tracks total vehicles, over-limit count, and average speed in a live rolling window
🗺️ Random Camera Button — Instantly jump to any of 900+ cameras across NYC
🖥️ Live Dashboard — Real-time feed with HUD overlays, speed display, speed limit sign, and vehicle log


🗂️ Project Structure
CarCam/
├── roadwatch_server.py       # Python Flask backend
├── roadwatch_dashboard.html  # Frontend dashboard (HTML/CSS/JS)
├── .env                      # Your API keys (never pushed to GitHub)
├── .gitignore                # Protects .env and other sensitive files
└── README.md                 # This file

🛠️ Tech Stack
LayerTechnologyBackendPython, Flask, Flask-CORSComputer VisionOpenCV (plate blur + optical flow)AI DetectionGoogle Gemini 2.0 Flash (free tier)Camera FeedNYC DOT Public API (no key needed)FrontendHTML, CSS, JavaScriptImage ProcessingPillow, NumPy

⚙️ Setup & Installation
1. Clone the repository
bashgit clone https://github.com/mohammedbharoocha0930/CarCam.git
cd CarCam
2. Install dependencies
bashpip install flask flask-cors requests opencv-python numpy Pillow google-generativeai python-dotenv
```

### 3. Get a FREE Gemini API key
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API Key** → **Create API Key**
4. Copy the key — it starts with `AIza...`
5. No credit card required — completely free

### 4. Create your `.env` file
Create a file called `.env` in the `CarCam/` folder and add:
```
GEMINI_API_KEY=AIzaYourKeyHere

⚠️ Never share this file. It is already blocked from GitHub by .gitignore

5. Run the backend server
bashpython roadwatch_server.py
```
Wait until you see:
```
ROADWATCH Server running on http://localhost:3001
Cameras loaded: 949
Gemini key: SET ✓
6. Serve the frontend
Open a second terminal and run:
bashpython -m http.server 8080
```

### 7. Open the dashboard
Go to your browser and visit:
```
http://localhost:8080/roadwatch_dashboard.html
The green dot in the top left confirms the server is connected. Hit RANDOM CAM to load a live feed.

🔌 API Endpoints
MethodEndpointDescriptionGET/api/statusServer health checkGET/api/camerasList all 900+ online camerasGET/api/cameras/randomGet a random online cameraPOST/api/cameras/<id>/activateSet the active camera for pollingGET/api/cameras/<id>/frameGet latest plate-blurred JPEG frameGET/api/cameras/<id>/speedGet optical flow speed estimate in MPHPOST/api/cameras/<id>/analyzeRun Gemini AI analysis on current frame

📡 Camera Sources
SourceCoverageAPI KeyNYC DOTNew York City (900+ cams)❌ Not required511 NYNew York State✅ Free registrationCA DOTCalifornia❌ Not requiredFL 511Florida✅ Free registrationTX DOTTexas✅ Free registration

🚀 How Speed Detection Works

The server fetches a new frame every 3 seconds
Each frame is converted to grayscale
OpenCV Farneback optical flow compares current frame to previous
Magnitude of pixel movement is calculated across the frame
90th percentile of displacement is used to ignore background noise
Pixel displacement is converted to MPH using a pixel-to-meter ratio
Result is served via /api/cameras/<id>/speed


🔒 Privacy

License plates are blurred server-side before frames reach the browser
No footage is stored or recorded — frames are processed in memory only
No personal data is collected


📄 License
MIT — free to use, modify, and distribute.

👤 Author
Built by Mohammed Bharoocha
