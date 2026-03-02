🚦 ROADWATCH — Real-Time Traffic Intelligence Dashboard

A real-time traffic analytics platform that streams live NYC DOT traffic cameras, applies privacy-preserving computer vision, estimates vehicle speed using optical flow, and leverages Google Gemini AI to classify vehicle attributes — all displayed in a live interactive dashboard.

📸 Overview

ROADWATCH continuously:

Pulls live JPEG snapshots from NYC DOT public cameras (900+ across all 5 boroughs)

Applies server-side license plate blurring using OpenCV

Estimates vehicle speed using Farneback optical flow

Uses Google Gemini Vision to identify:

Vehicle type

Make & model

Color

Direction of travel

Displays a live 30-second rolling analytics dashboard

All processing occurs server-side before images are sent to the browser.

✨ Key Features

🎥 900+ Live NYC Cameras (no API key required)

🔒 Privacy-First Processing — license plate region blurred server-side

📏 Speed Estimation — Optical flow → pixel displacement → MPH conversion

🤖 AI Vehicle Classification — Gemini Vision structured analysis

📊 30-Second Rolling Analytics Window

🗺️ Random Camera Selection

🖥️ Live HUD Dashboard UI

🗂️ Project Structure
CarCam/
├── roadwatch_server.py        # Flask backend server
├── roadwatch_dashboard.html   # Frontend dashboard
├── .env                       # API keys (never committed)
├── .gitignore                 # Prevents secret leakage
└── README.md
🛠️ Tech Stack
Layer	Technology
Backend	Python, Flask, Flask-CORS
Computer Vision	OpenCV (Gaussian blur, Farneback optical flow)
AI Detection	Google Gemini 2.0 Flash
Image Processing	Pillow, NumPy
Camera Feed	NYC DOT Public Traffic API
Frontend	HTML, CSS, JavaScript
⚙️ Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/mohammedbharoocha0930/CarCam.git
cd CarCam
2️⃣ Install Dependencies
pip install flask flask-cors requests opencv-python numpy Pillow google-generativeai python-dotenv
🔑 Get a Free Gemini API Key

Visit https://aistudio.google.com

Click Get API Key

Create a new key

Copy the key (starts with AIza...)

No credit card required (free tier available).

🔒 Create Your .env File

Inside the CarCam/ folder:

GEMINI_API_KEY=your_key_here

⚠️ .env is excluded from Git via .gitignore.
Never commit your API key.

▶️ Run the Backend
python roadwatch_server.py

Expected output:

ROADWATCH Server running on http://localhost:3001
Cameras loaded: 900+
Gemini key: SET ✓
🌐 Serve the Frontend

In a second terminal:

python -m http.server 8080

Then open:

http://localhost:8080/roadwatch_dashboard.html

Green status indicator = server connected.

🔌 API Endpoints
Method	Endpoint	Description
GET	/api/status	Server health check
GET	/api/cameras	List available cameras
GET	/api/cameras/random	Random camera selection
POST	/api/cameras/<id>/activate	Set active camera
GET	/api/cameras/<id>/frame	Latest blurred frame
GET	/api/cameras/<id>/speed	Speed estimate (MPH)
POST	/api/cameras/<id>/analyze	Run Gemini AI analysis
🚀 How Speed Detection Works

Fetch frame every 3 seconds

Convert to grayscale

Apply Farneback optical flow

Compute pixel displacement magnitude

Use 90th percentile to reduce background noise

Convert displacement → MPH via calibrated pixel-to-meter ratio

Return speed via API endpoint

🔒 Privacy & Security

License plates blurred before leaving server

No frame storage (in-memory processing only)

No personal data collection

API keys secured via environment variables

.env excluded from Git tracking

📡 Camera Sources
Source	Coverage	API Key
NYC DOT	New York City (900+)	❌ Not required
511 NY	New York State	✅ Free
CA DOT	California	❌ Not required
FL 511	Florida	✅ Free
TX DOT	Texas	✅ Free
📄 License

MIT License — free to use and modify.

👤 Author

Mohammed Bharoocha
B.A. Computer Science — Florida International University
Software Engineering | Backend Systems | Computer Vision | AI Integration
