# 🚦 ROADWATCH

## Real-Time Traffic Intelligence Platform

Live NYC traffic cameras powered by computer vision and AI.

---

## Overview

ROADWATCH streams public NYC DOT traffic cameras and performs real-time analysis on each frame before it reaches the browser.

All image processing occurs server-side.

---

## Key Features

- Streams 900+ live NYC DOT traffic cameras
- Applies server-side license plate blurring using OpenCV
- Estimates vehicle speed using Farneback optical flow
- Converts pixel displacement into MPH
- Uses Google Gemini Vision for AI vehicle classification
- Displays a live 30-second rolling analytics dashboard
- Provides random camera selection across NYC

---

## AI Vehicle Classification

Gemini Vision extracts structured vehicle data including:

- Vehicle type
- Make
- Model
- Color
- Direction of travel

---

## Privacy & Security

- License plates are blurred before frames reach the browser
- No footage is stored
- Frames are processed entirely in memory
- API keys are stored in environment variables
- `.env` is excluded via `.gitignore`

---

## System Flow

Camera Feed → Flask Backend → OpenCV Processing → Gemini AI → REST API → Live Dashboard

---

## Tech Stack

- Python
- Flask
- OpenCV
- NumPy
- Pillow
- Google Gemini 2.0 Flash
- HTML
- CSS
- JavaScript

---

## Local Setup

### Clone the repository

```bash
git clone https://github.com/mohammedbharoocha0930/CarCam.git
cd CarCam

Install dependencies
pip install flask flask-cors requests opencv-python numpy Pillow google- generativeai python-dotenv

Create a .env file in the project root
GEMINI_API_KEY=your_key_here

Run the backend
python roadwatch_server.py

Serve the frontend
python -m http.server 8080

Open in browser:

http://localhost:8080/roadwatch_dashboard.html

```
# Author

Mohammed Bharoocha
B.A. Computer Science
