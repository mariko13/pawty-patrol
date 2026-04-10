# Pawty Patrol

Pawty Patrol is a web app that detects when a dog is pooping in a video using a machine learning model.

## Features
- Upload a video
- Detect pooping events
- Display timestamps

## Tech Stack
- FastAPI
- PyTorch (MobileNetV2)
- OpenCV

## How to Run

pip install -r requirements.txt

uvicorn app.main:app --reload

## Model
The model was trained using a labeled dataset of dog pooping images with transfer learning.

## Author
Mariko