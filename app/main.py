from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os

from app.model import PoopModel
from app.utils import extract_frames

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

model = PoopModel("model/poop_model.pth")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames, timestamps = extract_frames(file_path)

    detections = []
    saved_images = []

    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        if frame.mean() < 10:
            continue

        pred, conf = model.predict(frame)

        if conf < 0.6:
            continue

        if pred == 1 and conf > 0.7:
            timestamp = round(ts, 2)
            detections.append(f"{timestamp}s (confidence: {round(conf, 2)})")

            os.makedirs("app/static/detections", exist_ok=True)
            image_path = f"app/static/detections/detection_{i}.jpg"
            import cv2
            cv2.imwrite(image_path, frame)

            saved_images.append(f"/static/detections/detection_{i}.jpg")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": detections,
        "images": saved_images
    })
