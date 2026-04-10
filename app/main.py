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

model = PoopModel("model/poop_model8.pth")

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

    # detections = []
    #
    # for frame, ts in zip(frames, timestamps):
    #     pred, conf = model.predict(frame)
    #
    #     if pred == 0 and conf > 0.8:
    #         detections.append(round(ts, 2))
    #
    # return templates.TemplateResponse("index.html", {
    #     "request": request,
    #     "results": detections
    # })

    detections = []
    saved_images = []

    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        pred, conf = model.predict(frame)

        if pred == 0 and conf > 0.8:
            timestamp = round(ts, 2)
            # detections.append(timestamp)
            detections.append(f"{timestamp}s (confidence: {round(conf, 2)})")

            image_path = f"app/static/detection_{i}.jpg"
            import cv2
            cv2.imwrite(image_path, frame)

            saved_images.append(f"/static/detection_{i}.jpg")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": detections,
        "images": saved_images
    })
