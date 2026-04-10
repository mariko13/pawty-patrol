import cv2

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    timestamps = []

    count = 0
    success, frame = cap.read()

    while success:
        if int(count % (fps * frame_rate)) == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(frame)
            timestamps.append(timestamp)

        success, frame = cap.read()
        count += 1

    cap.release()
    return frames, timestamps