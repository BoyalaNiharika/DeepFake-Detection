import torch
import cv2
import numpy as np
import face_recognition

from torchvision import transforms

# --------------------------
# Device Setup
# --------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# --------------------------
# Transform
# --------------------------

im_size = 112

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --------------------------
# Extract Frames (FIXED)
# --------------------------

def extract_frames(video_path, seq_len):

    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise ValueError("Error opening video file")

    frames = []

    while True:

        success, frame = vid.read()

        if not success:
            break

        # Skip invalid frames
        if frame is None:
            continue

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert BGR → RGB
        if len(frame.shape) == 3:

            if frame.shape[2] == 3:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2RGB
                )

        # --------------------------
        # Face Detection (Optional)
        # --------------------------

        try:

            faces = face_recognition.face_locations(frame)

            if len(faces) > 0:

                top, right, bottom, left = faces[0]

                h, w = frame.shape[:2]

                top = max(0, top)
                left = max(0, left)
                bottom = min(h, bottom)
                right = min(w, right)

                face = frame[top:bottom, left:right]

                if face.size != 0:
                    frame = face

        except Exception:
            pass   # Keep full frame

        # Append frame always
        frames.append(frame)

        if len(frames) == seq_len:
            break

    vid.release()

    # --------------------------
    # Handle Short Videos
    # --------------------------

    if len(frames) == 0:

        raise ValueError(
            "No frames extracted from video."
        )

    # Pad if fewer frames
    if len(frames) < seq_len:

        last_frame = frames[-1]

        while len(frames) < seq_len:

            frames.append(last_frame)

    print("Frames extracted:", len(frames))

    return frames[:seq_len]


# --------------------------
# Predict Video (FIXED)
# --------------------------

def predict_video(model, video_path, seq_len):

    frames = extract_frames(
        video_path,
        seq_len
    )

    processed_frames = []

    for frame in frames:

        processed_frames.append(
            transform(frame)
        )

    # Create tensor once
    frames_tensor = torch.stack(
        processed_frames
    ).unsqueeze(0).to(device)

    model = model.to(device)

    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        outputs = model(frames_tensor)

        probs = sm(outputs)

        _, pred = torch.max(
            probs,
            1
        )

        confidence = (
            probs[0][pred.item()].item()
            * 100
        )

    return pred.item(), confidence