import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)

ASCII_CHARS = (
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/"
    "\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
)

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

# =========================
# ENHANCEMENT
# =========================
def enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


# =========================
# AUTO FACE CROP
# =========================
def crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return frame

    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    pad = int(0.3*w)

    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(frame.shape[1], x+w+pad)
    y2 = min(frame.shape[0], y+h+pad)

    return frame[y1:y2, x1:x2]


# =========================
# FRAME → ASCII (FAST)
# =========================
def frame_to_ascii(frame, width=180):
    frame = crop_face(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance(gray)

    h, w = gray.shape
    aspect_ratio = h / w
    new_h = int(width * aspect_ratio * 0.55)

    small = cv2.resize(gray, (width, new_h))

    ascii_img = []
    for row in small:
        line = "".join(
            ASCII_CHARS[int(pixel/255*(len(ASCII_CHARS)-1))]
            for pixel in row
        )
        ascii_img.append(line)

    return "\n".join(ascii_img)


# =========================
# LIVE STREAM
# =========================
def generate_ascii():
    while True:
        success, frame = camera.read()
        if not success:
            break

        ascii_frame = frame_to_ascii(frame, width=180)
        yield f"data: {ascii_frame}\n\n"


# =========================
# 4K IMAGE GENERATOR
# =========================
def image_to_ascii_4k(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = enhance(gray)

    width = 320  # 🔥 4K MODE
    h, w = gray.shape
    new_h = int(width * (h/w) * 0.55)
    small = cv2.resize(gray, (width, new_h))

    ascii_lines = []
    for row in small:
        line = "".join(
            ASCII_CHARS[int(pixel/255*(len(ASCII_CHARS)-1))]
            for pixel in row
        )
        ascii_lines.append(line)

    font = ImageFont.load_default()

    dummy = Image.new("RGB", (1,1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0,0),"A",font=font)
    cw = bbox[2]-bbox[0]
    ch = bbox[3]-bbox[1]

    out = Image.new("RGB",(cw*width, ch*len(ascii_lines)),"black")
    draw = ImageDraw.Draw(out)

    y=0
    for line in ascii_lines:
        draw.text((0,y),line,fill=(0,255,0),font=font)
        y+=ch

    out_path = "ascii_4k_output.png"
    out.save(out_path)
    return out_path


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_ascii(),
                    mimetype="text/event-stream")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    path = "upload.jpg"
    file.save(path)

    out_path = image_to_ascii_4k(path)
    return send_file(out_path, as_attachment=True)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)