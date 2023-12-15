import argparse
from collections import deque

import numpy as np

import cv2
from ultralytics import YOLO

# SCRIPT PARAMETERS

# Testing parameters
WEIGHTS_PATH = "src/Tracking/best.pt"
IMAGE_PATH = "outputs/Tracking/annotated2.jpg"
VIDEO_PATH = "data/Tracking/rally_men.mp4"
VIDEO_OUPUT_PATH = "outputs/Tracking/test.mp4"

# Use openCV standars for color : BGR
color_dict = {
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "red": [0, 0, 255],
    "green": [0, 255, 0],
    "purple": [128, 0, 128],
    "blue": [255, 0, 0],
    "yellow": [0, 255, 255],
    "cyan": [255, 255, 0],
    "gray": [128, 128, 128],
    "navy": [128, 0, 0],
    "pink": [147, 20, 255],
    "orange": [0, 69, 255],
}


### Parse Arguments ###


def parser_function():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/Tracking/small/gilb.jpg",
        help="path to the video or image with volleyball in it",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/Tracking/test.jpg",
        help="path for the output (.mp4 for videos and .jpg/.png for images), default new Output folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="src/Tracking/best.pt",
        help="which model to use for prediction",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.2, help="prediction confidence"
    )

    parser.add_argument(
        "--marker",
        type=str,
        default="circle",
        choices=["circle", "box"],
        help="how to highlight the ball",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="yellow",
        choices=[
            "black",
            "white",
            "red",
            "green",
            "purple",
            "blue",
            "yellow",
            "cyan",
            "gray",
            "navy",
        ],
        help="color for highlighting the ball",
    )
    parser.add_argument(
        "--no_trace", action="store_true", help="don't draw trajectory of the ball"
    )

    return parser


# Create parser
parser = parser_function()

# Get Inputs

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path
model_path = args.model
conf = args.confidence
marker = args.marker
no_trace = args.no_trace
color = color_dict[args.color]

###################

### Load Model ###

model = YOLO(model_path)


def process_image(image_path, output_path, model, conf):
    image = cv2.imread(image_path)
    result = model(image)

    annotated = result[0].plot(
        conf=conf,
        labels=["ball"],
        line_width=3,
    )

    cv2.imwrite(output_path, annotated)

    return result


# Now same process but for video
#
def process_video(video_path, output_path, model, conf, marker, color):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    ball_trajectory = deque(maxlen=7)
    for i in range(7):
        ball_trajectory.appendleft(None)
    while cap.isOpened():
        ret, frame = cap.read()
        # Check if frame is empty
        if not ret:
            break
        image = np.array(frame)
        result = model(image)
        boxe = result[0].boxes.xywh.cpu().numpy()
        ball_trajectory.pop()
        if boxe.size != 0:
            x, y, w, h = boxe[0]
            ball_trajectory.appendleft((x, y, w, h))
        else:
            ball_trajectory.appendleft(None)
        for i in range(7):
            if ball_trajectory[i] is not None:
                if marker == "box":
                    x, y, w, h = ball_trajectory[i]
                    cv2.rectangle(
                        image,
                        (int(x), int(y)),
                        (int(x + w), int(y + h)),
                        color,
                        3,
                    )
                elif marker == "circle":
                    x, y, w, h = ball_trajectory[i]
                    cv2.circle(
                        image,
                        (int(x + w / 2), int(y + h / 2)),
                        10,
                        color,
                        3,
                    )
        out.write(image)
    cap.release()
    cv2.destroyAllWindows()
    out.release()


process_video(VIDEO_PATH, VIDEO_OUPUT_PATH, model, conf, marker, color)

print("Done")
