import argparse
from collections import deque

import numpy as np
import os

from tqdm import tqdm

import cv2
from ultralytics import YOLO

# SCRIPT PARAMETERS

# Testing parameters
WEIGHTS_PATH = "src/Tracking/best.pt"
IMAGE_PATH = "outputs/Tracking/annotated2.jpg"
VIDEO_PATH = "data/Tracking/rally_men.mp4"
VIDEO_OUPUT_PATH = "outputs/Tracking/test.mp4"
MODEL_PATH = "models/yolov8_tracking/best.pt"

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
        default=None,
        help="path to the video or image to process | Default : None",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="path for the output (.mp4 for videos and .jpg/.png for images) | Default : outputs/Tracking/inference.[jpg/mp4]",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8_tracking/best.pt",
        help="path to yolov8 model | Default : models/yolov8_tracking/bestDpt",
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

    return parser


###################


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
def process_video(video_path, output_path, model, conf, marker, color):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    ball_trajectory = deque(maxlen=7)
    for i in range(7):
        ball_trajectory.appendleft(None)
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        # Check if frame is empty
        if not ret:
            break
        image = np.array(frame)
        result = model(image, conf=conf, verbose=False)
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


if __name__ == "__main__":
    # Create parser
    parser = parser_function()

    # Get Inputs

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model
    conf = args.confidence
    marker = args.marker
    color = color_dict[args.color]

    # Verify Default input path
    if input_path is None:
        raise Exception("Please provide an input")

    # Determine if input is an image or video based on file extension
    file_extension = os.path.splitext(args.input_path)[1]

    if output_path is None:
        output_path = "outputs/Tracking/inference" + file_extension

    if file_extension in [".jpg", ".png", ".jpeg"]:
        model = YOLO(model_path)
        process_image(input_path, output_path, model, conf)
    elif file_extension in [".mp4", ".avi"]:
        model = YOLO(model_path)
        print("Processing video")
        process_video(input_path, output_path, model, conf, marker, color)
    else:
        raise Exception(
            "Please provide a valid input format (.jpg, jpeg, .png, .mp4, .avi)"
        )

    print("Done")
