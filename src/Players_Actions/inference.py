import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import YOLO


# Parse command-line arguments
#
def parser_function():
    parser = argparse.ArgumentParser(description="YOLOv8 Image/Video Processing")
    parser.add_argument(
        "--model_path",
        default="models/yolov8_players/weights/best.pt",
        help="path to yolov8 model | Default : models/yolov8_players/weights/best.pt",
    )
    parser.add_argument(
        "--input_path",
        default=None,
        help="path to the video or image to process | Default : None",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="path for the output (.mp4 for videos and .jpg/.png for images) | Default : outputs/Players_Actions/inference.[jpg/mp4]",
    )
    parser.add_argument(
        "--show_conf",
        default=False,
        action="store_true",
        help="Whether to show the confidence scores",
    )
    parser.add_argument(
        "--show_labels",
        default=False,
        action="store_true",
        help="Whether to show the labels",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Object confidence threshold for detection",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=300,
        help="Maximum number of detections per image",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None, help="List of classes to detect"
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=3,
        help="Line width for bounding box visualization",
    )
    parser.add_argument(
        "--font_size", type=float, default=3, help="Font size for label visualization"
    )

    return parser


def process_image(
    image_path,
    output_path,
    model,
    conf,
    max_det,
    classes,
    line_width,
    font_size,
    show_conf,
    show_labels,
    verbose=True,
):
    # Load and preprocess the image
    img = cv2.imread(image_path)

    # Perform prediction
    results = model(
        img,
        conf=conf,
        max_det=max_det,
        classes=classes,
        verbose=verbose,
    )

    # Annotate the image with bounding boxes
    annotated = results[0].plot(
        conf=show_conf,
        labels=show_labels,
        line_width=line_width,
        font_size=font_size,
    )

    # Save the annotated image
    cv2.imwrite(output_path, annotated)


def process_video(
    video_path,
    output_path,
    model,
    conf,
    max_det,
    classes,
    line_width,
    font_size,
    show_conf,
    show_labels,
    verbose=False,
):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Loop through the video frames
    for _ in tqdm(range(total_frames), desc="Processing video"):
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        image = np.array(frame)
        # Run YOLOv8 inference on the frame
        results = model(
            image,
            conf=conf,
            max_det=max_det,
            classes=classes,
            verbose=verbose,
        )

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot(
            conf=show_conf,
            labels=show_labels,
            line_width=line_width,
            font_size=font_size,
        )

        # Write the annotated frame to the output video
        out.write(annotated_frame)


if __name__ == "__main__":
    parser = parser_function()
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    conf = args.conf
    max_det = args.max_det
    classes = args.classes
    line_width = args.line_width
    font_size = args.font_size
    show_conf = args.show_conf
    show_labels = args.show_labels

    # Verify Default input path
    if input_path is None:
        raise Exception("Please provide an input")

    # Determine if input is an image or video based on file extension
    file_extension = os.path.splitext(args.input_path)[1]

    if output_path is None:
        output_path = "outputs/Players_Actions/inference" + file_extension
    model = YOLO(model_path)

    if file_extension in (".jpg", ".png"):
        print("Processing image")
        process_image(
            input_path,
            output_path,
            model,
            conf,
            max_det,
            classes,
            line_width,
            font_size,
            show_conf,
            show_labels,
        )

    elif file_extension in (".mp4", ".avi", ".mkv", ".mov", ".mpv"):
        print("Processing video")
        process_video(
            input_path,
            output_path,
            model,
            conf,
            max_det,
            classes,
            line_width,
            font_size,
            show_conf,
            show_labels,
        )
    else:
        raise Exception("Invalid input file type")

    print("Done!")
