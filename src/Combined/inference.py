import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
from ultralytics import YOLO
import torch


### SCRIPT PARAMETERS ###

CONF = 0.3
MAX_DET = 300


def parser_function():
    parser = argparse.ArgumentParser(description="Process an image or a video.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to the input video or image.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the processed file.",
    )

    parser.add_argument(
        "--model_tracking",
        type=str,
        default="models/yolov8_tracking/best.pt",
        help="Path to the tracking model",
    )

    parser.add_argument(
        "--model_court",
        type=str,
        default="models/yolov8_court/best.pt",
        help="Path to the court detection model",
    )

    parser.add_argument(
        "--model_actions",
        type=str,
        default="models/yolov8_actions/weights/best.pt",
        help="Path to the actions detection model",
    )

    parser.add_argument(
        "--classes", nargs="+", default=None, help="List of classes to detect"
    )
    return parser


def process_image(
    input_path, output_path, model_tracking, model_court, model_actions, classes
):
    image = cv2.imread(input_path)
    image_height, image_width = image.shape[:2]

    image_resized = cv2.resize(image, (640, 640))

    results_tracking = model_tracking(image, conf=CONF, save=False)
    results_court = model_court(image_resized, conf=CONF, save=False)
    results_actions = model_actions(
        image,
        conf=CONF,
        classes=classes,
        save=False,
    )

    # Add ball tracking results

    annotated = results_tracking[0].plot(
        labels=True,
        conf=False,
        line_width=3,
        img=image,
    )

    # Add court detection results

    result_court = results_court[0]

    masks = result_court.masks.data
    boxes = result_court.boxes.data

    clss = boxes[:, 5]
    court_indices = torch.where(clss == 0)
    court_masks = masks[court_indices[0]]

    court_mask = torch.any(court_masks, dim=0).int() * 255
    mask_image = court_mask.cpu().numpy()
    # convert to CV_8UC1 image
    mask_image = mask_image.astype(np.uint8)
    # Resize the mask image to the original image size
    mask_image = cv2.resize(mask_image, (image_width, image_height))

    # Find and Draw Contours
    contours, _ = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    trapezoid = cv2.approxPolyDP(largest_contour, epsilon, True)

    cv2.drawContours(annotated, [trapezoid], 0, (0, 0, 0), 5)

    # Add actions detection results

    annotated = results_actions[0].plot(
        labels=True,
        conf=False,
        line_width=3,
        img=annotated,
    )

    cv2.imwrite(output_path, annotated)


def process_video(
    video_path,
    output_path,
    model_tracking,
    model_court,
    model_actions,
    classes,
):
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
        image_height, image_width = image.shape[:2]
        image_resized = cv2.resize(image, (640, 640))
        results_tracking = model_tracking(image, conf=CONF, save=False, verbose=False)
        results_court = model_court(image_resized, conf=CONF, save=False, verbose=False)
        results_actions = model_actions(
            image,
            conf=CONF,
            classes=classes,
            save=False,
            verbose=False,
        )

        # Add ball tracking results

        boxe = results_tracking[0].boxes.xywh.cpu().numpy()
        ball_trajectory.pop()

        if boxe.size != 0:
            x, y, w, h = boxe[0]
            ball_trajectory.appendleft((x, y, w, h))
        else:
            ball_trajectory.appendleft(None)

        for i in range(7):
            if ball_trajectory[i] is not None:
                x, y, w, h = ball_trajectory[i]

                # Reduce the size of the box by 10%*i
                x += 0.05 * i * w
                y += 0.05 * i * h
                w -= 0.05 * i * w
                h -= 0.05 * i * h

                # Create circle
                cv2.circle(
                    image,
                    (int(x), int(y)),
                    int(w / 2),
                    (0, 255, 255),
                    3,
                )

        # Add court detection results
        result_court = results_court[0]
        masks = result_court.masks.data
        boxes = result_court.boxes.data
        clss = boxes[:, 5]
        court_indices = torch.where(clss == 0)
        court_masks = masks[court_indices[0]]
        court_mask = torch.any(court_masks, dim=0).int() * 255
        mask_image = court_mask.cpu().numpy()
        # convert to CV_8UC1 image
        mask_image = mask_image.astype(np.uint8)
        # Resize the mask image to the original image size
        mask_image = cv2.resize(mask_image, (image_width, image_height))
        # Find and Draw Contours
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        trapezoid = cv2.approxPolyDP(largest_contour, epsilon, True)
        cv2.drawContours(image, [trapezoid], 0, (0, 0, 0), 5)

        # Add actions detection results

        annotated = results_actions[0].plot(
            labels=True,
            conf=False,
            line_width=3,
            img=image,
        )

        out.write(annotated)

    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    parser = parser_function()
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_tracking = YOLO(args.model_tracking)
    model_court = YOLO(args.model_court)
    model_actions = YOLO(args.model_actions)
    classes = args.classes

    if input_path is None:
        raise Exception("Please provide an input")

    file_extension = os.path.splitext(args.input_path)[1]

    if output_path is None:
        output_path = "outputs/test/test" + file_extension

    if file_extension in (".jpg", ".png"):
        print("Processing image")
        process_image(
            input_path,
            output_path,
            model_tracking,
            model_court,
            model_actions,
            classes,
        )
    elif file_extension in (".mp4", ".avi"):
        print("Processing video")
        process_video(
            input_path,
            output_path,
            model_tracking,
            model_court,
            model_actions,
            classes,
        )
    else:
        raise Exception("Invalid input file type")

    print(f'Processed file saved at "{output_path}"')
