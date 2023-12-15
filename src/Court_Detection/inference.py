import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch


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
        "--model_path",
        type=str,
        default="models/yolov8_court/best.pt",
        help="Path to save the processed file.",
    )

    return parser


def process_image(input_path, output_path, model):
    img = cv2.imread(input_path)
    image_height, image_width = img.shape[:2]

    input = cv2.resize(img, (640, 640))
    results = model.predict(input, conf=0.25, save=False)
    result = results[0]
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0
    court_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    court_masks = masks[court_indices[0]]

    # scale for visualizing results (optional)
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
    img = cv2.imread(input_path)
    cv2.drawContours(img, [trapezoid], 0, (0, 0, 0), 5)
    cv2.imwrite(output_path, img)


def process_video(input_path, output_path, model):
    # Load the video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Loop through each frame of the video
    for _ in tqdm(range(total_frames), desc="Processing video"):
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Process
        image = np.array(frame)
        image_copy = image.copy()
        image_copy = cv2.resize(image_copy, (640, 640))
        results = model(image_copy, verbose=False)

        result = results[0]
        # get array results
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        # get indices of results where class is 0
        court_indices = torch.where(clss == 0)
        # use these indices to extract the relevant masks
        court_masks = masks[court_indices[0]]

        # scale for visualizing results (optional)
        court_mask = torch.any(court_masks, dim=0).int() * 255
        mask_image = court_mask.cpu().numpy()
        # convert to CV_8UC1 image
        mask_image = mask_image.astype(np.uint8)
        # Resize the mask image to the original image size
        mask_image = cv2.resize(mask_image, (frame_size[0], frame_size[1]))

        # Find and Draw Contours
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        trapezoid = cv2.approxPolyDP(largest_contour, epsilon, True)

        cv2.drawContours(image, [trapezoid], 0, (0, 0, 0), 5)
        out.write(image)

    # Release the video capture and output video
    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    # Define and parse command-line arguments
    parser = parser_function()
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path

    # Verify Default input path
    if input_path is None:
        raise Exception("Please provide an input")

    # Determine if input is an image or video based on file extension
    file_extension = os.path.splitext(args.input_path)[1]

    if output_path is None:
        output_path = "outputs/Court_Detection/inference" + file_extension

    model = YOLO(model_path)
    if file_extension in [".jpg", ".png", ".jpeg"]:
        print("Processing image...")
        process_image(input_path, output_path, model)
    elif file_extension in [".mp4", ".avi"]:
        print("Processing video...")
        process_video(input_path, output_path, model)
    else:
        raise Exception(
            "Please provide a valid input format (.jpg, jpeg, .png, .mp4, .avi)"
        )

    print("Done!")
