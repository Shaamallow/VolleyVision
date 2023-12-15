# README

<div align="center">
  <h1>Volley Vision</h1>
  <img src="./assets/volleyball_logo.jpeg" width="350" title="Volleyball Logo">
</div>


## Teaser 

<div align="center">
<img src="assets/yolov8_tracking.gif" width="400">
<img src="assets/all_in_one.gif" width="400">
</div>

Tracking volleyball over video and images, detecting players actions and field recognition

## Installation

We advise to use a dedicated environnement (conda, mamba, whatever you like) to install the librairies.

Clone repo : 

```bash
git clone https://github.com/Shaamallow/VolleyVision.git
```

Install env (either conda or pip directly)

_I'm not to sure about the dependencies compatibility with other distro, it was tested with : 
**Ubuntu 22.04.3 LTS** & **CUDA 12.2**_ (*#itWorksOnMyMachine*). If running into troubles, please try to install the dependencies on the inferences.py files. 

```bash
conda env create -f environment.yml
```

OR

```bash
pip install -r requirements.txt
```

### Execute Scripts

Scripts must be executed from the root folder of the project, be careful about the paths

#### Tracking

```bash
python src/Tracking/inference.py --input_path [your_path] --output_path [your_path]
```

```bash
usage: inference.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--model MODEL] [--confidence CONFIDENCE] [--marker {circle,box}]
                    [--color {black,white,red,green,purple,blue,yellow,cyan,gray,navy}]

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        path to the video or image to process | Default : None
  --output_path OUTPUT_PATH
                        path for the output (.mp4 for videos and .jpg/.png for images) | Default : outputs/Tracking/inference.[jpg/mp4]
  --model MODEL         path to yolov8 model | Default : models/yolov8_tracking/best.pt
  --confidence CONFIDENCE
                        prediction confidence
  --marker {circle,box}
                        how to highlight the ball
  --color {black,white,red,green,purple,blue,yellow,cyan,gray,navy}
                        color for highlighting the ball
```

#### Court Detection

```bash
python src/Court_Detection/inference.py --input_path [your_path] --output_path [your_path]
```

```bash
usage: inference.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--model_path MODEL_PATH]

Process an image or a video.

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to the input video or image.
  --output_path OUTPUT_PATH
                        Path to save the processed file.
  --model_path MODEL_PATH
                        Path to save the processed file.
```

#### Player Action

```bash
python src/Players_Actions/inference.py --model_path models/yolov8_actions/weights/best.pt --input_path [your_path] --output_path [your_path] --show_labels 
```

```bash
usage: inference.py [-h] [--model_path MODEL_PATH] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--show_conf] [--show_labels] [--conf CONF] [--max_det MAX_DET] [--classes CLASSES [CLASSES ...]]
                    [--line_width LINE_WIDTH] [--font_size FONT_SIZE]

YOLOv8 Image/Video Processing

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to yolov8 model | Default : models/yolov8_players/weights/best.pt
  --input_path INPUT_PATH
                        path to the video or image to process | Default : None
  --output_path OUTPUT_PATH
                        path for the output (.mp4 for videos and .jpg/.png for images) | Default : outputs/Players_Actions/inference.[jpg/mp4]
  --show_conf           Whether to show the confidence scores
  --show_labels         Whether to show the labels
  --conf CONF           Object confidence threshold for detection
  --max_det MAX_DET     Maximum number of detections per image
  --classes CLASSES [CLASSES ...]
                        List of classes to detect
  --line_width LINE_WIDTH
                        Line width for bounding box visualization
  --font_size FONT_SIZE
                        Font size for label visualization
```


#### Player Segmentation

```bash
python src/Players_Actions/inference.py --model_path models/yolov8_players/weights/best.pt --input_path [your_path] --output_path [your_path] --show_labels 
```

```bash
usage: inference.py [-h] [--model_path MODEL_PATH] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--show_conf] [--show_labels] [--conf CONF] [--max_det MAX_DET] [--classes CLASSES [CLASSES ...]]
                    [--line_width LINE_WIDTH] [--font_size FONT_SIZE]

YOLOv8 Image/Video Processing

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to yolov8 model | Default : models/yolov8_players/weights/best.pt
  --input_path INPUT_PATH
                        path to the video or image to process | Default : None
  --output_path OUTPUT_PATH
                        path for the output (.mp4 for videos and .jpg/.png for images) | Default : outputs/Players_Actions/inference.[jpg/mp4]
  --show_conf           Whether to show the confidence scores
  --show_labels         Whether to show the labels
  --conf CONF           Object confidence threshold for detection
  --max_det MAX_DET     Maximum number of detections per image
  --classes CLASSES [CLASSES ...]
                        List of classes to detect
  --line_width LINE_WIDTH
                        Line width for bounding box visualization
  --font_size FONT_SIZE
                        Font size for label visualization
```

#### Combined

Be careful, combined model **NEEDS** to detect a field to be working properly, not enough time to fix that part unfortunately.

```bash
python src/Combined/inference.py --input_path [your_path] --output_path [your_path]
```

```bash
usage: inference.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--model_tracking MODEL_TRACKING] [--model_court MODEL_COURT] [--model_actions MODEL_ACTIONS] [--classes CLASSES [CLASSES ...]]

Process an image or a video.

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to the input video or image.
  --output_path OUTPUT_PATH
                        Path to save the processed file.
  --model_tracking MODEL_TRACKING
                        Path to the tracking model
  --model_court MODEL_COURT
                        Path to the court detection model
  --model_actions MODEL_ACTIONS
                        Path to the actions detection model
  --classes CLASSES [CLASSES ...]
                        List of classes to detect
```

## Repository Structure

_Some of the folders are empty because of the size limit of github (dataset folders especially)_

Hence the need to download them separately and then unzip them in the respective folders. (links further down below)
Also, some testing videos and images are present in their corresponding data folders but not visible in the structure below for simplicity.

```bash
.
├── assets
│   ├── all_in_one.gif
│   ├── volleyball_logo.jpeg
│   ├── volleyball_shelf.jpg
│   └── yolov8_tracking.gif
├── data
│   ├── Court_Detection
│   │   └── dataset
│   ├── Player_Actions
│   │   └── dataset
│   └── Tracking
│       └── dataset
├── docs
│   ├── all_in_one.mp4
│   ├── screenshots
│   └── tracking_playing.mp4
├── environment.yml
├── models
│   ├── yolov8
│   ├── yolov8_actions
│   ├── yolov8_court
│   ├── yolov8_players
│   └── yolov8_tracking
├── outputs
│   ├── Court_Detection
│   ├── Players_Actions
│   └── Tracking
├── README.md
├── requirements.txt
├── runs
│   ├── Court_Detection
│   └── Tracking
└── src
    ├── Combined
    ├── Court_Detection
    ├── Players_Actions
    └── Tracking
```

## Dataset

To retrain the model, the dataset need to be unzip in the correct folder `data/[Tracking/Court_Detection]/dataset/.`

Example : 
```bash
└── Tracking
    ├── back_view.mp4
    ├── dataset
    │   ├── data.yaml
    │   ├── train
    │   └── valid
```

- [Court_Field](https://stratus.binets.fr/s/KgKW37c9b79WK7i) dataset
- [Tracking](https://stratus.binets.fr/s/5Tr6NtbmayiB9PA) dataset

Using the Nextcloud infrastructure of the [Binet Réseau](https://br.binets.fr/)
