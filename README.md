# README

## Repository Structure

_Some of the folders are empty because of the size limit of github (dataset folders especially)_

Hence the need to download them separately and then unzip them in the respective folders. (The folder structure is given below)
Also, some testing videos and images are present in their corresponding data folders but not visible in the structure below for simplicity.
```
.
├── assets
├── data
│   ├── Court_Detection
│   │   └── dataset
│   ├── Player_Actions
│   │   └── dataset
│   └── Tracking
│       └── dataset
├── docs
│   └── screenshots
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
    ├── Court_Detection
    │   ├── inference.py
    │   ├── requirements.txt
    │   └── train.py
    ├── Players_Actions
    │   └── inference.py
    └── Tracking
        ├── inference.py
        ├── requirements.txt
        └── train.py
```

## TODO 

- [ ] Players Action
  - [x] Local model
  - [x] GPU available
  - [x] Script Working
  - [ ] Train with diff perspective
  - [ ] Gradio WebApp

- [ ] Court Detection
  - [x] Local Model
  - [x] GPU available
  - [x] Script Working
  - [x] Change output format
  - [x] Train with diff perspective
  - [ ] Gradio WebApp

- [ ] Tracking
  - [x] Local Model
  - [x] Train YoloV8
    - [x] use ultralytics to train and then rewrite the script for tracking
  - [x] GPU available
  - [x] Script working
  - [ ] Train with diff perspective
  - [ ] Gradio WebApp
