# README

```
.
├── assets
│   ├── 3d_activity.png
│   ├── action_Cut.jpg
│   ├── actions_collage.png
│   ├── actions.gif
│   ├── actions_screen.jpg
│   ├── app_back.png
│   ├── app_main.png
│   ├── header.png
│   ├── in_progress.jpg
│   ├── players_collage.png
│   ├── players.gif
│   ├── players_men.gif
│   ├── players_red.gif
│   ├── players_screen.jpg
│   ├── rf_backview.gif
│   ├── rf_men_rally.gif
│   ├── rf_women_rally.gif
│   ├── roboflow_logo.png
│   ├── sliding_window.gif
│   ├── track_men.gif
│   ├── volley-collage.jpg
│   ├── vv_logo.png
│   └── y7Detect_volleyball15.gif
├── data
│   ├── Court_Detection
│   ├── Player_Actions
│   └── Tracking
├── docs
│   ├── FINAL_REPORT.pdf
│   ├── Report
│   └── shakhansho_synopsis.docx
├── models
│   ├── actions
│   ├── DaSiamRPN
│   ├── players
│   └── yV7-tiny
├── outputs
│   ├── Court_Detection
│   ├── Player_Actions
│   └── Tracking
├── README.md
├── requirements.txt
└── src
    ├── Court_Detection
    ├── Players_Actions
    └── Tracking
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
