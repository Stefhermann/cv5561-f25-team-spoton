# SpotOn

## Notes for user

#### Venv
```bash
uv venv # or use `python -m venv` and pip if you like ig
source .venv/bin/activate
uv pip install -r requirements.txt # TODO: simplify requirements.txt
```

#### OpenCV dependencies
`sudo apt install libglib2.0-0 libsm6 libxrender1 libxext6` may be required to install dependencies for opencv gui stuff

#### Data
Copy the contents of [this google drive folder](https://drive.google.com/drive/folders/1VSm9NnftphKB87fiTHYBRZ7yb-MMln61?usp=sharing) into `/video_data`, then run `python data_prep/video_data_rename.py` to rename/rearrange these videos to the format expected by the project.

### Running
`python pipeline.py` to run on example video.
To try another video, just modify the script to take new input.


## Advanced notes
### Training
- `data_prep/yolo_preprocess.py` must be run before training YOLO model

### Tips for connecting camera
Camdroid and OBS seem to work well

## Methods

- Recognition (`classify_dice.py`)
Bounding boxes (with classes `die`, `ace`, `two`, `three`, `four`) are via YOLOv8-obb. Dice are then further classified by the player classifier (color-based via $k$-means) and the value classifier (YOLOv8-cls). The recognition pipeline outputs classified bounding boxes (classes: `ace`, `two`, `three`, `four`, `red_1`, `red_2`, ..., `blue_5`, `blue_6`, and `INVALID_DIE`).

- State tracking and scoring (`spatial_association.py`, `tracking.py`, `scoring.py`)
Dice are associated with cards based on if their bounding box centroids fall within the card's bounding box. This information is passed to the dice tracker, which stores information about the game state over time. We experimented with more complex methods and enhancements to this procedure (e.g. IoU-based associatiation, Kalman filter in tracker to mitigate frame-to-frame flicker), but we found simple methods to produce the best results.


Pipeline visualization
```mermaid
---
config:
    layout: elk
---
flowchart
%% --- Styles ---
classDef data fill:#ccccee,stroke:#555599,stroke-width:1px,color:#000;
classDef func fill:#c9c9c9,stroke:#555555,stroke-width:1px,color:#000;
style recognition fill:#fefaee,stroke:#000000;
style state-tracking fill:#fefaee,stroke:#000000;

%% --- Node definitions ---
%% Data nodes

%% Function nodes
A-1([Input Frame]):::data

subgraph recognition ["Recognition"]
    A0([Input Frame]):::data
    A1{{YOLO OBB}}:::func
    A2([Card and Die Bounding Boxes]):::data
    A3{{Dice Cropper}}:::func
    A4([Dice Closeups]):::data
    A5{{Color Classifier #40;KMeans#41;}}:::func
    A6{{Value Classifier #40;YOLO CLS#41;}}:::func
    A7([Die Player]):::data
    A8([Die Value]):::data
    A9{{Combine Results}}:::func
end

A10([Labeled OBBs]):::data
A11{{Spatial Association}}:::func
A12([Associations]):::data

subgraph state-tracking ["State Tracking"]
    A19([Past Game State]):::data
    A18{{Game State Tracker}}:::func
    A20([Game State]):::data
end

A13{{Scoring}}:::func
A14{{Visualization}}:::func
A15([Scores]):::data
A16([Output Frames]):::data



%% --- Edges ---
A-1 --> A0
A0 --> A1
A0 --> A3
A1 --> A2
A2 --> A3
A2 --> A9
A3 --> A4
A4 --> A5
A5 --> A7
A4 --> A6
A6 --> A8
A7 --> A9
A8 --> A9
A9 --> A10
A10 --> A11
A11 --> A12
A12 --> A18
A19 --> A18
A18 --> A20
A13 --> A15
A20 --> A13
A-1 --> A14
A15 --> A14
A20 --> A14
A14 --> A16
```

