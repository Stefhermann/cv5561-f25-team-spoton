# SpotOn

### Setup
```bash
uv venv # or use `python -m venv` and pip if you like ig
source .venv/bin/activate
uv pip install -r requirements.txt # TODO: simplify requirements.txt
```

### Notes
- `data_prep/yolo_preprocess.py` must be run before training YOLO model

### Training
Download training videos from google drive, use `data_prep/video_data_rename.py` to clean up names

### Tips for connecting camera
Camdroid and OBS seem to work well

### OpenCV dependencies
`sudo apt install libglib2.0-0 libsm6 libxrender1 libxext6` may be required to install dependencies for opencv gui stuff
