import pathlib
import sys
sys.dont_write_bytecode = True
# %load_ext autoreload
# %autoreload 1
# %aimport classify_dice
from ultralytics import YOLO
import sklearn
import cv2
from matplotlib import pyplot as plt
import numpy as np
import einops

"""
Strategy outline

Inputs:
- video frames?
- dice bounding boxes?

Desired outputs (remember, MVP!!):


- Identify dice oriented bounding boxes via yolo, potentially with low-res video
- For oob in frame:
    - Crop image to frame
    - Run kmeans to identify dominant colors and thereby hopefully player
    - Use a small CNN (yolo? if so should use LOTS of hue augmentation) to classify face

"""

DIE = 0

example_frame = "data/frames/frame_0214.jpg"

def generate_dice_dataset(obb_model):
    frames_path = pathlib.Path("data/frames")
    dice_path = pathlib.Path("data/dice")

    all_dice = []

    for f_path in frames_path.glob("*.jpg"):
        # f_path.stem
        frame = cv2.imread(f_path)
        yolo_res = obb_model(frame, task='obb')[0]
        frame_dice = batch_crop_dice(frame, yolo_res)
        all_dice.append(frame_dice)

    all_dice = np.concat(all_dice, axis=0)
    return all_dice


def imshow(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def infer(frame):
    ...

# def make_color_patch(color, h=32,w=32)



def imshow(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def crop_bb(img, bb, res_shape=(128,128)):
    res_points = np.array([
        [0, 0],
        [0, res_shape[1]],
        [res_shape[0], res_shape[1]],
        [res_shape[0], 0],
    ])
    M = cv2.getPerspectiveTransform(np.float32(bb), np.float32(res_points))
    cropped = cv2.warpPerspective(img, M, res_shape)
    return cropped

def batch_crop_dice(img, yolo_res, res_shape=(128,128)):

    res = []
    dice_bbs = yolo_res.obb.xyxyxyxy[yolo_res.obb.cls == DIE,:,:]
    n_dice,_,_ = dice_bbs.shape
    assert dice_bbs.shape == (n_dice, 4, 2) # (die, point, x/y) -> coord
    for die_idx in range(n_dice):
        bb = dice_bbs[die_idx,...].cpu().numpy()
        cropped = crop_bb(img, bb, res_shape)
        res.append(cropped)

    if len(res) != 0:
        res = np.stack(res, axis=0)
    else:
        res = np.zeros((0, res_shape[0], res_shape[1], 3))
    assert res.shape == (n_dice, res_shape[0], res_shape[1], 3) # (die, x, y, channel) -> intensity
    return res

def write_dice():
    all_dice = np.load('data/dice/all_dice.npz')['arr_0']
    out_path = pathlib.Path('data/dice/all_dice/')
    label_path = pathlib.Path('data/dice/labeled')
    n_dice = all_dice.shape[0]
    np.random.seed(1337)
    to_label = set(x.item() for x in np.random.permutation(n_dice)[:200])
    print(to_label)
    for die_idx in range(n_dice):
        cv2.imwrite(out_path/f'{die_idx}.png', all_dice[die_idx,:,:,:])
        if die_idx in to_label:
            cv2.imwrite(label_path/f'{die_idx}.png', all_dice[die_idx,:,:,:])
    with open(label_path/"label_template.txt", 'w') as label_file:
        label_file.write(" \n".join(map(str, sorted(to_label))))




if __name__ == '__main__':
    obb_model = YOLO("model/rdg_obb/weights/best.pt")

    example_frame = "data/frames/frame_0214.jpg"
    img = cv2.imread(example_frame)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imshow(img)

    res = obb_model(img, task='obb')[0]
    res

    res.obb

    dice_bbs = res.obb.xyxyxyxy[res.obb.cls == DIE, ...]
    die_idx = 0
    bb = dice_bbs[die_idx,...].cpu().numpy()


    res_shape = (128,128)