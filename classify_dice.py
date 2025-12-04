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

def kmeans_color(imgs, k=16):
    N,H,W,C = imgs.shape
    k_means = sklearn.cluster.KMeans(n_clusters=k, random_state=1337, n_init=10)

    colors_only = einops.rearrange(imgs, "N H W C -> (N H W) C")
    k_means = k_means.fit(colors_only)

    palette = []
    for c_idx in range(k_means.cluster_centers_.shape[0]):
        color = k_means.cluster_centers_[c_idx,:]
        palette.append(color)
        # plt.figure()
        # classify_dice.imshow(color.reshape(1,1,3).astype(np.uint8))
        # plt.show()
    palette = np.stack(palette, axis=0).astype(np.uint8)
    # imshow(einops.rearrange(palette, "(h w) C -> h w C", w=8))

    return k_means,palette

def posterize(imgs, k_means, palette):
    N,H,W,C = imgs.shape
    assert C == 3
    flat_in = einops.rearrange(imgs, "N H W C -> (N H W) C")
    flat_out = k_means.predict(flat_in)
    flat_colors = palette[flat_out,:]
    classes = einops.rearrange(flat_out, "(N H W) -> N H W", N=N, H=H, W=W)
    posterized = einops.rearrange(flat_colors, "(N H W) C -> N H W C", N=N, H=H, W=W, C=C)

    return classes,posterized

def load_labeled_dice():
    labeled_dice_path = pathlib.Path("data/dice/labeled")
    labels_path = labeled_dice_path/"labels.txt"
    labels = dict()
    legal_player_labels = {
        'R': 0,
        'Y': 1,
        'B': 2,
    }   
    legal_value_labels = {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
    }

    spoiled_indices = set()
    ids = []
    players = []
    values = []
    imgs = []
    
    img_dict = dict()
    for img_path in labeled_dice_path.glob("*.png"):
        img_id = int(img_path.stem)
        img_dict[img_id] = cv2.imread(img_path)

    with open(labels_path, 'r') as label_file:
        for line in label_file:
            idx,player_label,value_label = line.split()
            id = int(idx)
            if id not in img_dict or player_label not in legal_player_labels or value_label not in legal_value_labels:
                spoiled_indices.add(idx)
                continue
            ids.append(id)
            players.append(legal_player_labels[player_label])
            values.append(legal_value_labels[value_label])

    ids = np.array(ids)
    players = np.array(players)
    values = np.array(values)
    imgs = np.stack(imgs, axis=0)
    
    return imgs,ids,players,values

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
