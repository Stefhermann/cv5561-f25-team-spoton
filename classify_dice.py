import pathlib
import sys
sys.dont_write_bytecode = True
# %load_ext autoreload
# %autoreload 1
# %aimport classify_dice
import ultralytics
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
NO_PLAYER = -1


example_frame = "data/frames/frame_0214.jpg"

def show_imgs_grid(imgs, width=10):
    imgs = imgs.copy()
    N,H,W,C = imgs.shape
    slots_missing = (-N) % width
    imgs = np.pad(imgs, ((0,slots_missing), (0,0), (0,0), (0,0)))
    NN = N + slots_missing
    assert imgs.shape == (NN,H,W,C)
    grid = einops.rearrange(imgs, "(nh nw) H W C -> (nh H) (nw W) C", nw=width)
    plt.figure()
    imshow(grid)


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
    dice_cropped = []
    dice_bbs = yolo_res.obb.xyxyxyxy[yolo_res.obb.cls == DIE,:,:]
    n_dice,_,_ = dice_bbs.shape
    assert dice_bbs.shape == (n_dice, 4, 2) # (die, point, x/y) -> coord
    for die_idx in range(n_dice):
        bb = dice_bbs[die_idx,...].cpu().numpy()
        cropped = crop_bb(img, bb, res_shape)
        dice_cropped.append(cropped)

    if len(dice_cropped) != 0:
        dice_cropped = np.stack(dice_cropped, axis=0)
    else:
        dice_cropped = np.zeros((0, res_shape[0], res_shape[1], 3))
    assert dice_cropped.shape == (n_dice, res_shape[0], res_shape[1], 3) # (die, x, y, channel) -> intensity
    return dice_cropped

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

def kmeans_color(dice_cropped, k=16):
    N,H,W,C = dice_cropped.shape
    k_means = sklearn.cluster.KMeans(n_clusters=k, random_state=1337, n_init=10)

    colors_only = einops.rearrange(dice_cropped, "N H W C -> (N H W) C")
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

def posterize(dice_cropped, k_means, palette):
    N,H,W,C = dice_cropped.shape
    assert C == 3
    flat_in = einops.rearrange(dice_cropped, "N H W C -> (N H W) C")
    flat_out = k_means.predict(flat_in)
    flat_colors = palette[flat_out,:]
    color_classes = einops.rearrange(flat_out, "(N H W) -> N H W", N=N, H=H, W=W)
    posterized = einops.rearrange(flat_colors, "(N H W) C -> N H W C", N=N, H=H, W=W, C=C)

    return color_classes,posterized

def dice_to_histograms(dice_cropped, k_means):
    n_dice,H,W,C = dice_cropped.shape
    assert C == 3

    flat_in = einops.rearrange(dice_cropped, "N H W C -> (N H W) C")
    flat_out = k_means.predict(flat_in)
    color_classes_flattened = einops.rearrange(flat_out, "(N H W) -> N (H W)", N=n_dice, H=H, W=W)
    
    palette_size = k_means.n_clusters
    histograms = np.zeros((n_dice, palette_size))

    for die_idx in range(n_dice): # TODO: vectorize?
        # print(f"{color_classes_flattened.shape=}")
        histograms[die_idx,:] = np.bincount(color_classes_flattened[die_idx], minlength=palette_size)
        histograms[die_idx,:] /= histograms[die_idx,:].sum()

    return histograms


def load_labeled_dice():
    labeled_dice_path = pathlib.Path("data/dice/labeled")
    labels_path = labeled_dice_path/"labels.txt"
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
            imgs.append(img_dict[id])
            

    ids = np.array(ids)
    players = np.array(players)
    values = np.array(values)
    imgs = np.stack(imgs, axis=0)

    N,H,W,C = imgs.shape
    assert imgs.shape == (N,H,W,C)
    assert ids.shape == (N,)
    assert values.shape == (N,)
    assert players.shape == (N,)
    
    return imgs,ids,players,values



def kmeans_cluster_players(histograms, n_players):
    """will hopefully provdie pretty good guesss of player assignments for each die"""
    player_k_means = sklearn.cluster.KMeans(n_clusters=n_players, random_state=1337, n_init=10) # `n_init=10` might be overkill

    player_k_means = player_k_means.fit(histograms)
    predictions = player_k_means.predict(histograms)
    # return player_k_means
    # reminder for self: want to return cluster labels for each player
    return player_k_means,predictions

# TODO: find out if svm actually needed/useful. perhaps just directly using kmeans will perform well enough. However, this svm also provides the ability/flexibility for us to correct kmeans-generated labels.
def fit_player_svm(histograms, label_train, n_players):
    svms = []
    for cls in range(n_players):
        in_class = np.where(label_train == cls, 1, -1)
        svm = sklearn.svm.LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=1337
        )
        svm.fit(histograms, in_class)
        svms.append(svm)
    return svms

def infer_player_svm(histograms, svms):
    n_samples = histograms.shape[0]
    n_classes = len(svms)
    scores = np.zeros((n_samples, n_classes))
    for cls,svm in enumerate(svms):
        scores[:,cls] = svm.decision_function(histograms)

    pred = np.argmax(scores, axis=1) # (n_samples,)
    return pred

def fit_player_colors(obb_model, unsupervised_dice_imgs, player_dice_imgs, palette_size=16, n_extra_clusters=1):
    """
    inputs:
    - 1 or more images of all dice on game table, unlabeled (used to learn color palette)
    - 1 or more images of each player's die (used to specify which index to give to each color)

    outputs:
    """

    n_players = len(player_dice_imgs)

    # (data shape assertions)
    C = 3
    N_unsupervised,H,W,_ = unsupervised_dice_imgs.shape
    assert unsupervised_dice_imgs.shape == (N_unsupervised, H, W, C)
    N_labeled = [0]*n_players
    for player_idx,dice_imgs in enumerate(player_dice_imgs):
        N_labeled[player_idx],H,W,_ = dice_imgs.shape
        assert dice_imgs.shape == (N_labeled[player_idx], H, W, C)

    print('Learning color palette from unsupervised dice img(s)...')
    dice_cropped = []
    for img_idx in range(N_unsupervised):
        unsupervised_img = unsupervised_dice_imgs[img_idx,:,:,:]
        res = obb_model(unsupervised_img)[0]
        # unsupervised_img = cv2.resize(unsupervised_img, (512,512))
        # res = res[0]
        # print(res)
        curr_dice_cropped = batch_crop_dice(unsupervised_img, res) # (n_dice_in_img, H, W, C)
        show_imgs_grid(curr_dice_cropped)
        dice_cropped.append(curr_dice_cropped)
    dice_cropped = np.concat(dice_cropped, axis=0) # (n_dice_unsupervised, H, W, C)

    color_k_means,palette = kmeans_color(dice_cropped, k=palette_size)
    
    # debug: view palette
    imshow(einops.rearrange(palette, "(h w) C -> h w C", w=8))
    plt.title("Color palette")

    print("Clustering to obtain player assignments...")
    all_dice_cropped = []
    histograms = []
    class_labels = []
    for p in range(n_players):
        n_imgs,H,W,_ = player_dice_imgs[p].shape
        assert player_dice_imgs[p].shape == (n_imgs, H, W, C)

        # debug: view all images for this player
        print(p)
        plt.figure()
        show_imgs_grid(player_dice_imgs[p], width=1)
        plt.title(f"Player {p} images")
        plt.show()

        for img_idx in range(n_imgs):
            labeled_img = player_dice_imgs[p][img_idx,:,:,:]

            res = obb_model(labeled_img)[0]
            labeled_dice_cropped = batch_crop_dice(labeled_img, res)
            n_dice,H,W,_ = labeled_dice_cropped.shape
            assert labeled_dice_cropped.shape == (n_dice, H, W, C)
            all_dice_cropped.append(labeled_dice_cropped)

            # debug: view dice in this image
            plt.figure()
            show_imgs_grid(labeled_dice_cropped)
            plt.title(f"Dice in image")
            plt.show()
            
            curr_histograms = dice_to_histograms(labeled_dice_cropped, color_k_means)
            curr_class_labels = np.array([p] * n_dice)
            histograms.append(curr_histograms)

            class_labels.append(curr_class_labels)

    all_dice_cropped = np.concat(all_dice_cropped, axis=0)
    histograms = np.concat(histograms, axis=0)    
    class_labels = np.concat(class_labels, axis=0)

    print(f"{all_dice_cropped.shape=}")
    plt.figure()
    show_imgs_grid(all_dice_cropped)
    plt.title("Dice found in `player_dice_imgs`")

    player_k_means,cluster_assignments = kmeans_cluster_players(
        histograms=histograms,
        n_players=n_players+n_extra_clusters # add a pseudo-player to try to catch spurious dice detections, e.g. the emblems on cards
    )

    # debug: visualize
    pca = sklearn.decomposition.PCA(n_components=2, whiten=True, random_state=1337)
    hist_pca = pca.fit_transform(histograms)
    plt.figure()
    plt.scatter(hist_pca[:,0], hist_pca[:,1], c=class_labels)
    plt.colorbar()
    plt.title("Draft player assignments")
    plt.figure()
    plt.scatter(hist_pca[:,0], hist_pca[:,1], c=cluster_assignments)
    plt.title("Cluster assignments")
    plt.colorbar()

    cluster_to_player = {}
    for cluster_id in range(n_players+n_extra_clusters):
        is_member = (cluster_assignments==cluster_id)
        plt.figure()
        show_imgs_grid(all_dice_cropped[is_member,:,:,:])
        print(class_labels[is_member])
        classes_present = np.unique(class_labels[is_member])
        print(classes_present)
        if len(classes_present) == 1:
            cluster_to_player[cluster_id] = (classes_present.item())
            plt.title(f"Cluster {cluster_id} members (homogeneous) -> Player {classes_present.item()}")
        else: plt.title(f"Cluster {cluster_id} members (heterogeneous)")
    print(f"{cluster_to_player=}")

    return color_k_means,player_k_means,cluster_to_player

def predict_player_colors(dice_cropped, color_k_means, player_k_means, cluster_to_player: dict):
    n_dice,H,W,_ = dice_cropped.shape
    assert dice_cropped.shape == (n_dice,H,W,3)

    histograms = dice_to_histograms(dice_cropped, color_k_means)
    cluster_memberships = player_k_means.predict(histograms)
    cluster_to_player_v = np.vectorize(lambda x: cluster_to_player.get(x, NO_PLAYER))
    player_assignments = cluster_to_player_v(cluster_memberships)
    return player_assignments

# def predict_player_colors(img, obb_res, color_k_means, player_k_means, cluster_to_player: dict):
#     dice_cropped = batch_crop_dice(img, obb_res)
#     n_dice,H,W,_ = dice_cropped.shape
#     assert dice_cropped.shape == (n_dice,H,W,3)

#     histograms = dice_to_histograms(dice_cropped, color_k_means)
#     cluster_memberships = player_k_means.predict(histograms)
#     NO_PLAYER = -1
#     cluster_to_player_v = np.vectorize(lambda x: cluster_to_player.get(x, NO_PLAYER))
#     player_assignments = cluster_to_player_v(cluster_memberships)
#     return player_assignments

def predict_die_value(dice_cropped, value_cls_model):
    res = value_cls_model([die for die in dice_cropped])
    n_dice = len(res)
    top1 = np.zeros(n_dice)
    top1conf = np.zeros(n_dice)

    for i,r in enumerate(res):
        top1[i] = r.probs.top1
        top1conf[i] = r.probs.top1conf

    return top1,top1conf

class ObbShim:
    # making this shim so we can set fields of immutable objects (e.g. setting obb.cls, which does not ordinarily have a setter)
    # a little scuffed i think
    def __init__(self, original_res):
        self._original_res = original_res

    def __getattr__(self, name):
        if name == '_original_res': return self._original_res
        return getattr(self._original_res, name)

class RecognitionModel:
    def __init__(
            self,
            obb_model_path="model/rdg_obb/weights/best.pt",
            value_cls_model_path="classifier_models/die_number_classifier/weights/best.pt",
        ):
        self.obb_model = YOLO(obb_model_path)
        self.value_cls_model = YOLO(value_cls_model_path)

        self.color_k_means = None
        self.player_k_means = None
        self.cluster_to_player = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def train_player_vocab(self, unsupervised_imgs, player_imgs, n_extra_clusters=1):
        self.color_k_means,self.player_k_means,self.cluster_to_player = fit_player_colors(
            self.obb_model,
            unsupervised_imgs,
            player_imgs,
            n_extra_clusters=n_extra_clusters,
        )

    def predict(self, frame, obb_confidence=0.25, value_confidence=0.55):
        obb_res = self.obb_model(frame, conf=obb_confidence)[0]
        obb_res.obb = ObbShim(obb_res.obb) # what horrors am i committing
        obb_res.obb.cls = obb_res.obb.cls.detach().cpu().numpy()
        dice_indices = np.argwhere(obb_res.obb.cls == DIE)

        dice_cropped = batch_crop_dice(frame, obb_res, res_shape=(128,128))
        dice_players = predict_player_colors(dice_cropped, self.color_k_means, self.player_k_means, self.cluster_to_player)
        dice_values,dice_val_conf = predict_die_value(dice_cropped, self.value_cls_model)

        bad_dice_mask = (
            (dice_val_conf < value_confidence) |
            (dice_players == NO_PLAYER)
        )
        
        objects_to_exclude = dice_indices[bad_dice_mask]
        # hacky method but gets the job done
        obb_res.obb.cls[objects_to_exclude] = -1

        INVALID = -1
        for die,obj_idx in enumerate(dice_indices):
            if bad_dice_mask[die]:
                obb_res.obb.cls[obj_idx] = INVALID
            else:
                player = dice_players[die]
                value = dice_values[die]
                obb_res.obb.cls[obj_idx] = f"{player}_{value}"

        return obb_res

def main():
    recognition_model = RecognitionModel()

    # unsupervised_imgs = np.stack(cv2.imread("data/misc/all.jpg"), axis=0)
    unsupervised_imgs = np.stack([cv2.imread("data/misc/all.jpg")], axis=0)
    player_imgs = [
        np.stack([cv2.imread("data/misc/red.jpg")], axis=0),
        np.stack([cv2.imread("data/misc/yellow.jpg")], axis=0),
        np.stack([cv2.imread("data/misc/purple.jpg")], axis=0),
    ]

    recognition_model.train_player_vocab(
        unsupervised_imgs,
        player_imgs,
    )

    return recognition_model.predict(unsupervised_imgs[0,:,:,:])




if __name__ == '__main__':
    main()
    # obb_model = YOLO("model/rdg_obb/weights/best.pt")

    # example_frame = "data/frames/frame_0214.jpg"
    # img = cv2.imread(example_frame)
    # # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # imshow(img)


    # res = obb_model(img, task='obb')[0]
    # res

    # res.obb

    # dice_bbs = res.obb.xyxyxyxy[res.obb.cls == DIE, ...]
    # die_idx = 0
    # bb = dice_bbs[die_idx,...].cpu().numpy()


    # res_shape = (128,128)
