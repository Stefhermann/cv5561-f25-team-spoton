import os
import random
import shutil

labels = "data/dice/labeled/labels.txt"
die_dir = "data/dice/labeled"
output_dir = "data/dice/die_yolo"


split = 0.8

images = []

with open(labels, "r") as f:
    for line in f:
        line = line.strip().split(" ")

        img_name, _, number = line

        img_name = img_name + ".png"

        print(img_name, number)

        images.append((img_name, number))

random.shuffle(images)
split_idx = int(len(images) * split)
train_entries = images[:split_idx]
val_entries = images[split_idx:]


def split(name, split_entries):
    for img_name, number in split_entries:
        src = os.path.join(die_dir, img_name)

        class_dir = os.path.join(output_dir, name, number)
        os.makedirs(class_dir, exist_ok=True)

        dst = os.path.join(class_dir, img_name)

        shutil.copy2(src, dst)


split("train", train_entries)
split("val", val_entries)
