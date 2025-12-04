import pathlib

# should be run from repo root

data_path = pathlib.Path("data/")
subdirs = ['obb-train', 'obb-valid', 'obb-test']

old_schema = {
    0: 'ace',
    1: 'blue_1',
    2: 'blue_2',
    3: 'blue_3',
    4: 'blue_4',
    5: 'blue_5',
    6: 'blue_6',
    7: 'four',
    8: 'red_1',
    9: 'red_2',
    10: 'red_3',
    11: 'red_4',
    12: 'red_5',
    13: 'red_6',
    14: 'three',
    15: 'two',
    16: 'yellow_1',
    17: 'yellow_2',
    18: 'yellow_3',
    19: 'yellow_4',
    20: 'yellow_5',
    21: 'yellow_6',
    22: 'die',
}

old_dice_labels = {
    'blue_1',
    'blue_2',
    'blue_3',
    'blue_4',
    'blue_5',
    'blue_6',
    'red_1',
    'red_2',
    'red_3',
    'red_4',
    'red_5',
    'red_6',
    'yellow_1',
    'yellow_2',
    'yellow_3',
    'yellow_4',
    'yellow_5',
    'yellow_6',
    'die',
}

new_schema = {
    0: 'die',
    1: 'ace',
    2: 'two',
    3: 'three',
    4: 'four',
}

def construct_mapping(old_schema, old_dice_labels, new_schema):
    inverse_new = {v:k for k,v in new_schema.items()}
    mapping = {}
    for idx,label in old_schema.items():
        if label in old_dice_labels: label = 'die'
        # mapping[idx] = inverse_new[label]
        mapping[str(idx)] = str(inverse_new[label])
    return mapping

# thus,
label_mapping = construct_mapping(old_schema, old_dice_labels, new_schema)

# unify dice labels
for sub in subdirs:
    input_dir = data_path/sub/"labels_raw"
    output_dir = data_path/sub/"labels"

    if not input_dir.exists():
        print(f"Alert: {input_dir} does not exist. Continuing...")
        continue
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_dir.glob("*.txt"):
        output_path = output_dir/input_path.name
        with open(input_path, 'r', encoding='utf8') as input_labels, open(output_path, 'w+', encoding='utf8') as output_labels:
            for line in input_labels:
                label,rest = line.split(" ", maxsplit=1)
                output_labels.write(f"{label_mapping[label]} {rest}")

if __name__ == '__main__':
    ...