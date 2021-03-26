import argparse
import glob
import os
import pathlib
import random
import xml.etree.ElementTree as ET
from collections import namedtuple
import shutil

import albumentations as A
from albumentations.augmentations.transforms import RandomCrop
import cv2
import pandas as pd
from numpy import float_power, true_divide
from pascal_voc_writer import Writer
from PIL import Image
from hydra.utils import get_original_cwd, to_absolute_path
import logging

logger = logging.getLogger(__name__)


def xml_to_csv(path):
    logger.info("augmente pascal fromat data in %s", path)
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            imagefile = root.find("filename").text
            true_file = None
            if pathlib.Path(imagefile).is_file():
                true_file = pathlib.Path(imagefile).absolute()
            elif pathlib.Path(path + "/" + imagefile).is_file():
                true_file = pathlib.Path(path + "/" + imagefile).absolute()
            else:
                true_file = imagefile
            value = (
                str(true_file),
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def aug(source, images_output_path, size):
    images_path = images_output_path + "/JPEGImages/"
    os.makedirs(images_path, exist_ok=True)

    xml_path = images_output_path + "/Annotations/"
    os.makedirs(xml_path, exist_ok=True)

    transform = A.Compose(
        [
            # A.CLAHE(),
            # A.RandomScale(scale_limit=[0.5, 1]),
            # A.RandomCrop(width=450, height=450),
            A.OneOf(
                [
                    A.Sequential(
                        [A.RandomCrop(width=800, height=600), A.RandomRotate90()]
                    ),
                    # A.Sequential(
                    #     [
                    #         A.RandomSizedBBoxSafeCrop(width=800, height=600),
                    #         A.RandomRotate90(),
                    #     ]
                    # ),
                    A.Sequential(
                        [
                            A.RandomScale(scale_limit=0.2),
                            A.Flip(),
                            A.RandomRotate90(),
                        ],
                        # p=0.3,
                    ),
                    A.Sequential(
                        [
                            A.Rotate(),
                        ],
                        p=0.3,
                    ),
                ]
            )
            # A.Transpose(),
            # A.Resize(0.9, 0.9),
            # A.Blur(blur_limit=3),
            # A.OpticalDistortion(),
            # A.GridDistortion(),
            # A.HueSaturationValue(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.5, label_fields=["class_labels"]
        ),
    )

    rows = []
    random.seed(42)

    images_index = 1
    for name, group in source.groupby("filename"):
        row = group.iloc[0]
        print(row["filename"])
        image = cv2.imread(row["filename"])
        same = set()

        bboxes = []
        class_labels = []

        aleady_box = {}
        for _, vrow in group.iterrows():
            bboxes.append([vrow["xmin"], vrow["ymin"], vrow["xmax"], vrow["ymax"]])
            class_labels.append(vrow["class"])
            aleady_box[vrow["class"]] = set()
        all_count = 0
        print(aleady_box)
        while int(all_count) < size:
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            file_name = f"{images_index}.jpg"

            if len(augmented["bboxes"]) < 1:
                continue

            writer = Writer(
                file_name, augmented["image"].shape[1], augmented["image"].shape[0]
            )

            findbox = False
            for index, bbox in enumerate(augmented["bboxes"]):
                x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox[:4])

                same.add(x_min)
                rows.append(
                    {
                        "filename": f"{images_path}/{file_name}",
                        "width": augmented["image"].shape[1],
                        "height": augmented["image"].shape[0],
                        "class": augmented["class_labels"][index],
                        "xmin": x_min,
                        "ymin": y_min,
                        "xmax": x_max,
                        "ymax": y_max,
                        "imageindex": str(images_index),
                    }
                )
                writer.addObject(
                    augmented["class_labels"][index], x_min, y_min, x_max, y_max
                )
                if len(aleady_box[augmented["class_labels"][index]]) >= size:

                    continue
                aleady_box[augmented["class_labels"][index]].add(x_min)
                findbox = True
            if findbox:
                cv2.imwrite(f"{images_path}/{file_name}", augmented["image"])
                writer.save(f"{xml_path}/{images_index}.xml")
                images_index += 1
                print(aleady_box)

            all_count = sum([min(len(v), size) for k, v in aleady_box.items()]) / len(
                aleady_box
            )
    df = pd.DataFrame(rows)
    return df


def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def augmente(cfg):

    args = cfg.augmentation

    # label_map = label_map_util.load_labelmap(args.labels_path)
    # label_map_dict = label_map_util.get_label_map_dict(label_map)

    df = xml_to_csv(to_absolute_path(args.input_dir))
    print(df)
    image_ouput_dir = args.output_dir
    if os.path.isdir(image_ouput_dir):
        shutil.rmtree(image_ouput_dir)

    df = aug(df, image_ouput_dir, args.factor)

    train_df = df.sample(frac=args.train_factor, random_state=100)
    test_df = df[~df.index.isin(train_df.index)]

    filepath = os.path.join(image_ouput_dir, "ImageSets", "Main")
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, "train.txt"), "w") as f:
        f.write(" ".join(train_df["imageindex"].to_list()))

    with open(os.path.join(filepath, "test.txt"), "w") as f:
        f.write(" ".join(test_df["imageindex"].to_list()))

    with open(os.path.join(filepath, "labels.txt"), "w") as f:
        f.writelines([v + "\n" for v in set(df["class"].to_list())])
