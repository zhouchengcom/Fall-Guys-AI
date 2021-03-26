import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import pprint


def main():
    image = cv2.imread("cuiyan.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def visualize(image):
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(image)
        # plt.show()

    transform = A.Compose(
        [
            A.RandomCrop(111, 222),
            A.OneOf([A.RGBShift(), A.HueSaturationValue()]),
        ]
    )

    random.seed(42)
    transformed = transform(image=image)
    visualize(transformed["image"])

    A.save(transform, "./transform.json")
    A.save(transform, "./transform.yml", data_format="yaml")
    pprint.pprint(A.to_dict(transform))


main()