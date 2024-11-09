# %% md
### Sample code for visualizing the stroke data
# %%
import os
import ndjson
import numpy as np
import struct
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# %% md
### Helper functions
# %%
def image_grid(imgs, rows, cols):
    """
    Concatenates multiple images
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def draw_strokes(strokes, height=256, width=256):
    """
    Make a new PIL image with the given strokes
    """
    image = Image.new("RGB", (width, height), "white")
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        # concat x and y coordinates
        points = list(zip(stroke[0], stroke[1]))

        # draw all points
        # image_draw.point(points, fill=0)
        image_draw.line(points, fill=0)

    return image


from tqdm import tqdm
import json

for category in ["cat", "garden", "helicopter"]:
    data_path = f"./data/{category}.ndjson"
    indices_path = f"./sketch_data/{category}/train_test_indices.json"

    with open(data_path, 'r') as f:
        data = ndjson.load(f)

    with open(indices_path, 'r') as f:
        indices = json.load(f)

    for idx in tqdm(indices["train"]):
        item = data[idx]
        strokes = item['drawing']
        for i in range(len(strokes)):
            image = draw_strokes(strokes[:i + 1])
            image.save(f"./images_first_stroke/{category}/train/{category}_{idx}_stroke_{i}.png")
        image = draw_strokes([strokes[0]])
        image.save(f"./images_first_stroke/{category}/train/{category}_{idx}.png")
        image_last = draw_strokes(strokes)
        image.save(f"./images_last_stroke/{category}/train/{category}_{idx}.png")

    for idx in tqdm(indices["test"]):
        item = data[idx]
        strokes = item['drawing']
        for i in range(len(strokes)):
            image = draw_strokes(strokes[:i + 1])
            image.save(f"./images_first_stroke/{category}/test/{category}_{idx}_stroke_{i}.png")
        image = draw_strokes([strokes[0]])
        image.save(f"./images_first_stroke/{category}/test/{category}_{idx}.png")
        image_last = draw_strokes(strokes)
        image.save(f"./images_last_stroke/{category}/test/{category}_{idx}.png")


