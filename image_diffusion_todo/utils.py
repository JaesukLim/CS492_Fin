import numpy as np
import os
import ndjson
import torch
import struct
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def strokes_to_tensor(strokes, H, W, S, pad_value=0):
    """
    Convert strokes into a tensor of shape [S, H, W].

    Parameters:
        strokes (list): List of strokes where each stroke is a list of [x, y] coordinates.
        H (int): Height of the tensor.
        W (int): Width of the tensor.
        S (int): Maximum number of strokes (padded if less).
        pad_value (int): Value to use for padding.

    Returns:
        numpy.ndarray: Tensor of shape [S, H, W].
    """
    tensor = np.zeros((S, H, W), dtype=np.float16)

    for i, stroke in enumerate(strokes):
        if i >= S:
            break
        x_coords, y_coords = stroke
        for x, y in zip(x_coords, y_coords):
            x = x // 4
            y = y // 4
            if 0 <= x < H and 0 <= y < W:
                tensor[i, x, y] = 255.0  # Set value at stroke points

    # Pad remaining strokes with the padding value
    if len(strokes) < S:
        tensor[len(strokes):] = pad_value

    return torch.tensor(tensor, dtype=torch.float16)


def tensor_to_strokes(tensor, pad_value=0):
    """
    Convert tensor of shape [S, H, W] back to strokes.

    Parameters:
        tensor (numpy.ndarray): Input tensor of shape [S, H, W].
        pad_value (int): Value used for padding.

    Returns:
        list: List of strokes where each stroke is a list of [x, y] coordinates.
    """
    strokes = []
    tensor = tensor.cpu().numpy()  # Convert torch tensor to numpy array for processing
    B, S, H, W = tensor.shape

    for b in range(B):
        temp = []
        print(tensor[b])
        for i in range(S):
            if np.all(tensor[b][i] == pad_value):
                continue  # Skip padded strokes
            x_coords, y_coords = np.where(tensor[b][i] == 255.0)
            x_coords *= 4
            y_coords *= 4
            temp.append([x_coords.tolist(), y_coords.tolist()])
        strokes.append(temp)

    return strokes

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

def draw_full_images(strokes):
    full_images = []
    for item in strokes:
        images = []
        for i in range(len(item)):
            image = draw_strokes(item[:i + 1])
            # add stroke number
            draw = ImageDraw.Draw(image)

            draw.text((20, 10), text=f"stroke #{i}", fill="black")
            images.append(image)

        images_concat = image_grid(images, 1, len(images))
        full_images.append(images_concat)

    return full_images