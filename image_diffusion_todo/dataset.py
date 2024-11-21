import os
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from rdp import rdp

import json
import ndjson
import numpy as np
import pickle


def listdir(dname):
    fnames = list(
        chain(
            *[
                list(Path(dname).rglob("*." + ext))
                for ext in ["png", "jpg", "jpeg", "JPG"]
            ]
        )
    )
    return fnames


def scale_sketch(sketch, size=(448, 448)):
    max_abs_0 = torch.max(torch.abs(sketch[:, 0]))
    max_abs_1 = torch.max(torch.abs(sketch[:, 1]))

    sketch[:, 0] = sketch[:, 0] / max_abs_0
    sketch[:, 1] = sketch[:, 1] / max_abs_1

    sketch_rescale = sketch * torch.tensor([[size[0], size[1], 1]], dtype=torch.float)
    return sketch_rescale.int()


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


from PIL import Image, ImageDraw


def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    """
    x: [B,C,H,W]
    """
    output = []
    B, S, _ = x.size()
    x = x.cpu().int()
    for b in range(B):
        # x[b, :, :] = scale_sketch(x[b, :, :], size=(256, 256))
        input_strokes = []
        prev_x = 0
        prev_y = 0
        temp_x = []
        temp_y = []
        for s in range(1, S):
            # Remove Padding
            if x[b, s, 0] == 0 and x[b, s, 1] == 0 and x[b, s, 2] == 1:
                break
            temp_x.append(prev_x + x[b, s, 0])
            temp_y.append(prev_y + x[b, s, 1])
            prev_x = x[b, s, 0]
            prev_y = x[b, s, 1]
            if x[b, s, 2] == 1:
                input_strokes.append([temp_x, temp_y])
                temp_x = []
                temp_y = []

        images = []
        for i in range(len(input_strokes)):
            image = draw_strokes(input_strokes[:i + 1])

            # add stroke number
            draw = ImageDraw.Draw(image)
            draw.text((20, 10), text=f"stroke #{i}", fill="black")
            images.append(image)

        # concatenate all images
        images_concat = image_grid(images, 1, len(images))
        output.append(images_concat)
    return output


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(
            self, root: str, split: str, transform=None, category="cat", stroke_type="full", label_offset=1,
            num_classes=1, stroke_length=30, coord_length=96
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_offset = label_offset
        self.category = category
        self.stroke_type = stroke_type
        self.num_classes = num_classes
        self.stroke_length = stroke_length
        self.coord_length = coord_length

        data_path = f"{root}/{category}.ndjson"
        indices_path = f"../sketch_data/{category}/train_test_indices.json"
        cache_path = f"{root}/{category}_{split}.pickle"

        with open(data_path, 'r') as f:
            self.data = ndjson.load(f)

        with open(indices_path, 'r') as f:
            indices = json.load(f)
        self.indices = indices[split]
        self.labels = [label_offset] * len(self.indices)  # 모든 파일의 라벨

        if os.path.exists(cache_path):
            print("Using Cached Data")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self.vectors = cached_data["vectors"]
            self.pen_label = cached_data["pen_label"]

        else:
            print("There is no cached data. Make new one")
            #### Preprocess into 3-Tuple ####
            full_data = []
            full_pen_label = []
            for idx in tqdm(self.indices):
                item = self.data[idx]
                strokes = item["drawing"]
                temp_stroke = [[0, 0, 1]]
                prev_x = 0
                prev_y = 0
                for stroke in strokes:
                    for i, (xi, yi) in enumerate(zip(stroke[0], stroke[1])):
                        pen_state = 0 if i != len(stroke[0])-1 else 1
                        temp_stroke.append([xi - prev_x, yi - prev_y, pen_state])
                        prev_x = xi
                        prev_y = yi

                temp_result = np.array(temp_stroke)
                # Reducing with RDP
                eps = 0.1
                while temp_result.shape[0] > self.coord_length:
                    # print(f"Reducing with RDP | current length: {temp_result.shape[0]} | eps: {eps}")
                    mask = rdp(temp_result[:, 0:2], epsilon=eps, return_mask=True)
                    temp_result = temp_result[mask]
                    eps += 0.1

                # Add Padding
                if temp_result.shape[0] < self.coord_length:
                    padding = np.array([[0, 0, 1] for _ in range(self.coord_length - temp_result.shape[0])])
                    temp_result = np.concatenate((temp_result, padding), axis=0)

                temp_result = temp_result.astype(np.float16)
                # Scale
                # temp_result[:, 0:2] /= np.std(temp_result[:, 0:2])
                full_data.append(torch.tensor(temp_result, dtype=torch.float32))
                full_pen_label.append(torch.tensor(temp_result[:, 2], dtype=torch.float32).unsqueeze(-1))

            self.vectors = full_data
            self.pen_label = full_pen_label

            print(f"Full size: {len(self.vectors)}, {self.vectors[0].size()}")
            print(f"Example Data: {self.vectors[0]}")

            with open(cache_path, "wb") as f:
                pickle.dump({"vectors": self.vectors, "pen_label": self.pen_label}, f)

    def __getitem__(self, idx):
        return self.vectors[idx], self.pen_label[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class QuickDrawDataModule(object):
    def __init__(
            self,
            root: str = "../data",
            batch_size: int = 32,
            num_workers: int = 4,
            image_resolution: int = 256,
            label_offset=1,
            category="cat",
            transform=None,
            stroke_length=30,
            coord_length=96
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.label_offset = label_offset
        self.transform = transform
        self.num_classes = 1
        self.category = category
        self.stroke_length = stroke_length
        self.coord_length = coord_length
        self._set_dataset()

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((20, 256)),  # 이미지를 20x256 크기로 조정
                    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (픽셀 값 [0, 1] 범위로 정규화)
                    transforms.Normalize((0.5, 0.5), (0.5, 0.5)),  # 이미지 평균을 0.5로, 표준 편차를 0.5로 정규화
                ]
            )

        self.train_ds = QuickDrawDataset(
            self.root,
            "train",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes=self.num_classes,
            stroke_length=self.stroke_length,
            coord_length=self.coord_length
        )
        self.val_ds = QuickDrawDataset(
            self.root,
            "test",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes=self.num_classes,
            stroke_length=self.stroke_length,
            coord_length=self.coord_length
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )


if __name__ == "__main__":
    data_module = QuickDrawDataModule("../data", 32, 4, 256, 1)

    # eval_dir = Path(data_module.root) / "eval"
    # eval_dir.mkdir(exist_ok=True)

    # def func(path):
    #     fn = path.name
    #     cmd = f"cp {path} {eval_dir / fn}"
    #     os.system(cmd)
    #     img = Image.open(str(eval_dir / fn))
    #     img = img.resize((256, 256))
    #     img.save(str(eval_dir / fn))
    #     print(fn)
    #
    #
    # with Pool(8) as pool:
    #     pool.map(func, data_module.val_ds.fnames)
    #
    # print(f"Constructed eval dir at {eval_dir}")
