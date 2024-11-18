import json
from itertools import chain
from tqdm import tqdm
from multiprocessing.pool import Pool
from pathlib import Path
from utils import *

import torch
import torchvision.transforms as transforms
from PIL import Image


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


def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    """
    x: [B,C,H,W]
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        single_image = True

    x = (x * 0.5 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (x * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    if single_image:
        return images[0]
    return images


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
        self, root: str, split: str, transform=None, category="cat", stroke_type="full", label_offset=1, num_classes=1
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_offset = label_offset
        self.category = category
        self.stroke_type = stroke_type
        self.num_classes = num_classes

        fnames, labels = [], []

        category_dir = os.path.join(root, category, split)
        cat_fnames = listdir(category_dir)
        cat_fnames = sorted(cat_fnames)

        fnames += cat_fnames
        labels += [label_offset] * len(cat_fnames)  # label 0 is for null class.

        self.fnames = fnames
        self.labels = labels

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx]).convert("RGB")
        label = self.labels[idx]
        assert label >= self.label_offset
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class QuickDrawPointDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, split: str, transform=None, category="cat", stroke_type="full", label_offset=1, num_classes=1
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.label_offset = label_offset
        self.category = category
        self.stroke_type = stroke_type
        self.num_classes = num_classes

        tensors, labels = [], []

        category_dir = os.path.join(root, category + ".ndjson")
        with open(category_dir, "r") as f:
            data = ndjson.load(f)

        indices_path = f"../sketch_data/{category}/train_test_indices.json"
        with open(indices_path, "r") as f:
            indices = json.load(f)

        for i in tqdm(indices[split]):
            item = data[i]
            tensors.append(strokes_to_tensor(item['drawing'], 64, 64, 30))

        labels += [label_offset] * len(tensors)  # label 0 is for null class.
        print(f"Sample shape: {tensors[0].shape}")
        self.tensors = tensors
        self.labels = labels

    def __getitem__(self, idx):
        img = self.tensors[idx]
        label = self.labels[idx]
        assert label >= self.label_offset
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class QuickDrawDataModule(object):
    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_resolution: int = 64,
        label_offset=1,
        category="cat",
        transform=None
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.label_offset = label_offset
        self.transform = transform
        self.num_classes = 1
        self.category = category
        self._set_dataset()

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.train_ds = QuickDrawDataset(
            self.root,
            "train",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes
        )
        self.val_ds = QuickDrawDataset(
            self.root,
            "test",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes
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

class QuickDrawPointDataModule(object):
    def __init__(
        self,
        root: str = "./images",
        batch_size: int = 32,
        num_workers: int = 4,
        image_resolution: int = 64,
        label_offset=1,
        category="cat",
        transform=None
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.label_offset = label_offset
        self.transform = transform
        self.num_classes = 1
        self.category = category
        self._set_dataset()

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5] * 30, std=[0.5] * 30),
                ]
            )
        self.train_ds = QuickDrawPointDataset(
            self.root,
            "train",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes
        )
        self.val_ds = QuickDrawPointDataset(
            self.root,
            "test",
            self.transform,
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes
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
    data_module = QuickDrawPointDataModule("../data", 1, 4, 256, 1)

    eval_dir = Path(data_module.root) / "eval"
    eval_dir.mkdir(exist_ok=True)
    def func(path):
        fn = path.name
        cmd = f"cp {path} {eval_dir / fn}"
        os.system(cmd)
        img = Image.open(str(eval_dir / fn))
        img = img.resize((256, 256))
        img.save(str(eval_dir / fn))
        print(fn)

    with Pool(8) as pool:
        pool.map(func, data_module.val_ds.fnames)

    print(f"Constructed eval dir at {eval_dir}")
