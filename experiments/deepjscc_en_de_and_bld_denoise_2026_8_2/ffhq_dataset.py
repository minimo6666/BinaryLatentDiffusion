"""Dependency-light FFHQ split loader matching the project's 256x256 lists."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FFHQListDataset(Dataset):
    def __init__(self, image_root: Path, list_path: Path):
        self.image_root = Path(image_root)
        with Path(list_path).open("r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        self.paths = [self.image_root / name for name in names]
        if not self.paths:
            raise RuntimeError(f"Empty FFHQ split list: {list_path}")
        if not self.paths[0].is_file():
            raise FileNotFoundError(f"FFHQ image not found: {self.paths[0]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (256, 256):
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.uint8).copy()
        return {"image": array, "file_path_": str(path)}
