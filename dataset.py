from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from charset import encode

IMG_HEIGHT = 112
IMG_WIDTH = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PlateDataset(Dataset):
    """
    Đọc dữ liệu từ file nhãn dạng:
        <tên_file_ảnh>\t<biển_số>

    Hoặc:
        <tên_file_ảnh> <biển_số>

    Ảnh được tìm trong cùng thư mục với file nhãn.
    """

    def __init__(self, labels_path: str | Path, train: bool) -> None:
        self.labels_path = Path(labels_path)
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {self.labels_path}")

        self.root_dir = self.labels_path.parent
        self.transform = build_transforms(train)

        self.samples: list[tuple[Path, str]] = []

        with self.labels_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Ưu tiên tab, nếu không có thì tách bằng whitespace.
                if "\t" in line:
                    parts = line.split("\t", maxsplit=1)
                else:
                    parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    raise ValueError(
                        f"{self.labels_path}:{line_num} sai định dạng, "
                        f"cần '<tên_file_ảnh>\\t<biển_số>' hoặc "
                        f"'<tên_file_ảnh> <biển_số>', nhận được: {line!r}"
                    )

                image_name, plate_text = parts
                image_name = image_name.strip()
                plate_text = plate_text.strip()

                self.samples.append(
                    (self.root_dir / "images" / image_name, plate_text)
                )

        if not self.samples:
            raise ValueError(f"File nhãn rỗng: {self.labels_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, plate_text = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        target = torch.tensor(encode(plate_text), dtype=torch.long)
        return image_tensor, target, plate_text


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Gộp batch cho CTCLoss:
        images       -> [B, 3, H, W]
        targets      -> nối liền tất cả nhãn thành 1 vector 1D
        target_lens  -> độ dài từng nhãn, dùng để CTCLoss tách lại targets
        texts        -> nhãn dạng chuỗi gốc, dùng để tính CER lúc validate
    """
    images, targets, texts = zip(*batch)

    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)

    return images, targets_concat, target_lengths, list(texts)
