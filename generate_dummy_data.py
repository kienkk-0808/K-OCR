from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHKLMNPSTUVXYZ"  # chữ cái hay dùng trên biển số VN
PROVINCE_CODES = [f"{i:02d}" for i in range(11, 100)]


def random_plate_text(rng: random.Random) -> str:
    province = rng.choice(PROVINCE_CODES)
    letter = rng.choice(LETTERS)
    number = rng.randint(0, 99999)
    if rng.random() < 0.5:
        return f"{province}{letter}-{number:05d}"
    return f"{province}{letter}-{number // 1000:03d}.{number % 1000:02d}"


def render_plate_image(text: str, rng: random.Random) -> Image.Image:
    width, height = 224, 112
    bg_color = tuple(rng.randint(210, 255) for _ in range(3))
    fg_color = tuple(rng.randint(0, 40) for _ in range(3))

    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max((width - text_w) // 2, 0)
    y = max((height - text_h) // 2, 0)
    draw.text((x, y), text, fill=fg_color, font=font)

    return image


def generate_split(
    output_dir: Path, num_samples: int, seed: int
) -> None:
    rng = random.Random(seed)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for i in range(num_samples):
            plate_text = random_plate_text(rng)
            image = render_plate_image(plate_text, rng)

            image_name = f"plate_{i:05d}.jpg"
            image.save(images_dir / image_name, quality=90)

            f.write(f"{image_name}\t{plate_text}\n")

    print(f"Đã tạo {num_samples} ảnh giả lập tại: {output_dir}")


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    generate_split(data_dir / "train", args.train_samples, seed=args.seed)
    generate_split(data_dir / "val", args.val_samples, seed=args.seed + 1)
    print(
        "\nLưu ý: đây chỉ là dữ liệu giả lập (chữ render bằng font máy tính) "
        "để chạy thử pipeline train/tune/infer. Phải thay bằng ảnh biển số "
        "thật trước khi dùng cho sản phẩm thật."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo dữ liệu biển số giả lập (synthetic) để test nhanh toàn bộ "
            "pipeline train/tune/infer khi chưa có dữ liệu thật."
        )
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
