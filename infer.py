from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch
from PIL import Image

from charset import ctc_greedy_decode
from dataset import build_transforms
from plate_recognizer import PlateRecognizer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_model(checkpoint_path: str, device: torch.device) -> PlateRecognizer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = PlateRecognizer(
        num_classes=config["num_classes"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: PlateRecognizer, image_path: Path, device: torch.device
) -> str:
    transform = build_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    return ctc_greedy_decode(pred_indices)


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Không tìm thấy: {input_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Sử dụng device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Đã load checkpoint: {args.checkpoint}")

    image_paths = collect_image_paths(Path(args.input))
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong: {args.input}")
        return

    results = {}
    for image_path in image_paths:
        plate_text = predict_image(model, image_path, device)
        results[str(image_path)] = plate_text
        print(f"{image_path.name:<40} -> {plate_text}")

    if args.output_json:
        output_path = Path(args.output_json)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu kết quả JSON tại: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy thử model OCR biển số trên 1 ảnh hoặc 1 thư mục ảnh."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Đường dẫn checkpoint (.pt) đã train",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn 1 ảnh biển số hoặc 1 thư mục chứa nhiều ảnh",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="(Tùy chọn) lưu kết quả dự đoán ra file JSON",
    )
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
