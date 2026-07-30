from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch

from train import run_training


def count_train_samples(data_dir: str) -> int:
    labels_path = Path(data_dir) / "train" / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def auto_finetune_hparams(num_train_samples: int) -> dict:
    """
    Đề xuất batch_size / epochs / lr mặc định khi fine-tune tiếp từ một
    checkpoint đã train, dựa trên số lượng ảnh train.

    So với train từ đầu (auto_hparams() trong train.py), fine-tune xuất
    phát từ weight đã tốt sẵn nên cần lr nhỏ hơn hẳn (tránh phá vỡ những
    gì model đã học) và số epoch ít hơn (hội tụ nhanh hơn train từ đầu).
    """
    if num_train_samples < 500:
        return {"batch_size": 8, "epochs": 60, "lr": 1e-4}
    if num_train_samples < 2_000:
        return {"batch_size": 16, "epochs": 40, "lr": 1.5e-4}
    if num_train_samples < 10_000:
        return {"batch_size": 32, "epochs": 25, "lr": 2e-4}
    if num_train_samples < 50_000:
        return {"batch_size": 64, "epochs": 15, "lr": 3e-4}
    return {"batch_size": 128, "epochs": 10, "lr": 5e-4}


def finetune(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    print(
        f"Load kiến trúc từ checkpoint '{checkpoint_path}': "
        f"lstm_hidden={config['lstm_hidden']} "
        f"lstm_layers={config['lstm_layers']} "
        f"dropout={config['dropout']} "
        f"(epoch cũ={checkpoint.get('epoch', '?')}, "
        f"val_cer cũ={checkpoint.get('val_cer', float('nan')):.4f})"
    )

    num_train_samples = count_train_samples(args.data_dir)
    defaults = auto_finetune_hparams(num_train_samples)

    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    lr = args.lr if args.lr is not None else defaults["lr"]

    print(
        f"Số ảnh train: {num_train_samples} -> "
        f"batch_size={batch_size} epochs={epochs} lr={lr:.2e} "
        f"(tự động nếu không truyền tay; lr thấp hơn train từ đầu vì đây "
        f"là train tiếp/fine-tune)"
    )

    run_training(
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=args.device,
        resume=str(checkpoint_path),
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune / train tiếp model OCR biển số từ một checkpoint đã "
            "train sẵn (ví dụ checkpoints/best_model.pt), thay vì train lại "
            "từ đầu. Kiến trúc (lstm_hidden/lstm_layers/dropout) được lấy "
            "tự động từ checkpoint, không cần truyền lại."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint đã train (ví dụ checkpoints/best_model.pt)",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Mặc định: tự động theo số lượng ảnh train (xem auto_finetune_hparams())",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mặc định: tự động theo số lượng ảnh train (xem auto_finetune_hparams())",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Mặc định: tự động, thấp hơn hẳn so với train từ đầu "
            "(xem auto_finetune_hparams())"
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints_finetuned",
        help="Nơi lưu best_model.pt/last_model.pt sau khi fine-tune",
    )
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
