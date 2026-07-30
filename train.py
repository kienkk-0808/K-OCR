from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch
from torch.utils.data import DataLoader

from charset import NUM_CLASSES, ctc_greedy_decode
from dataset import PlateDataset, collate_fn
from plate_recognizer import PlateRecognizer


def levenshtein(a: str, b: str) -> int:
    """Khoảng cách chỉnh sửa (edit distance), dùng để tính CER."""
    if len(a) < len(b):
        a, b = b, a
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row
    return prev_row[-1]


def character_error_rate(preds: list[str], targets: list[str]) -> float:
    total_chars = 0
    total_errors = 0
    for pred, target in zip(preds, targets):
        total_errors += levenshtein(pred, target)
        total_chars += max(len(target), 1)
    return total_errors / max(total_chars, 1)


def auto_hparams(num_train_samples: int) -> dict:
    """
    Đề xuất batch_size / epochs / lr mặc định dựa trên số lượng ảnh train.

    Ý tưởng: data ít thì cần train nhiều epoch hơn (dataset nhỏ hội tụ
    chậm hơn tính theo epoch) nhưng batch_size nhỏ để còn đủ số bước
    cập nhật mỗi epoch; data nhiều thì batch_size lớn hơn (tận dụng
    thông lượng), epoch ít hơn (mỗi epoch đã có nhiều bước cập nhật) và
    lr cao hơn một chút theo kinh nghiệm "linear scaling rule".
    """
    if num_train_samples < 500:
        return {"batch_size": 8, "epochs": 150, "lr": 5e-4}
    if num_train_samples < 2_000:
        return {"batch_size": 16, "epochs": 100, "lr": 7e-4}
    if num_train_samples < 10_000:
        return {"batch_size": 32, "epochs": 60, "lr": 1e-3}
    if num_train_samples < 50_000:
        return {"batch_size": 64, "epochs": 40, "lr": 1.5e-3}
    return {"batch_size": 128, "epochs": 25, "lr": 2e-3}


def build_model(
    lstm_hidden: int,
    lstm_layers: int,
    dropout: float,
) -> PlateRecognizer:
    return PlateRecognizer(
        num_classes=NUM_CLASSES,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )


def run_epoch(
    model: PlateRecognizer,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0
    all_preds: list[str] = []
    all_targets: list[str] = []

    for images, targets, target_lengths, texts in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.set_grad_enabled(is_train):
            log_probs = model.log_probs_for_ctc(images)  # [T, B, C]
            time_steps, batch_size, _ = log_probs.shape
            input_lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            pred_indices = log_probs.detach().argmax(dim=-1).transpose(0, 1)  # [B, T]
            for row in pred_indices.cpu().tolist():
                all_preds.append(ctc_greedy_decode(row))
            all_targets.extend(texts)

    return {
        "loss": total_loss / max(num_batches, 1),
        "cer": character_error_rate(all_preds, all_targets),
    }


def run_training(
    data_dir: str,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    weight_decay: float,
    lstm_hidden: int,
    lstm_layers: int,
    dropout: float,
    num_workers: int,
    output_dir: str,
    device: str = "",
    resume: str = "",
    verbose: bool = True,
) -> float:
    """
    Chạy toàn bộ vòng lặp train + validate, lưu checkpoint, trả về CER
    tốt nhất trên tập val. Dùng chung cho cả train.py (CLI) và
    tune.py (chạy nhiều cấu hình để tìm siêu tham số tốt nhất).

    epochs/batch_size/lr = None nghĩa là "tự động chọn theo số lượng ảnh
    train" (xem auto_hparams()).
    """
    torch_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print(f"Sử dụng device: {torch_device}")

    data_dir_path = Path(data_dir)
    train_labels = data_dir_path / "train" / "labels.txt"
    val_labels = data_dir_path / "val" / "labels.txt"

    train_dataset = PlateDataset(train_labels, train=True)
    val_dataset = PlateDataset(val_labels, train=False)
    if verbose:
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    defaults = auto_hparams(len(train_dataset))
    if batch_size is None:
        batch_size = defaults["batch_size"]
    if epochs is None:
        epochs = defaults["epochs"]
    if lr is None:
        lr = defaults["lr"]
    if batch_size > len(train_dataset):
        batch_size = max(len(train_dataset), 1)
    if verbose:
        print(
            f"Siêu tham số dùng cho lần train này: "
            f"batch_size={batch_size} epochs={epochs} lr={lr:.2e} "
            f"(tự động theo {len(train_dataset)} ảnh train nếu không truyền tay)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(lstm_hidden, lstm_layers, dropout).to(torch_device)

    if resume:
        state = torch.load(resume, map_location=torch_device)
        model.load_state_dict(state["model_state_dict"])
        if verbose:
            print(f"Đã load checkpoint: {resume}")

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = {
        "num_classes": NUM_CLASSES,
        "lstm_hidden": lstm_hidden,
        "lstm_layers": lstm_layers,
        "dropout": dropout,
    }
    with (output_dir_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_cer = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_metrics = run_epoch(model, train_loader, torch_device, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, torch_device, criterion, None)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - start
        lr_now = optimizer.param_groups[0]["lr"]

        if verbose:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_cer={val_metrics['cer']:.4f} "
                f"lr={lr_now:.2e} "
                f"({elapsed:.1f}s)"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_cer": val_metrics["cer"],
                "lr": lr_now,
            }
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_cer": val_metrics["cer"],
        }
        torch.save(checkpoint, output_dir_path / "last_model.pt")

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            torch.save(checkpoint, output_dir_path / "best_model.pt")
            if verbose:
                print(f"  -> CER tốt nhất mới: {best_cer:.4f}, đã lưu best_model.pt")

    with (output_dir_path / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nHoàn tất train. Best val CER = {best_cer:.4f}")
        print(f"Checkpoint tại: {output_dir_path / 'best_model.pt'}")

    return best_cer


def train(args: argparse.Namespace) -> None:
    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model OCR biển số xe.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Thư mục gốc chứa train/labels.txt và val/labels.txt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Mặc định: tự động chọn theo số lượng ảnh train (xem auto_hparams())",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mặc định: tự động chọn theo số lượng ảnh train (xem auto_hparams())",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Mặc định: tự động chọn theo số lượng ảnh train (xem auto_hparams())",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lstm-hidden", type=int, default=192)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--resume", type=str, default="", help="Đường dẫn checkpoint để train tiếp"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
