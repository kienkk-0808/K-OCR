from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch

from dataset import build_transforms
from infer import collect_image_paths, load_model

DUMMY_INPUT_SHAPE = (1, 3, 112, 224)


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    print(f"\n[ONNX] Export sang {output_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
    )
    print(f"[ONNX] Đã lưu: {output_path}")


def load_calibration_images(
    data_dir: str, num_samples: int
) -> list[torch.Tensor]:
    """Lấy ảnh thật từ data/train để calibrate INT8 (thay vì random)."""
    images_dir = Path(data_dir) / "train" / "images"
    image_paths = collect_image_paths(images_dir)[:num_samples]
    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh để calibrate INT8 tại: {images_dir}"
        )

    transform = build_transforms(train=False)
    tensors = []
    for path in image_paths:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        tensors.append(transform(image).unsqueeze(0))

    print(f"[OpenVINO INT8] Dùng {len(tensors)} ảnh thật từ {images_dir} để calibrate.")
    return tensors


def export_openvino(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_dir: Path,
    fp16: bool,
    int8: bool,
    data_dir: str,
    int8_calibration_samples: int,
) -> None:
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("Thiếu OpenVINO. Cài bằng: pip install openvino") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[OpenVINO] Convert model PyTorch -> OpenVINO IR...")
    ov_model = ov.convert_model(model, example_input=dummy_input)

    fp32_path = output_dir / "plate_recognizer_fp32.xml"
    ov.save_model(ov_model, fp32_path, compress_to_fp16=False)
    print(f"[OpenVINO] Đã lưu FP32: {fp32_path}")

    if fp16:
        fp16_path = output_dir / "plate_recognizer_fp16.xml"
        ov.save_model(ov_model, fp16_path, compress_to_fp16=True)
        print(f"[OpenVINO] Đã lưu FP16: {fp16_path}")

    if int8:
        try:
            import nncf
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu NNCF để quantize INT8. Cài bằng: pip install nncf"
            ) from exc

        calibration_tensors = load_calibration_images(
            data_dir, int8_calibration_samples
        )
        calibration_data = [t.numpy() for t in calibration_tensors]
        calibration_dataset = nncf.Dataset(calibration_data)

        fp32_ov_model = ov.Core().read_model(fp32_path)

        print(
            f"[OpenVINO INT8] Quantizing bằng {len(calibration_data)} "
            f"ảnh thật từ data/train..."
        )
        quantized_model = nncf.quantize(
            fp32_ov_model,
            calibration_dataset,
            subset_size=len(calibration_data),
        )

        int8_path = output_dir / "plate_recognizer_int8.xml"
        ov.save_model(quantized_model, int8_path, compress_to_fp16=False)
        print(f"[OpenVINO INT8] Đã lưu: {int8_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model = load_model(args.checkpoint, device)
    model.eval()
    print(f"Đã load checkpoint: {args.checkpoint}")

    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, dtype=torch.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("onnx", "both"):
        export_onnx(
            model,
            dummy_input,
            output_dir / "plate_recognizer.onnx",
            opset=args.opset,
        )

    if args.format in ("openvino", "both"):
        export_openvino(
            model,
            dummy_input,
            output_dir,
            fp16=args.fp16,
            int8=args.int8,
            data_dir=args.data_dir,
            int8_calibration_samples=args.int8_calibration_samples,
        )

    print(f"\nHoàn tất export. Kết quả tại: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model OCR biển số sang ONNX và/hoặc OpenVINO IR."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Checkpoint (.pt) đã train",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "openvino", "both"],
        default="both",
        help="Định dạng xuất ra",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Thư mục lưu model xuất ra",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Xuất thêm bản OpenVINO FP16 (mặc định bật)",
    )
    parser.add_argument(
        "--no-fp16", dest="fp16", action="store_false", help="Bỏ qua bản FP16"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "Xuất thêm bản OpenVINO INT8, quantize bằng ảnh thật lấy từ "
            "data/train (cần cài nncf)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Thư mục data, dùng để lấy ảnh calibrate khi bật --int8",
    )
    parser.add_argument(
        "--int8-calibration-samples",
        type=int,
        default=100,
        help="Số ảnh thật dùng để calibrate INT8",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
