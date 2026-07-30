from __future__ import annotations

"""
Bảng ký tự dùng cho OCR biển số xe Việt Nam.

Index 0 luôn là ký tự "blank" của CTC (bắt buộc theo PyTorch CTCLoss).
Các ký tự còn lại: 10 chữ số + 26 chữ cái in hoa + "-" và "." (biển số
Việt Nam thường có định dạng kiểu "29A-123.45").
"""

BLANK_IDX = 0

CHARS: list[str] = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")

# index 0 = blank, index i+1 = CHARS[i]
NUM_CLASSES = len(CHARS) + 1

_CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
_IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}


def encode(text: str) -> list[int]:
    """Chuyển chuỗi biển số thành danh sách index (không gồm blank)."""
    text = text.strip().upper()
    unknown = [ch for ch in text if ch not in _CHAR_TO_IDX]
    if unknown:
        raise ValueError(
            f"Ký tự không hợp lệ {unknown} trong nhãn '{text}'. "
            f"Ký tự cho phép: {CHARS}"
        )
    return [_CHAR_TO_IDX[ch] for ch in text]


def decode_indices(indices: list[int]) -> str:
    """Chuyển danh sách index (đã loại blank) thành chuỗi ký tự."""
    return "".join(_IDX_TO_CHAR[i] for i in indices if i in _IDX_TO_CHAR)


def ctc_greedy_decode(pred_indices: list[int]) -> str:
    """
    Giải mã CTC kiểu greedy (best-path): gộp các ký tự lặp liên tiếp
    rồi bỏ hết blank.

    pred_indices: danh sách index dự đoán tại từng time-step
                  (argmax theo chiều class), độ dài = T.
    """
    collapsed: list[int] = []
    prev = None
    for idx in pred_indices:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    collapsed = [idx for idx in collapsed if idx != BLANK_IDX]
    return decode_indices(collapsed)
