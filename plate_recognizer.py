from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Lightweight residual block for OCR.

    Spatial downsampling is controlled by stride.
    For license plate OCR, we reduce height more aggressively than width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class PlateRecognizer(nn.Module):
    """
    Vietnamese license plate recognizer.

    Expected input:
        [B, 3, 112, 224]

    Main tensor shapes:
        Input                  -> [B,   3, 112, 224]
        Stem                   -> [B,  32,  56, 112]
        Stage 1                -> [B,  64,  28,  56]
        Stage 2                -> [B, 128,  14,  56]
        Stage 3                -> [B, 256,   7,  56]
        AdaptiveAvgPool(H=1)   -> [B, 256,   1,  56]
        Sequence               -> [B,  56, 256]
        2x BiLSTM              -> [B,  56, 384]
        Classifier             -> [B,  56, num_classes]

    Notes:
        - num_classes must include the CTC blank class.
        - Output is raw logits. Apply log_softmax before CTCLoss.
        - Width is downsampled only 4x: 224 -> 112 -> 56.
          This preserves 56 sequence time steps for CTC.
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 192,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 and include the CTC blank class.")

        # 112x224 -> 56x112
        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                32,
                kernel_size=3,
                stride=(1, 1),
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 56x112 -> 28x56
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=(2, 2)),
            ResidualBlock(64, 64, stride=(1, 1)),
        )

        # 28x56 -> 14x56
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=(2, 1)),
            ResidualBlock(128, 128, stride=(1, 1)),
        )

        # 14x56 -> 7x56
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=(2, 1)),
            ResidualBlock(256, 256, stride=(1, 1)),
        )

        # Collapse only the height dimension, preserve width=56.
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        self.sequence_model = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits: [B, T, num_classes]
                    With 112x224 input, T is normally 56.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.height_pool(x)  # [B, 256, 1, W]
        x = x.squeeze(2)         # [B, 256, W]
        x = x.transpose(1, 2)    # [B, W, 256]

        x, _ = self.sequence_model(x)
        logits = self.classifier(x)
        return logits

    def log_probs_for_ctc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for PyTorch CTCLoss.

        PyTorch CTCLoss expects:
            [T, B, C]

        Returns:
            log_probs: [T, B, num_classes]
        """
        logits = self.forward(x)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.transpose(0, 1)


def build_model(
    num_classes: int,
    lstm_hidden: int = 192,
) -> PlateRecognizer:
    return PlateRecognizer(
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=2,
        dropout=0.1,
    )


if __name__ == "__main__":
    # Example:
    # 10 digits + 26 uppercase letters + 2 "-", "." + 1 CTC blank = 39 classes.
    NUM_CLASSES = 39

    model = build_model(NUM_CLASSES)
    model.eval()

    dummy_input = torch.randn(1, 3, 112, 224)

    with torch.no_grad():
        logits = model(dummy_input)
        log_probs = model.log_probs_for_ctc(dummy_input)

    print("Input shape     :", tuple(dummy_input.shape))
    print("Logits shape    :", tuple(logits.shape))
    print("CTC shape       :", tuple(log_probs.shape))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Expected:
    # Input shape  : (1, 3, 112, 224)
    # Logits shape : (1, 56, 39)
    # CTC shape    : (56, 1, 39)

