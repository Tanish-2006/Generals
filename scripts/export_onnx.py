import sys
import argparse
import torch
import torch.onnx
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.network import GeneralsNet  # noqa: E402
from config import NETWORK  # noqa: E402


def export_to_onnx(checkpoint_path, output_path, device="cpu"):
    print(f"Loading model from {checkpoint_path}...")

    model = GeneralsNet(
        action_dim=NETWORK.action_dim,
        channels=NETWORK.input_channels,
        num_res_blocks=NETWORK.num_res_blocks,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    dummy_input = torch.randn(
        1, NETWORK.input_channels, NETWORK.board_size, NETWORK.board_size
    ).to(device)
    
    print(f"Exporting to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    print("Export successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GeneralsNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/model_latest.pth",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/checkpoints/model_latest.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for export"
    )

    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.device)
