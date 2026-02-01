import sys
import argparse
import torch
import torch.onnx
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.network import GeneralsNet  # noqa: E402
from config import NETWORK  # noqa: E402


def export_to_onnx(checkpoint_path, output_path, device="cpu"):
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model = GeneralsNet(
        action_dim=NETWORK.action_dim,
        channels=NETWORK.hidden_channels,
        num_res_blocks=NETWORK.num_res_blocks,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    dummy_input = torch.randn(
        1,
        NETWORK.input_channels,
        NETWORK.board_size,
        NETWORK.board_size,
        device=device,
    )

    print(f"[INFO] Exporting ONNX → {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=14,                 # Phoenix-safe
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
    )

    print("[SUCCESS] ONNX export complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model_fp32.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.device)
