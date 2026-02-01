import sys
from pathlib import Path
import numpy as np

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.replay_buffer import ReplayBuffer  # noqa: E402
from config import PATHS  # noqa: E402


# ---------------- CALIBRATION READER ----------------
class AlphaZeroCalibrationReader(CalibrationDataReader):
    def __init__(self, num_samples=512):
        self.samples = self._load_samples(num_samples)
        self.index = 0

    def _load_samples(self, n):
        print(f"[Calibration] Loading replay data from {PATHS.replay_dir}")

        replay = ReplayBuffer(save_dir=str(PATHS.replay_dir))
        states, _, _ = replay.load_all()

        total = len(states)
        print(f"[Calibration] Loaded {total} total states")

        if total == 0:
            raise RuntimeError("Replay buffer is empty — cannot calibrate")

        # Sample without replacement
        if total > n:
            indices = np.random.choice(total, n, replace=False)
            selected = [states[i] for i in indices]
        else:
            print(
                f"[Calibration] Warning: requested {n} samples, using {total}"
            )
            selected = states

        processed = []
        for s in selected:
            arr = np.asarray(s, dtype=np.float32)

            # Ensure shape = (1, 9, 10, 10)
            if arr.shape == (9, 10, 10):
                arr = arr[np.newaxis, ...]
            elif arr.shape == (1, 9, 10, 10):
                pass
            else:
                raise ValueError(
                    f"Invalid state shape {arr.shape}, expected (9,10,10)"
                )

            processed.append({"input": arr})

        print(f"[Calibration] Using {len(processed)} calibration samples")
        return processed

    def get_next(self):
        if self.index >= len(self.samples):
            return None

        sample = self.samples[self.index]
        self.index += 1
        return sample


# ---------------- QUANTIZATION ----------------
def quantize(fp32_model, int8_model):
    print("[INFO] Starting INT8 static quantization")

    reader = AlphaZeroCalibrationReader(num_samples=512)

    quantize_static(
        model_input=fp32_model,
        model_output=int8_model,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,     # REQUIRED for Vitis AI
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    print("[SUCCESS] INT8 model saved →", int8_model)


if __name__ == "__main__":
    quantize("model_fp32.onnx", "model_int8.onnx")
