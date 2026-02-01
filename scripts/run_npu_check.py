import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

assert "RYZEN_AI_INSTALLATION_PATH" in os.environ, "RYZEN_AI_INSTALLATION_PATH not set"

MODEL_PATH = "model_int8.onnx"
CACHE_DIR = Path("./npu_cache").resolve()
CACHE_DIR.mkdir(exist_ok=True)

providers = ["VitisAIExecutionProvider"]
provider_options = [{
    "cacheDir": str(CACHE_DIR),
    "cacheKey": "alphazero_npu",
    # Phoenix uses auto xclbin via runtime
}]

print("[INFO] Creating ORT session with VitisAIExecutionProvider")

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=providers,
    provider_options=provider_options,
)

print("[INFO] Session providers:", sess.get_providers())

# Dummy AlphaZero board
x = np.random.rand(1, 9, 10, 10).astype(np.float32)

print("[INFO] Running inference…")
policy, value = sess.run(None, {"input": x})

print("[SUCCESS] NPU inference complete")
print("Policy shape:", policy.shape)
print("Value:", value[0][0])
