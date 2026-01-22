import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from utils.batched_inference import InferenceServer
from model.network import GeneralsNet


def test_inference_server_reload():
    print("=" * 70)
    print("TESTING INFERENCESERVER RELOAD FUNCTIONALITY")
    print("=" * 70)
    
    checkpoint_dir = Path(__file__).parent.parent / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test] Using device: {device}")
    
    model1 = GeneralsNet().to(device)
    checkpoint1 = checkpoint_dir / "test_checkpoint_1.pth"
    torch.save(model1.state_dict(), checkpoint1)
    print(f"[Test] Saved initial model to {checkpoint1}")
    
    param_name = list(model1.state_dict().keys())[0]
    weights1 = model1.state_dict()[param_name].cpu().numpy()
    fingerprint1 = np.mean(weights1)
    print(f"[Test] Model 1 fingerprint ({param_name}): {fingerprint1:.6f}")
    
    model2 = GeneralsNet().to(device)
    with torch.no_grad():
        for param in model2.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    checkpoint2 = checkpoint_dir / "test_checkpoint_2.pth"
    torch.save(model2.state_dict(), checkpoint2)
    print(f"[Test] Saved modified model to {checkpoint2}")
    
    weights2 = model2.state_dict()[param_name].cpu().numpy()
    fingerprint2 = np.mean(weights2)
    print(f"[Test] Model 2 fingerprint ({param_name}): {fingerprint2:.6f}")
    
    assert abs(fingerprint1 - fingerprint2) >= 1e-6, "Model weights are identical! Test setup failed."
    print(f"[Test] Models are different (diff: {abs(fingerprint1 - fingerprint2):.6f})")
    
    print("\n" + "=" * 70)
    print("TESTING RELOAD MECHANISM")
    print("=" * 70)
    
    inference_server = InferenceServer(model1, batch_size=4)
    
    current_weights = inference_server.model.state_dict()[param_name].cpu().numpy()
    current_fingerprint = np.mean(current_weights)
    print(f"\n[Test] InferenceServer initial fingerprint: {current_fingerprint:.6f}")
    
    assert abs(current_fingerprint - fingerprint1) <= 1e-6, "InferenceServer doesn't match model1!"
    print("[Test] InferenceServer correctly loaded model1")
    
    print(f"\n[Test] Calling reload_model({checkpoint2})...")
    inference_server.reload_model(str(checkpoint2))
    
    reloaded_weights = inference_server.model.state_dict()[param_name].cpu().numpy()
    reloaded_fingerprint = np.mean(reloaded_weights)
    print(f"[Test] InferenceServer after reload fingerprint: {reloaded_fingerprint:.6f}")
    
    assert abs(reloaded_fingerprint - fingerprint2) <= 1e-6, f"Reload failed! Expected: {fingerprint2:.6f}, Got: {reloaded_fingerprint:.6f}"
    
    print("\n" + "=" * 70)
    print("SUCCESS: InferenceServer reload works correctly!")
    print("=" * 70)
    print(f"   Initial:  {fingerprint1:.6f}")
    print(f"   Reloaded: {reloaded_fingerprint:.6f}")
    print(f"   Expected: {fingerprint2:.6f}")
    print(f"   Match: (diff: {abs(reloaded_fingerprint - fingerprint2):.9f})")
    
    checkpoint1.unlink()
    checkpoint2.unlink()
    print(f"\n[Test] Cleaned up temporary checkpoints")


if __name__ == "__main__":
    test_inference_server_reload()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - InferenceServer reload is working!")
    print("=" * 70)
