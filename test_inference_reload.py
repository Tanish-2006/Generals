"""
Quick sanity test to verify InferenceServer reload functionality.
Run this before starting full training to confirm the fix works.
"""

import torch
import numpy as np
from pathlib import Path
from utils.batched_inference import InferenceServer
from model.network import GeneralsNet

def test_inference_server_reload():
    """Test that InferenceServer can reload model weights correctly."""
    
    print("="*70)
    print("TESTING INFERENCESERVER RELOAD FUNCTIONALITY")
    print("="*70)
    
    # Create temp checkpoint directory
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model and save initial weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test] Using device: {device}")
    
    model1 = GeneralsNet().to(device)
    checkpoint1 = checkpoint_dir / "test_checkpoint_1.pth"
    torch.save(model1.state_dict(), checkpoint1)
    print(f"[Test] Saved initial model to {checkpoint1}")
    
    # Get fingerprint of first model
    param_name = list(model1.state_dict().keys())[0]
    weights1 = model1.state_dict()[param_name].cpu().numpy()
    fingerprint1 = np.mean(weights1)
    print(f"[Test] Model 1 fingerprint ({param_name}): {fingerprint1:.6f}")
    
    # Create modified model with different weights
    model2 = GeneralsNet().to(device)
    # Modify weights slightly
    with torch.no_grad():
        for param in model2.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    checkpoint2 = checkpoint_dir / "test_checkpoint_2.pth"
    torch.save(model2.state_dict(), checkpoint2)
    print(f"[Test] Saved modified model to {checkpoint2}")
    
    weights2 = model2.state_dict()[param_name].cpu().numpy()
    fingerprint2 = np.mean(weights2)
    print(f"[Test] Model 2 fingerprint ({param_name}): {fingerprint2:.6f}")
    
    # Verify models are different
    if abs(fingerprint1 - fingerprint2) < 1e-6:
        print("\n❌ ERROR: Model weights are identical! Test setup failed.")
        return False
    print(f"[Test] ✓ Models are different (diff: {abs(fingerprint1 - fingerprint2):.6f})")
    
    # Create InferenceServer with model1
    print("\n" + "="*70)
    print("TESTING RELOAD MECHANISM")
    print("="*70)
    
    inference_server = InferenceServer(model1, batch_size=4)
    
    # Check initial weights
    current_weights = inference_server.model.state_dict()[param_name].cpu().numpy()
    current_fingerprint = np.mean(current_weights)
    print(f"\n[Test] InferenceServer initial fingerprint: {current_fingerprint:.6f}")
    
    if abs(current_fingerprint - fingerprint1) > 1e-6:
        print("❌ ERROR: InferenceServer doesn't match model1!")
        return False
    print("[Test] ✓ InferenceServer correctly loaded model1")
    
    # Reload with model2 checkpoint
    print(f"\n[Test] Calling reload_model({checkpoint2})...")
    inference_server.reload_model(str(checkpoint2))
    
    # Check reloaded weights
    reloaded_weights = inference_server.model.state_dict()[param_name].cpu().numpy()
    reloaded_fingerprint = np.mean(reloaded_weights)
    print(f"[Test] InferenceServer after reload fingerprint: {reloaded_fingerprint:.6f}")
    
    # Verify reload worked
    if abs(reloaded_fingerprint - fingerprint2) > 1e-6:
        print(f"\n❌ CRITICAL ERROR: Reload failed!")
        print(f"   Expected: {fingerprint2:.6f}")
        print(f"   Got:      {reloaded_fingerprint:.6f}")
        print(f"   Diff:     {abs(reloaded_fingerprint - fingerprint2):.6f}")
        return False
    
    print("\n" + "="*70)
    print("✅ SUCCESS: InferenceServer reload works correctly!")
    print("="*70)
    print(f"   Initial:  {fingerprint1:.6f}")
    print(f"   Reloaded: {reloaded_fingerprint:.6f}")
    print(f"   Expected: {fingerprint2:.6f}")
    print(f"   Match: ✓ (diff: {abs(reloaded_fingerprint - fingerprint2):.9f})")
    
    # Cleanup
    checkpoint1.unlink()
    checkpoint2.unlink()
    print(f"\n[Test] Cleaned up temporary checkpoints")
    
    return True


if __name__ == "__main__":
    success = test_inference_server_reload()
    
    if success:
        print("\n" + "🎉"*35)
        print("ALL TESTS PASSED - InferenceServer reload is working!")
        print("You can now start training with confidence.")
        print("🎉"*35)
        exit(0)
    else:
        print("\n" + "❌"*35)
        print("TEST FAILED - InferenceServer reload is NOT working!")
        print("DO NOT start training until this is fixed.")
        print("❌"*35)
        exit(1)
