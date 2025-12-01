"""
Debug instrumentation to track which model checkpoints are loaded by each component.
Add this code to your training pipeline to verify the InferenceServer bug.
"""

import hashlib
import torch
from pathlib import Path


def get_model_fingerprint(model):
    """
    Get a fingerprint (hash) of model weights to verify which checkpoint is loaded.
    
    Args:
        model: PyTorch model or state_dict
    
    Returns:
        str: MD5 hash of first layer weights
    """
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    
    # Hash the first parameter tensor to identify model version
    first_key = list(state_dict.keys())[0]
    first_param = state_dict[first_key].cpu().numpy().tobytes()
    
    md5 = hashlib.md5(first_param).hexdigest()[:8]
    return md5


def verify_checkpoint_file(path):
    """
    Get MD5 hash of checkpoint file on disk.
    
    Args:
        path: Path to .pth file
    
    Returns:
        str: MD5 hash of file
    """
    if not Path(path).exists():
        return "FILE_NOT_FOUND"
    
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    return file_hash


def log_model_state(component_name, model=None, checkpoint_path=None):
    """
    Log which model a component is using.
    
    Args:
        component_name: Name of component (e.g., "InferenceServer", "Arena_ModelA")
        model: PyTorch model (optional)
        checkpoint_path: Path to checkpoint file (optional)
    """
    print(f"\n{'='*70}")
    print(f"[DEBUG] {component_name} - Model State")
    print(f"{'='*70}")
    
    if checkpoint_path:
        file_hash = verify_checkpoint_file(checkpoint_path)
        print(f"  Checkpoint Path: {checkpoint_path}")
        print(f"  File MD5:        {file_hash}")
    
    if model is not None:
        weights_hash = get_model_fingerprint(model)
        print(f"  Model Weights:   {weights_hash}")
    
    print(f"{'='*70}\n")


# ============================================================================
# HOW TO USE - Add these lines to your code:
# ============================================================================

"""
1. In main.py, after creating InferenceServer (line 86-87):
   
   from debug_model_loading import log_model_state
   
   inference_server = InferenceServer(trainer.net, batch_size=32)
   await inference_server.start()
   log_model_state("InferenceServer (Initial)", model=trainer.net)


2. In main.py, after training (line 119):
   
   trainer.train(states_all, policies_all, values_all, save_name=save_name)
   log_model_state("Trainer (After Training)", model=trainer.net, checkpoint_path=latest_path)
   log_model_state("InferenceServer (Before Reload)", model=inference_server.model)


3. In evaluate/evaluate.py, in Arena.__init__ (after line 29):
   
   from debug_model_loading import log_model_state
   
   self.model_B.eval()
   log_model_state("Arena Model A (new)", model=self.model_A, checkpoint_path=model_A_path)
   log_model_state("Arena Model B (old)", model=self.model_B, checkpoint_path=model_B_path)


4. In selfplay/selfplay.py, in play_one_game (after line 37):
   
   from debug_model_loading import log_model_state
   
   env = self.env_class()
   # Add this on FIRST game only:
   if not hasattr(self, '_logged_model'):
       log_model_state("SelfPlay (via InferenceServer)", 
                      model=self.inference_server.model)
       self._logged_model = True
"""

# ============================================================================
# EXPECTED OUTPUT (Before Fix):
# ============================================================================
"""
Iteration 1:
  InferenceServer (Initial):    weights=abc12345
  Trainer (After Training):     weights=def67890  ← Different!
  InferenceServer (Before):     weights=abc12345  ← STILL OLD!
  Arena Model A:                weights=def67890
  Arena Model B:                weights=abc12345
  SelfPlay:                     weights=abc12345  ← Uses old model!

Iteration 2:
  Trainer (After Training):     weights=xyz11111  ← New training
  InferenceServer:              weights=abc12345  ← NEVER CHANGED!
  SelfPlay:                     weights=abc12345  ← STUCK ON ITERATION 0!

This proves self-play is using iteration 0 model forever.
"""

# ============================================================================
# EXPECTED OUTPUT (After Fix with reload_model):
# ============================================================================
"""
Iteration 1:
  InferenceServer (Initial):    weights=abc12345
  Trainer (After Training):     weights=def67890
  InferenceServer (After Reload): weights=def67890  ← UPDATED!
  SelfPlay (Next iteration):    weights=def67890  ← Correct!

Iteration 2:
  Trainer (After Training):     weights=xyz11111
  InferenceServer (After Reload): weights=xyz11111  ← UPDATED!
  SelfPlay:                     weights=xyz11111  ← Uses latest model!
"""
