import asyncio
import torch
import numpy as np
import time

class InferenceServer:
    def __init__(self, model, batch_size=32, timeout=0.05):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.device = next(model.parameters()).device
        self._worker_task = None

    async def start(self):
        self.stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker())
        print(f"[InferenceServer] Started on {self.device} with batch_size={self.batch_size}")


    async def stop(self):
        self.stop_event.set()
        if self._worker_task:
            await self._worker_task

    def reload_model(self, checkpoint_path):
        """
        Reload model weights from a checkpoint file.
        CRITICAL: Call this after each training iteration to update InferenceServer.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[InferenceServer] ✓ Reloaded weights from {checkpoint_path}")

    async def predict(self, state):
        future = asyncio.Future()
        await self.queue.put((state, future))
        return await future

    async def _worker(self):
        while not self.stop_event.is_set():
            batch = []
            futures = []
            
            # Collect batch
            try:
                # Wait for first item
                item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                batch.append(item[0])
                futures.append(item[1])
                
                # Collect remaining items up to batch_size or timeout
                start_time = time.time()
                while len(batch) < self.batch_size:
                    remaining_time = self.timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                        batch.append(item[0])
                        futures.append(item[1])
                    except asyncio.TimeoutError:
                        break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not batch:
                continue

            try:
                states = np.array(batch)
                states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    policy_logits, values = self.model(states_tensor)
                
                policy_logits = policy_logits.cpu().numpy()
                values = values.cpu().numpy().flatten()
                
                for i, future in enumerate(futures):
                    if not future.done():
                        future.set_result((policy_logits[i], values[i]))
                        
            except Exception as e:
                print(f"[InferenceServer] Error processing batch: {e}")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
