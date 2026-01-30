import asyncio
import torch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor


class InferenceServer:
    def __init__(self, model, batch_size=32, timeout=0.01):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.device = next(model.parameters()).device
        self._worker_task = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def start(self):
        self.stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker())
        print(
            f"[InferenceServer] Started on {self.device} with batch_size={self.batch_size}"
        )

    async def stop(self):
        self.stop_event.set()
        if self._worker_task:
            await self._worker_task
        self.executor.shutdown(wait=True)

    def reload_model(self, checkpoint_path):
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[InferenceServer] Reloaded weights from {checkpoint_path}")

    async def predict(self, state):
        future = asyncio.Future()
        await self.queue.put((state, future))
        return await future

    def _run_inference(self, batch):
        try:
            states = np.array(batch)
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                policy_logits, values = self.model(states_tensor)

            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy().flatten()

            return policy_logits, values
        except Exception as e:
            raise e

    async def _worker(self):
        loop = asyncio.get_running_loop()

        while not self.stop_event.is_set():
            batch = []
            futures = []

            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                batch.append(item[0])
                futures.append(item[1])

                start_time = time.time()
                while len(batch) < self.batch_size:
                    remaining_time = self.timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=remaining_time
                        )
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

            inference_future = loop.run_in_executor(
                self.executor, self._run_inference, batch
            )
            asyncio.create_task(self._distribute_results(inference_future, futures))

    async def _distribute_results(self, inference_future, item_futures):
        try:
            policy_logits, values = await inference_future

            for i, future in enumerate(item_futures):
                if not future.done():
                    future.set_result((policy_logits[i], values[i]))

        except Exception as e:
            print(f"[InferenceServer] Error processing batch: {e}")
            for future in item_futures:
                if not future.done():
                    future.set_exception(e)
