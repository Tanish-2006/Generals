import asyncio
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.batched_inference import InferenceServer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)  # dummy

    def forward(self, x):
        time.sleep(0.1)
        return torch.randn(x.size(0), 10003), torch.randn(x.size(0), 1)


async def heartbeat(stop_event):
    ticks = 0
    start = time.time()
    while not stop_event.is_set():
        await asyncio.sleep(0.01)  # Yield immediately
        ticks += 1
    duration = time.time() - start
    print(
        f"[Heartbeat] {ticks} ticks in {duration:.2f}s (Rate: {ticks / duration:.1f} Hz)"
    )
    return ticks / duration


async def stress_test():
    print("=" * 60)
    print("TEST: Async Inference Responsiveness")
    print("=" * 60)

    model = MockModel()
    server = InferenceServer(model, batch_size=4, timeout=0.01)
    await server.start()

    stop_event = asyncio.Event()

    # Start heartbeat
    heartbeat_task = asyncio.create_task(heartbeat(stop_event))

    # Generate load
    print("[Test] Sending 50 inference requests...")
    tasks = []
    start_req = time.time()
    for _ in range(50):
        # dummy state
        state = np.zeros((10,), dtype=np.float32)
        # Actually our InferenceServer expects (17, 10, 10) but mock model handles whatever for now?
        # InferenceServer: `states_tensor = torch.tensor(states, dtype=torch.float32)`
        # If input is (10,), batch is (N, 10). MockModel works.
        tasks.append(server.predict(state))
        # Small sleep to trickle them in or burst? Burst is better stress.

    await asyncio.gather(*tasks)
    end_req = time.time()
    print(f"[Test] Processed 50 reqs in {end_req - start_req:.2f}s")

    # Stop heartbeat
    stop_event.set()
    hz = await heartbeat_task

    await server.stop()

    if hz < 10:
        print(
            "\n[FAIL] Heartbeat dropped significantly! Code is blocking the event loop."
        )
    else:
        print("\n[PASS] Heartbeat remained active. Inference is NON-BLOCKING.")


if __name__ == "__main__":
    import numpy as np

    asyncio.run(stress_test())
