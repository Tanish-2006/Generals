import asyncio
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import logging

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger("inference.onnx")


class ONNXInferenceServer:
    def __init__(self, model_path, batch_size=32, timeout=0.01, provider="auto"):
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Please install it to use ONNX inference."
            )

        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.timeout = timeout
        self.provider = provider
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self._worker_task = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.session = None

        self.input_name = None
        self.output_names = None

    def _init_session(self):
        available_providers = ort.get_available_providers()
        print(f"[ONNX] Available providers: {available_providers}")

        providers = []

        if self.provider == "cuda":
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            else:
                logger.warning("[ONNX] CUDA provider requested but not available.")

        elif self.provider == "dml":
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")
            else:
                logger.warning("[ONNX] DirectML provider requested but not available.")

        elif self.provider == "auto":
            # Prioritize CUDA, then DML, then CPU
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")

        # Always fallback to CPU
        providers.append("CPUExecutionProvider")

        print(f"[ONNX] Using providers: {providers}")
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    async def start(self):
        self.stop_event.clear()

        # Initialize session in executor to avoid blocking main loop during load
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._init_session)

        self._worker_task = asyncio.create_task(self._worker())
        print(f"[ONNXInferenceServer] Started with batch_size={self.batch_size}")

    async def stop(self):
        self.stop_event.set()
        if self._worker_task:
            await self._worker_task
        self.executor.shutdown(wait=True)

    async def predict(self, state):
        future = asyncio.Future()
        await self.queue.put((state, future))
        return await future

    def _run_inference(self, batch):
        try:
            # Prepare input
            states = np.array(batch, dtype=np.float32)

            # Run inference
            inputs = {self.input_name: states}
            outputs = self.session.run(self.output_names, inputs)

            policy_logits = outputs[0]
            values = outputs[1].flatten()

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
            print(f"[ONNXInferenceServer] Error processing batch: {e}")
            for future in item_futures:
                if not future.done():
                    future.set_exception(e)
