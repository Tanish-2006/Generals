import logging
import torch
from pathlib import Path
from .config import settings

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference")

# Lazy imports to avoid circular dependency issues at the top level before system path fix
try:
    from model.network import GeneralsNet
    from utils.batched_inference import InferenceServer
except ImportError:
    logger.warning(
        "Could not import model/utils during basic initialization. This is expected during build steps."
    )
    pass


class InferenceManager:
    _instance = None
    _server: "InferenceServer" = None
    _model: "GeneralsNet" = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InferenceManager()
        return cls._instance

    def load_model(self):
        """
        Loads the neural network checkpoint and initializes the inference server.
        """
        if not settings.CHECKPOINT_PATH.exists():
            logger.error(f"Checkpoint not found at: {settings.CHECKPOINT_PATH}")
            raise FileNotFoundError(
                f"Checkpoint not found at {settings.CHECKPOINT_PATH}"
            )

        logger.info(f"Loading model from {settings.CHECKPOINT_PATH}...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        try:
            if settings.INFERENCE_MODE == "onnx":
                logger.info(
                    f"Using ONNX Inference Mode. Provider: {settings.ONNX_PROVIDER}"
                )

                if not settings.ONNX_PATH.exists():
                    logger.warning(
                        f"ONNX model not found at {settings.ONNX_PATH}. Attempting fallback to PyTorch or export."
                    )
                    # For now, just raise, or implementation could trigger export here.
                    raise FileNotFoundError(
                        f"ONNX model not found at {settings.ONNX_PATH}"
                    )

                # Initialize ONNX Server
                from utils.inference_onnx import ONNXInferenceServer

                self._server = ONNXInferenceServer(
                    model_path=settings.ONNX_PATH,
                    batch_size=settings.BATCH_SIZE,
                    timeout=settings.TIMEOUT,
                    provider=settings.ONNX_PROVIDER,
                )
                logger.info("ONNXInferenceServer initialized successfully.")

            else:
                # PyTorch Mode
                self._model = GeneralsNet(
                    action_dim=settings.MODEL_ACTION_DIM,
                    channels=settings.MODEL_CHANNELS,
                    num_res_blocks=settings.MODEL_RES_BLOCKS,
                )

                # Use weights_only=False carefully
                checkpoint = torch.load(
                    settings.CHECKPOINT_PATH, map_location=device, weights_only=False
                )

                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self._model.load_state_dict(checkpoint)
                else:
                    logger.warning(
                        "Checkpoint appears to be a raw model object, attempting to load parameters..."
                    )
                    self._model.load_state_dict(checkpoint)

                self._model.to(device)
                self._model.eval()

                self._server = InferenceServer(
                    self._model,
                    batch_size=settings.BATCH_SIZE,
                    timeout=settings.TIMEOUT,
                )
                logger.info("InferenceServer (PyTorch) initialized successfully.")

        except Exception as e:
            logger.critical(f"Failed to initialize model: {e}", exc_info=True)
            raise RuntimeError(f"Model initialization failed: {e}")

    async def start(self):
        if self._server:
            logger.info("Starting inference server worker...")
            await self._server.start()
        else:
            logger.error("Attempted to start InferenceServer before loading model.")

    async def stop(self):
        if self._server:
            logger.info("Stopping inference server worker...")
            await self._server.stop()

    @property
    def server(self) -> "InferenceServer":
        if self._server is None:
            raise RuntimeError(
                "InferenceServer not initialized. Call load_model() first."
            )
        return self._server


inference_manager = InferenceManager.get_instance()
