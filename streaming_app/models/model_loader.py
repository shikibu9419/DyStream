"""
Singleton model loader for DyStream streaming inference.
Handles loading, optimization, and caching of all required models.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ModelLoader:
    """Singleton class to load and cache all models."""

    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelLoader._models_loaded:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.models = {}
            self.config = None
            ModelLoader._models_loaded = False

    def load_all_models(self, config: Dict[str, Any]):
        """Load all required models with optimizations."""
        if ModelLoader._models_loaded:
            logger.info("Models already loaded, skipping...")
            return self.models

        self.config = config
        logger.info(f"Loading models on device: {self.device}")

        try:
            # Load DyStream motion generation model (includes Wav2Vec2 encoders)
            logger.info("Loading DyStream motion generation model...")
            self.models['dystream'] = self._load_dystream_model(config)

            # Note: Wav2Vec2 encoders are part of the DyStream model
            # Access via: model.audio_encoder_face and model.audio_encoder_other
            logger.info("Wav2Vec2 audio encoders are part of DyStream model")

            # Load visualization models (face encoder, generator, flow estimator)
            logger.info("Loading visualization models...")
            self.models.update(self._load_visualization_models(config))

            # Apply optimizations
            logger.info("Applying optimizations...")
            self._optimize_models(config)

            # Warmup inference
            logger.info("Warming up models...")
            self._warmup_inference()

            ModelLoader._models_loaded = True
            logger.info("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

        return self.models

    def _load_dystream_model(self, config: Dict[str, Any]):
        """Load the DyStream motion generation model."""
        from utils import instantiate_motion_gen, Config
        from omegaconf import OmegaConf
        from torch_ema import ExponentialMovingAverage

        # Build config with overrides (same as app.py)
        config_path = "configs/motion_gen/sample.yaml"
        override_args = {
            "exp_name": "streaming_server",
            "model.module_name": "model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder",
            "resume_ckpt": config.get('checkpoint_path', 'checkpoints/last.ckpt'),
        }

        cfg = Config(config_path, override_args)

        # Instantiate model
        model = instantiate_motion_gen(
            module_name=cfg.model.module_name,
            class_name=cfg.model.class_name,
            cfg=cfg.model,
            hfstyle=False
        )

        # Load checkpoint
        checkpoint_path = cfg.resume_ckpt
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Extract state dict
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Strip "model." prefix if from Lightning checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[len("model."):]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Checkpoint loaded successfully")

            # Load EMA if available
            ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
            if "ema_state" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state"])
                logger.info("EMA state loaded")

            # Store EMA for inference
            self.models['dystream_ema'] = ema
            self.models['dystream_cfg'] = cfg
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")

        model.eval().to(self.device)

        # Create noise scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **OmegaConf.to_container(cfg.noise_scheduler_kwargs, resolve=True)
        )
        self.models['noise_scheduler'] = noise_scheduler

        return model

    def _load_wav2vec_models(self, config: Dict[str, Any]):
        """
        Wav2Vec2 models are actually part of the DyStream model.
        No need to load separately - they're accessed via model.audio_encoder_face/audio_encoder_other
        """
        # Return None - not needed for streaming inference
        # The DyStream model contains the Wav2Vec encoders internally
        logger.info("Wav2Vec2 models are part of DyStream model - skipping separate loading")
        return None, None

    def _load_visualization_models(self, config: Dict[str, Any]):
        """
        Load models for visualization (motion latent to video frame).
        Based on app.py:142-187 (load_visualization_model)
        """
        import os
        import importlib.util as _ilu
        from omegaconf import OmegaConf

        # Setup paths (same as app.py)
        VIS_DIR = Path("tools/visualization_0416")
        VIS_MODEL_DIR = VIS_DIR / "utils" / "model_0506"

        if not VIS_DIR.exists():
            logger.error(f"Visualization directory not found: {VIS_DIR}")
            return {}

        # Add VIS_DIR to sys.path (for utils.face_detector etc.)
        # Do NOT add VIS_MODEL_DIR directly - it has its own `model/` package
        # that would shadow the project-root `model/` package.
        if str(VIS_DIR) not in sys.path:
            sys.path.insert(0, str(VIS_DIR))

        # Merge the two `model` namespaces so that both
        # `model.motion_generation.*` (project root) and
        # `model.head_animation.*` (vis tools) are importable.
        try:
            import model as _model_pkg  # loads PROJECT_ROOT/model
            _vis_model_dir_model = VIS_MODEL_DIR / "model"
            if str(_vis_model_dir_model) not in _model_pkg.__path__:
                _model_pkg.__path__.append(str(_vis_model_dir_model))
            logger.info("Merged model namespaces for visualization")
        except Exception as e:
            logger.error(f"Failed to merge model namespaces: {e}")
            return {}

        try:
            # Load config
            config_path = VIS_DIR / "configs" / "head_animator_best_0506.yaml"
            if not config_path.exists():
                logger.error(f"Visualization config not found: {config_path}")
                return {}

            vis_config = OmegaConf.load(str(config_path))

            # Fix relative checkpoint path
            vis_ckpt = vis_config.resume_ckpt
            if not os.path.isabs(vis_ckpt):
                vis_ckpt = os.path.normpath(os.path.join(str(VIS_DIR), vis_ckpt))

            if not Path(vis_ckpt).exists():
                logger.error(f"Visualization checkpoint not found: {vis_ckpt}")
                return {}

            # Load vis tools' own instantiate function
            # (to avoid conflicts with project-root utils.py)
            _vis_utils_spec = _ilu.spec_from_file_location(
                "vis_tools_utils", str(VIS_MODEL_DIR / "utils.py")
            )
            _vis_utils_mod = _ilu.module_from_spec(_vis_utils_spec)
            _vis_utils_spec.loader.exec_module(_vis_utils_mod)
            vis_instantiate = _vis_utils_mod.instantiate

            # Instantiate model
            module_cls = vis_instantiate(vis_config.model, instantiate_module=False)
            model = module_cls(config=vis_config)

            # Load checkpoint
            checkpoint = torch.load(vis_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model.eval().to(self.device)

            logger.info("Visualization models loaded successfully")

            return {
                'visualization': {
                    'flow_estimator': model.flow_estimator,
                    'face_generator': model.face_generator,
                    'face_encoder': model.face_encoder,
                    'motion_encoder': model.motion_encoder,
                }
            }

        except Exception as e:
            logger.error(f"Failed to load visualization models: {e}", exc_info=True)
            return {}

    def _optimize_models(self, config: Dict[str, Any]):
        """Apply optimizations to models."""

        # Note: Wav2Vec2 INT8 quantization is skipped as it's part of DyStream model
        # and quantizing it may cause issues with the integrated architecture

        # Torch Compile (optional - may cause issues with complex models)
        if config.get('use_torch_compile', False) and hasattr(torch, 'compile'):
            logger.info("torch.compile is disabled by default for stability")
            logger.info("Enable with caution - may cause issues with DyStream model")
            # Uncomment to enable:
            # try:
            #     self.models['dystream'] = torch.compile(
            #         self.models['dystream'],
            #         mode='reduce-overhead'
            #     )
            #     logger.info("torch.compile applied successfully")
            # except Exception as e:
            #     logger.warning(f"Failed to apply torch.compile: {e}")

        # Motion model compile (experimental)
        if config.get('use_torch_compile_motion', False) and hasattr(torch, 'compile'):
            try:
                logger.info("Applying torch.compile to motion model (experimental)")
                self.models['dystream'] = torch.compile(
                    self.models['dystream'],
                    mode='reduce-overhead'
                )
                logger.info("torch.compile applied to motion model")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile to motion model: {e}")

    def _warmup_inference(self):
        """Run warmup inference to initialize compiled graphs."""
        logger.info("Running warmup inference...")

        try:
            with torch.no_grad():
                # Create dummy inputs
                batch_size = 1
                seq_len = 32
                audio_dim = 768  # Wav2Vec2 feature dimension
                motion_dim = 512  # Motion latent dimension

                # Dummy audio features
                audio_self = torch.randn(batch_size, seq_len, audio_dim, device=self.device)
                audio_other = torch.randn(batch_size, seq_len, audio_dim, device=self.device)

                # Dummy motion anchor
                motion_anchor = torch.randn(batch_size, 94, motion_dim, device=self.device)

                # Run a few warmup iterations
                for _ in range(3):
                    # Note: actual warmup will depend on model's forward signature
                    # This is a placeholder - will be updated when integrating with actual model
                    pass

            logger.info("Warmup complete")

        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def get_models(self) -> Dict[str, Any]:
        """Get loaded models."""
        if not ModelLoader._models_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first.")
        return self.models

    def get_device(self) -> torch.device:
        """Get the device models are loaded on."""
        return self.device


# Global instance
model_loader = ModelLoader()
