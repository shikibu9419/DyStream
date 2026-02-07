"""
Session manager for WebSocket connections.
Maintains state for each streaming session.
"""

import uuid
import numpy as np
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """State for a single streaming session."""

    session_id: str
    mode: str = 'speaker'  # 'speaker' or 'listener'

    # Reference image and motion
    reference_image_path: Optional[str] = None
    motion_anchor: Optional[np.ndarray] = None  # Shape: (1, 512)
    face_features: Optional[torch.Tensor] = None  # Pre-encoded face features

    # Audio buffer
    audio_buffer: list = field(default_factory=list)

    # Motion context (maintained by inference engine)
    motion_context: Optional[torch.Tensor] = None

    # Configuration
    denoising_steps: int = 3
    cfg_audio: float = 0.5
    cfg_audio_other: float = 0.5
    cfg_anchor: float = 0.0
    cfg_all: float = 1.0

    # Status
    is_streaming: bool = False
    frames_generated: int = 0

    # Timestamps for latency tracking
    last_audio_timestamp: Optional[float] = None
    last_frame_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'mode': self.mode,
            'is_streaming': self.is_streaming,
            'frames_generated': self.frames_generated,
            'denoising_steps': self.denoising_steps,
            'cfg_audio': self.cfg_audio,
            'cfg_audio_other': self.cfg_audio_other,
            'cfg_anchor': self.cfg_anchor,
            'cfg_all': self.cfg_all
        }


class SessionManager:
    """Manages all active streaming sessions."""

    # Default avatars configuration
    DEFAULT_AVATARS = [
        {
            'id': 'avatar_1',
            'name': 'Default Avatar 1',
            'path': 'streaming_app/assets/default_avatars/avatar_1.png'
        },
        {
            'id': 'avatar_2',
            'name': 'Default Avatar 2',
            'path': 'streaming_app/assets/default_avatars/avatar_2.png'
        },
        {
            'id': 'avatar_3',
            'name': 'Default Avatar 3',
            'path': 'streaming_app/assets/default_avatars/avatar_3.png'
        }
    ]

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any], device: torch.device):
        self.config = config
        self.models = models
        self.device = device
        self.sessions: Dict[str, SessionState] = {}
        self.pending_reference_image_path: Optional[str] = None

        # Max concurrent sessions
        self.max_sessions = config.get('max_concurrent_sessions', 2)

    def create_session(self, avatar_id: str = 'avatar_1') -> SessionState:
        """
        Create a new streaming session with default avatar.

        Args:
            avatar_id: ID of default avatar to use

        Returns:
            New SessionState
        """
        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum concurrent sessions ({self.max_sessions}) reached")

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Get avatar info (or pending custom reference)
        avatar_info = next((a for a in self.DEFAULT_AVATARS if a['id'] == avatar_id), self.DEFAULT_AVATARS[0])
        reference_image_path = self.pending_reference_image_path or avatar_info['path']
        if self.pending_reference_image_path:
            logger.info(f"Using pending reference image for new session: {self.pending_reference_image_path}")
            self.pending_reference_image_path = None

        # Load motion anchor
        motion_anchor = self._load_motion_anchor(reference_image_path)

        # Create session
        session = SessionState(
            session_id=session_id,
            reference_image_path=reference_image_path,
            motion_anchor=motion_anchor,
            denoising_steps=self.config.get('denoising_steps', 3),
            cfg_audio=self.config.get('cfg_audio', 0.5),
            cfg_audio_other=self.config.get('cfg_audio_other', 0.5),
            cfg_anchor=self.config.get('cfg_anchor', 0.0),
            cfg_all=self.config.get('cfg_all', 1.0)
        )

        self.sessions[session_id] = session

        logger.info(f"Created session {session_id} with avatar {avatar_info['name']}")

        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Delete a session and clean up resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Clean up any GPU tensors
            if session.motion_context is not None:
                del session.motion_context
            if session.face_features is not None:
                del session.face_features

            del self.sessions[session_id]
            torch.cuda.empty_cache()

            logger.info(f"Deleted session {session_id}")

    def update_session_config(self, session_id: str, config: Dict[str, Any]):
        """Update session configuration."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # Update config fields
        if 'denoising_steps' in config:
            session.denoising_steps = config['denoising_steps']
        if 'cfg_audio' in config:
            session.cfg_audio = config['cfg_audio']
        if 'cfg_audio_other' in config:
            session.cfg_audio_other = config['cfg_audio_other']
        if 'cfg_anchor' in config:
            session.cfg_anchor = config['cfg_anchor']
        if 'cfg_all' in config:
            session.cfg_all = config['cfg_all']

        logger.info(f"Updated config for session {session_id}: {config}")

    def switch_mode(self, session_id: str, new_mode: str):
        """Switch session mode between 'speaker' and 'listener'."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if new_mode not in ['speaker', 'listener']:
            raise ValueError(f"Invalid mode: {new_mode}")

        session.mode = new_mode
        logger.info(f"Session {session_id} switched to {new_mode} mode")

    def update_reference_image(self, session_id: str, image_path: str):
        """
        Store a pending reference image for the next session.
        Current session is not updated to match app.py alignment policy.
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        self.pending_reference_image_path = image_path
        logger.info(f"Stored pending reference image for next session: {image_path}")

    def _load_motion_anchor(self, image_path: str) -> np.ndarray:
        """
        Load or compute motion anchor from reference image.

        Args:
            image_path: Path to reference image

        Returns:
            Motion anchor (94, 512)
        """
        # Check if pre-computed motion anchor exists
        npz_path = Path(image_path).with_suffix('.npz')

        if npz_path.exists():
            # Load pre-computed anchor
            data = np.load(npz_path)
            motion_anchor = data['motion_latent']
            logger.info(f"Loaded pre-computed motion anchor from {npz_path}")
        else:
            # Compute motion anchor from image
            logger.info(f"Computing motion anchor from {image_path}")
            motion_anchor = self._compute_motion_anchor(image_path)

            # Save for future use
            np.savez(npz_path, motion_latent=motion_anchor)
            logger.info(f"Saved motion anchor to {npz_path}")

        # Ensure correct shape (1, 512)
        if motion_anchor.ndim == 1:
            motion_anchor = motion_anchor[None, :]
        elif motion_anchor.ndim == 2 and motion_anchor.shape[0] > 1:
            motion_anchor = motion_anchor[:1]

        return motion_anchor

    def _compute_motion_anchor(self, image_path: str) -> np.ndarray:
        """
        Compute motion anchor from reference image using motion encoder.
        Simplified version focused on motion latent extraction.

        Args:
            image_path: Path to reference image

        Returns:
            Motion anchor (94, 512) - replicated single latent
        """
        try:
            from PIL import Image
            from streaming_app.utils.image_processing import process_image

            # Get visualization models
            vis_models = self.models.get('visualization')
            if vis_models is None:
                logger.error("Visualization models not loaded")
                return np.random.randn(1, 512).astype(np.float32)

            # Load and preprocess image (app.py-equivalent)
            img_pil = Image.open(image_path).convert("RGB")
            _, _, motion_latent = process_image(
                image_pil=img_pil,
                device=self.device,
                vis_models=vis_models,
                tools_path=self.config.get('tools_path', 'tools/visualization_0416')
            )

            motion_latent_np = motion_latent.cpu().numpy()  # (1, 512)
            logger.info(f"Computed motion anchor from {image_path}: shape={motion_latent_np.shape}")
            return motion_latent_np.astype(np.float32)

        except Exception as e:
            logger.error(f"Error computing motion anchor from {image_path}: {e}", exc_info=True)
            # Return dummy anchor as fallback
            return np.random.randn(1, 512).astype(np.float32)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all active sessions."""
        return {sid: session.to_dict() for sid, session in self.sessions.items()}

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)

    @classmethod
    def get_default_avatars(cls) -> list:
        """Get list of default avatars."""
        return cls.DEFAULT_AVATARS
