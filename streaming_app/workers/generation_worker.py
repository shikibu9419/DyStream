"""
Background worker for asynchronous motion generation and frame rendering.
Separates heavy computation from WebSocket audio reception.
"""

import asyncio
import torch
import numpy as np
from typing import Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class GenerationWorker:
    """
    Background worker for motion generation and frame rendering.
    Runs in separate asyncio task to prevent blocking audio reception.
    """

    def __init__(
        self,
        inference_engine,
        frame_renderer,
        audio_processor,
        session_id: str
    ):
        """
        Initialize generation worker.

        Args:
            inference_engine: StreamingInferenceEngine instance
            frame_renderer: FrameRenderer instance
            audio_processor: AudioProcessor instance
            session_id: Session identifier
        """
        self.inference_engine = inference_engine
        self.frame_renderer = frame_renderer
        self.audio_processor = audio_processor
        self.session_id = session_id
        self.device = getattr(inference_engine, "device", None)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda and self.device is None:
            self.device = torch.device("cuda")

        # Queues for async processing
        self.audio_queue = asyncio.Queue(maxsize=10)    # Audio chunks to process
        self.feature_queue = asyncio.Queue(maxsize=10)  # Extracted audio features
        self.motion_queue = asyncio.Queue(maxsize=10)   # Motion latents
        self.frame_queue = asyncio.Queue(maxsize=5)     # Encoded frames to send

        # Control flags
        self.running = False
        self.task = None
        self._tasks = []

        # CUDA streams and events for overlap (best-effort)
        if self.use_cuda:
            self.stream_audio = torch.cuda.Stream(device=self.device)
            self.stream_motion = torch.cuda.Stream(device=self.device)
            self.stream_render = torch.cuda.Stream(device=self.device)

        # Metrics
        self.frames_generated = 0
        self.total_audio_time = 0.0
        self.total_motion_time = 0.0
        self.total_frame_time = 0.0

    async def start(self):
        """Start the worker task."""
        if self.running:
            logger.warning(f"Worker for session {self.session_id} already running")
            return

        self.running = True
        # Pipeline tasks
        self._tasks = [
            asyncio.create_task(self._audio_loop()),
            asyncio.create_task(self._motion_loop()),
            asyncio.create_task(self._frame_loop()),
        ]
        logger.info(f"Worker started for session {self.session_id}")

    async def stop(self):
        """Stop the worker task."""
        if not self.running:
            return

        self.running = False

        # Wait for task to complete
        if self._tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"Worker task timeout for session {self.session_id}")
                for t in self._tasks:
                    t.cancel()

        logger.info(f"Worker stopped for session {self.session_id}. "
                   f"Generated {self.frames_generated} frames. "
                   f"Avg times - audio: {self.total_audio_time/max(1, self.frames_generated):.3f}s, "
                   f"motion: {self.total_motion_time/max(1, self.frames_generated):.3f}s, "
                   f"frame: {self.total_frame_time/max(1, self.frames_generated):.3f}s")

    async def queue_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        mode: str = 'speaker'
    ):
        """
        Queue an audio chunk for processing.

        Args:
            audio_chunk: Raw audio samples
            mode: 'speaker' or 'listener'
        """
        try:
            # Non-blocking put with timeout
            await asyncio.wait_for(
                self.audio_queue.put((audio_chunk, mode)),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            logger.warning(f"Audio queue full for session {self.session_id}, dropping chunk")

    async def get_next_frame(self, timeout: float = 0.5) -> Optional[bytes]:
        """
        Get the next generated frame.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            JPEG frame bytes or None if timeout
        """
        try:
            frame_bytes = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=timeout
            )
            return frame_bytes
        except asyncio.TimeoutError:
            return None

    async def _audio_loop(self):
        """Audio chunk -> features."""
        logger.info(f"Audio loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    audio_chunk, mode = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()
                if self.use_cuda:
                    def _run_audio():
                        with torch.cuda.stream(self.stream_audio):
                            return self.audio_processor.add_audio_chunk(audio_chunk, mode)
                    result = await asyncio.to_thread(_run_audio)
                else:
                    result = await asyncio.to_thread(self.audio_processor.add_audio_chunk, audio_chunk, mode)
                audio_time = time.time() - t0
                self.total_audio_time += audio_time

                if result is None:
                    continue

                if self.use_cuda:
                    event = torch.cuda.Event()
                    self.stream_audio.record_event(event)
                else:
                    event = None

                try:
                    await asyncio.wait_for(self.feature_queue.put((result, mode, event)), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Feature queue full for session {self.session_id}, dropping features")
            except Exception as e:
                logger.error(f"Error in audio loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Audio loop stopped for session {self.session_id}")

    async def _motion_loop(self):
        """Features -> motion latent."""
        logger.info(f"Motion loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    (audio_self_features, audio_other_features, audio_self_raw, audio_other_raw), mode, event = await asyncio.wait_for(
                        self.feature_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()
                if self.use_cuda:
                    def _run_motion():
                        if event is not None:
                            torch.cuda.current_stream().wait_event(event)
                        with torch.cuda.stream(self.stream_motion):
                            return self.inference_engine.generate_next_frame(
                                audio_self_features,
                                audio_other_features,
                                audio_self_raw,
                                audio_other_raw,
                                mode
                            )
                    motion_latent = await asyncio.to_thread(_run_motion)
                else:
                    motion_latent = await asyncio.to_thread(
                        self.inference_engine.generate_next_frame,
                        audio_self_features,
                        audio_other_features,
                        audio_self_raw,
                        audio_other_raw,
                        mode
                    )
                motion_time = time.time() - t0
                self.total_motion_time += motion_time

                if self.use_cuda:
                    event = torch.cuda.Event()
                    self.stream_motion.record_event(event)
                else:
                    event = None

                try:
                    await asyncio.wait_for(self.motion_queue.put((motion_latent, event)), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Motion queue full for session {self.session_id}, dropping motion")
            except Exception as e:
                logger.error(f"Error in motion loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Motion loop stopped for session {self.session_id}")

    async def _frame_loop(self):
        """Motion latent -> raw frame bytes."""
        import struct
        logger.info(f"Frame loop started for session {self.session_id}")
        while self.running:
            try:
                try:
                    motion_latent, event = await asyncio.wait_for(self.motion_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                t0 = time.time()
                if self.use_cuda:
                    def _run_render():
                        if event is not None:
                            torch.cuda.current_stream().wait_event(event)
                        with torch.cuda.stream(self.stream_render):
                            return self.frame_renderer.render_frame_raw(motion_latent)
                    w, h, rgb_bytes = await asyncio.to_thread(_run_render)
                else:
                    w, h, rgb_bytes = await asyncio.to_thread(
                        self.frame_renderer.render_frame_raw,
                        motion_latent
                    )
                frame_time = time.time() - t0
                self.total_frame_time += frame_time

                # Message format: [0x04][uint16 w][uint16 h][rgb]
                message = bytes([0x04]) + struct.pack("<HH", w, h) + rgb_bytes

                try:
                    await asyncio.wait_for(self.frame_queue.put(message), timeout=0.1)
                    self.frames_generated += 1

                    if self.frames_generated % 25 == 0:
                        total_time = self.total_audio_time / max(1, self.frames_generated)
                        motion_avg = self.total_motion_time / max(1, self.frames_generated)
                        frame_avg = self.total_frame_time / max(1, self.frames_generated)
                        logger.info(f"Session {self.session_id} - Frame {self.frames_generated}: "
                                   f"audio={total_time*1000:.1f}ms, "
                                   f"motion={motion_avg*1000:.1f}ms, "
                                   f"frame={frame_avg*1000:.1f}ms")
                except asyncio.TimeoutError:
                    logger.warning(f"Frame queue full for session {self.session_id}, dropping frame")
            except Exception as e:
                logger.error(f"Error in frame loop for session {self.session_id}: {e}", exc_info=True)

        logger.info(f"Frame loop stopped for session {self.session_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics."""
        if self.frames_generated == 0:
            return {
                'frames_generated': 0,
                'avg_audio_time_ms': 0,
                'avg_motion_time_ms': 0,
                'avg_frame_time_ms': 0,
                'avg_total_time_ms': 0
            }

        return {
            'frames_generated': self.frames_generated,
            'avg_audio_time_ms': (self.total_audio_time / self.frames_generated) * 1000,
            'avg_motion_time_ms': (self.total_motion_time / self.frames_generated) * 1000,
            'avg_frame_time_ms': (self.total_frame_time / self.frames_generated) * 1000,
            'avg_total_time_ms': ((self.total_audio_time + self.total_motion_time + self.total_frame_time) / self.frames_generated) * 1000
        }
