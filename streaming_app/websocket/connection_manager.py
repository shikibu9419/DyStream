"""
WebSocket connection manager.
Handles WebSocket connections and message routing.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import time

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self):
        # Active connections: session_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

        # Connection metadata
        self.connection_times: Dict[str, float] = {}

        # Message queues for each connection (for buffering)
        self.message_queues: Dict[str, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_times[session_id] = time.time()
        self.message_queues[session_id] = asyncio.Queue()

        logger.info(f"WebSocket connected: session {session_id}")

    def disconnect(self, session_id: str):
        """Disconnect a WebSocket."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            del self.connection_times[session_id]
            del self.message_queues[session_id]

            logger.info(f"WebSocket disconnected: session {session_id}")

    async def send_text(self, session_id: str, message: dict):
        """Send JSON text message to a session."""
        if session_id not in self.active_connections:
            logger.warning(f"Cannot send message - session {session_id} not connected")
            return

        try:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending text message to {session_id}: {e}")
            self.disconnect(session_id)

    async def send_bytes(self, session_id: str, data: bytes):
        """Send binary message to a session."""
        if session_id not in self.active_connections:
            logger.warning(f"Cannot send bytes - session {session_id} not connected")
            return

        try:
            websocket = self.active_connections[session_id]
            await websocket.send_bytes(data)
        except Exception as e:
            logger.error(f"Error sending binary message to {session_id}: {e}")
            self.disconnect(session_id)

    async def broadcast_text(self, message: dict):
        """Broadcast JSON text message to all connected sessions."""
        disconnected = []

        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)

        # Clean up disconnected sessions
        for session_id in disconnected:
            self.disconnect(session_id)

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_session_uptime(self, session_id: str) -> float:
        """Get session uptime in seconds."""
        if session_id not in self.connection_times:
            return 0.0
        return time.time() - self.connection_times[session_id]

    def is_connected(self, session_id: str) -> bool:
        """Check if session is connected."""
        return session_id in self.active_connections

    async def receive_message(self, websocket: WebSocket):
        """
        Receive message from WebSocket (text or binary).

        Returns:
            Tuple of (message_type, data) where message_type is 'text' or 'binary'
        """
        try:
            # Try to receive as text first
            data = await websocket.receive()

            if 'text' in data:
                return 'text', data['text']
            elif 'bytes' in data:
                return 'binary', data['bytes']
            else:
                logger.warning(f"Unknown message type: {data}")
                return None, None

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            raise
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None, None

    async def send_error(self, session_id: str, error_message: str, error_code: str = "ERROR"):
        """Send error message to a session."""
        await self.send_text(session_id, {
            'type': 'error',
            'code': error_code,
            'message': error_message
        })

    async def send_status(self, session_id: str, status: str, data: Optional[dict] = None):
        """Send status update to a session."""
        message = {
            'type': 'status',
            'status': status
        }
        if data:
            message.update(data)

        await self.send_text(session_id, message)

    def get_all_sessions(self) -> Set[str]:
        """Get all active session IDs."""
        return set(self.active_connections.keys())
