# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WebRTC-based video call conversational bot that provides real-time audio and video interaction with lip synchronization. The system uses the Pipecat framework for pipeline orchestration and integrates with various AI services (Groq LLM, ElevenLabs TTS, Whisper STT) and Decart's lipsync service.

## Key Architecture Components

**Main Server (`sidekick.py`)**
- WebSocket server listening on port 8088 that handles WebRTC signaling
- Creates Pipecat pipeline with audio/video processing components
- Manages WebRTC connection through SmallWebRTCTransport

**Pipeline Flow**
1. Audio input → Whisper STT → Groq LLM → ElevenLabs TTS
2. Video frames from `VideoFileStreamer` + TTS audio → `DecartLipsyncService` → synchronized audio/video output
3. Uses SmartTurnAnalyzer v3 for conversation turn management

**Lipsync Integration**
- `lipsync/client.py`: WebSocket client for Decart's lipsync API
- `processors/lipsync_service.py`: Pipecat processor that coordinates audio/video synchronization
- `processors/video_file_streamer.py`: Streams video frames from local file at specified FPS

## Development Commands

```bash
# Run the main server
python sidekick.py

# Environment setup - copy and configure API keys
cp .env.example .env
# Edit .env to add: GROQ_API_KEY, ELEVENLABS_API_KEY, DECART_API_KEY
```

## Key Dependencies

- pipecat-ai: Pipeline framework for audio/video processing
- aiortc: WebRTC implementation
- websockets: WebSocket server/client
- opencv-python (cv2): Video processing
- loguru: Logging

## Important Configuration

- WebSocket server: port 8088
- Audio sample rate: 16000 Hz
- Video output: 450x800 @ 25 FPS
- Video source: `videos/Cleopatra.mp4` (hardcoded in sidekick.py:184)
- Voice ID: ElevenLabs "Sarah" (hardcoded in sidekick.py:127)