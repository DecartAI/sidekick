"""
Latency instrumentation for the Sidekick pipeline.

Logs timestamps at key pipeline boundaries so we can measure end-to-end
LLM → TTS → LipSync latency.

Example log output:
  [LATENCY] llm_response_start: 1740437584.123
  [LATENCY] tts_first_audio: 1740437584.312 (delta +189ms from llm_response_start)
  [LATENCY] lipsync_first_frame: 1740437584.521 (delta +209ms from tts_first_audio)
"""

from time import perf_counter
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    OutputImageRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class LatencyLogger(FrameProcessor):
    """
    Passthrough processor that logs latency milestones.
    Insert after the TTS step and before the LipSync step in the pipeline.
    """

    def __init__(self):
        super().__init__()
        self._llm_start_t: float | None = None
        self._tts_first_t: float | None = None
        self._lipsync_first_t: float | None = None
        self._turn_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        now = perf_counter()

        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_start_t = now
            self._tts_first_t = None
            self._lipsync_first_t = None
            self._turn_active = True
            logger.info(f"[LATENCY] llm_response_start: {now:.6f}")

        elif isinstance(frame, TTSAudioRawFrame) and self._turn_active:
            if self._tts_first_t is None:
                self._tts_first_t = now
                delta = f"+{(now - self._llm_start_t) * 1000:.0f}ms" if self._llm_start_t else "n/a"
                logger.info(f"[LATENCY] tts_first_audio: {now:.6f} ({delta} from llm_response_start)")

        elif isinstance(frame, OutputImageRawFrame) and self._turn_active:
            if self._lipsync_first_t is None:
                self._lipsync_first_t = now
                delta = f"+{(now - self._tts_first_t) * 1000:.0f}ms" if self._tts_first_t else "n/a"
                logger.info(f"[LATENCY] lipsync_first_frame: {now:.6f} ({delta} from tts_first_audio)")

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._turn_active and self._llm_start_t and self._lipsync_first_t:
                total = (self._lipsync_first_t - self._llm_start_t) * 1000
                logger.info(f"[LATENCY] turn_total_to_first_frame: {total:.0f}ms")
            self._turn_active = False

        await self.push_frame(frame, direction)
