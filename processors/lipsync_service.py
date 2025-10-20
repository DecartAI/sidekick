import asyncio
from loguru import logger
from typing import Optional, Tuple

from pipecat.frames.frames import (
    Frame,
    OutputImageRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    StartFrame,
    StopFrame,
    CancelTaskFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


from lipsync.client import DecartLipsyncClient


class DecartLipsyncService(FrameProcessor):

    def __init__(self, api_key: str, audio_sample_rate: int = 16000, video_fps: int = 25, sync_latency: float = 0.0):
        super().__init__()
        self._lipsync_client = DecartLipsyncClient(
            api_key=api_key, audio_sample_rate=audio_sample_rate, video_fps=video_fps, sync_latency=sync_latency
        )
        self._audio_sample_rate = audio_sample_rate

        self._bot_is_talking: bool = False
        self._lipsynced_media_consumer_task: Optional[asyncio.Task] = None
        self._video_size: Tuple[int, int] = (0, 0)

    async def _setup(self):
        # Initialize connection
        try:
            await self._lipsync_client.connect()
        except Exception as e:
            logger.error(f"Error connecting to lipsync API: {e}")
            await self.push_frame(CancelTaskFrame(), FrameDirection.UPSTREAM)
        else:
            self._lipsynced_media_consumer_task = asyncio.create_task(self._consume_lipsynced_media())

    async def cleanup(self):
        await super().cleanup()

        if self._lipsynced_media_consumer_task:
            self._lipsynced_media_consumer_task.cancel()
            try:
                await self._lipsynced_media_consumer_task
            except asyncio.CancelledError:
                pass
            self._lipsynced_media_consumer_task = None

        if self._lipsync_client:
            await self._lipsync_client.disconnect()
            self._lipsync_client = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._setup()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_talking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_talking = False
        elif isinstance(frame, InterruptionFrame):
            logger.info("User started speaking, interrupting audio")
            # User started speaking, should clear server's audio queue
            await self._lipsync_client.interrupt_audio()
        elif isinstance(frame, TTSAudioRawFrame):
            await self._lipsync_client.send_audio(frame.audio)
            # Don't push TTS audio downstream - it will be sent after lipsync
            return
        elif isinstance(frame, OutputImageRawFrame):
            self._video_size = frame.size
            await self._lipsync_client.send_video(frame.image)
            # Don't push video frames downstream - they will be sent after lipsync
            return

        # Pass other frames through normally
        await self.push_frame(frame, direction)

    async def _handle_bot_silence(self):
        if self._bot_is_talking:
            logger.info("Bot stopped talking")
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_is_talking = False

    async def _consume_lipsynced_media(self):
        while True:
            video_frame, audio_frame = await self._lipsync_client.get_synced_output()

            is_silence = len(audio_frame) == 0 or all(b == 0 for b in audio_frame)
            if not is_silence:
                # Non silent audio frame, push as TTS frame
                await self.push_frame(
                    TTSAudioRawFrame(audio=audio_frame, sample_rate=self._audio_sample_rate, num_channels=1),
                    FrameDirection.DOWNSTREAM,
                )
            elif len(audio_frame) > 0:
                # Silent audio frame, first notify pipeline that bot is stopped talking
                await self._handle_bot_silence()

                # Then push silent audio frame, this keeps the video and audio in sync
                # (`OutputAudioRawFrame` doesn't trigger bot started speaking frame as `TTSAudioRawFrame` does)
                await self.push_frame(
                    OutputAudioRawFrame(audio=audio_frame, sample_rate=self._audio_sample_rate, num_channels=1),
                    FrameDirection.DOWNSTREAM,
                )
            else:
                # No audio frame, don't push anything (audio is sent once per chunk of 16 video frames)
                pass

            await self.push_frame(
                # TODO: Configurable size
                OutputImageRawFrame(image=video_frame, size=self._video_size, format="RGB24"),
                FrameDirection.DOWNSTREAM,
            )
