import asyncio
from pathlib import Path
import cv2
from loguru import logger
import fractions
import time
from typing import Tuple

from pipecat.frames.frames import OutputImageRawFrame, StartFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class VideoFileStreamer(FrameProcessor):
    """Frame processor that streams video frames at the correct timing"""

    def __init__(self, video_path: str, fps: int = 25):
        super().__init__()
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")

        self._fps = fps  # TODO: Currently we would play an higher fps video slower (in 25 fps), should enforce fps==25 or support other values

        self._video_cap = cv2.VideoCapture(str(video_path))
        self._video_size = (
            int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self._video_frames = []
        self._streaming_task = None

    @property
    def video_size(self) -> Tuple[int, int]:
        return self._video_size

    async def setup(self, setup):
        await super().setup(setup)

    async def cleanup(self):
        logger.info("Tearing down video file streamer")
        await super().cleanup()

        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._load_video()

            logger.info(f"Starting video file streaming with {len(self._video_frames)} frames at {self._fps} fps")
            self._streaming_task = asyncio.create_task(self._stream_video_frames())

        await self.push_frame(frame, direction)

    async def _load_video(self):
        """Load video frames from file"""

        def _load_video_sync():
            try:
                cap = self._video_cap
                self._video_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._video_frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # cv2 reads in BGR format, convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, encoded_frame = cv2.imencode(".jpeg", frame_rgb)

                    self._video_frames.append(encoded_frame.tobytes())

            except Exception as e:
                self._video_frames = []
                logger.error(f"Error loading video file: {e}")
                raise
            finally:
                if cap:
                    cap.release()
                logger.debug(f"Loaded {len(self._video_frames)} frames from {self._video_path}")

        await asyncio.to_thread(_load_video_sync)

    async def _stream_video_frames(self):
        if len(self._video_frames) == 0:
            logger.error("No video frames available for streaming")
            raise ValueError("No video frames available for streaming")

        try:
            frame_count = 0
            frame_interval = float(fractions.Fraction(1, self._fps))

            start_time = time.time()
            while True:
                # Get current frame (loop when reaching end)
                current_frame = self._video_frames[frame_count % len(self._video_frames)]

                # Create and push video frame
                video_frame = OutputImageRawFrame(image=current_frame, size=self._video_size, format="RGB")

                # Wait for next frame time
                time_til_frame = start_time + (frame_interval * frame_count) - time.time()
                if time_til_frame > 0:
                    await asyncio.sleep(time_til_frame)

                await self.push_frame(video_frame, FrameDirection.DOWNSTREAM)
                frame_count += 1

        except asyncio.CancelledError:
            logger.info(f"Video frame streaming cancelled after {frame_count} frames")
        except Exception as e:
            logger.error(f"Error streaming video frames after {frame_count} frames: {e}")
            raise
