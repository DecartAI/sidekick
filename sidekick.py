import aiohttp
import aiortc
import asyncio
import json
import websockets
import sys
import dotenv
import os
import argparse
import yaml
from loguru import logger

from processors.lipsync_service import DecartLipsyncService, DecartLipsyncClient
from processors.video_file_streamer import VideoFileStreamer

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsWsTTSService
from pipecat.services.whisper.stt import WhisperSTTService, MLXModel, WhisperSTTServiceMLX
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3


logger.remove()
logger.add(sys.stderr, level="INFO")

dotenv.load_dotenv()

class SidekickWebRTCServer:
    """WebRTC server that uses the Sidekick framework with LipSync"""

    def __init__(self, character_config: dict, use_mlx: bool = True, audio_sample_rate: int = 16000):
        self.websocket_connection = None
        self.rtc_connection = None
        self.pipeline_task = None
        self.pipeline_runner = None
        self.character_config = character_config
        self.use_mlx = use_mlx
        self.aiohttp_session = aiohttp.ClientSession()
        self.runner_task = None
        self.audio_sample_rate = audio_sample_rate

    async def handle_websocket_message(self, message):
        """Handle incoming WebSocket messages"""
        data = json.loads(message)
        msg_type = data.get("type")

        if msg_type == "offer":
            # Handle WebRTC offer - create RTCSessionDescription from raw data
            offer = aiortc.RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await self.setup_webrtc_connection(offer)
        else:
            logger.error(f"Unknown WebSocket message type: {msg_type}")

    async def setup_webrtc_connection(self, offer):
        """Setup WebRTC connection and Sidekick pipeline"""
        # Create WebRTC connection
        self.rtc_connection = SmallWebRTCConnection(ice_servers=["stun:stun.l.google.com:19302"])
        await self.rtc_connection.initialize(offer.sdp, offer.type)

        # Send answer back
        answer = self.rtc_connection.get_answer()
        await self.websocket_connection.send(json.dumps({"sdp": answer["sdp"], "type": answer["type"]}))

        # Create the Sidekick pipeline
        await self.create_sidekick_pipeline()

    async def create_sidekick_pipeline(self):
        """Create the pipeline manually with full control"""

        if self.use_mlx:
            stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO)
        else:
            stt = WhisperSTTService(
                model="base",
                device="cpu",
                compute_type="int8",
            )

        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

        tts = ElevenLabsWsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=self.character_config["voice_id"],
            model="eleven_v3",
            aiohttp_session=self.aiohttp_session,
        )

        messages = [
            {
                "role": "system",
                "content": self.character_config["system_prompt"],
            }
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(
            context, assistant_params=LLMAssistantAggregatorParams(expect_stripped_words=False)
        )

        video_streamer = VideoFileStreamer(self.character_config["video_path"], fps=DecartLipsyncClient.VIDEO_FPS)

        lipsync_processor = DecartLipsyncService(
            os.getenv("DECART_API_KEY"),
            video_fps=DecartLipsyncClient.VIDEO_FPS,
            audio_sample_rate=self.audio_sample_rate,
        )

        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=video_streamer.video_size[0],
            video_out_height=video_streamer.video_size[1],
            video_out_framerate=DecartLipsyncClient.VIDEO_FPS,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.8, start_secs=0.25, stop_secs=0.2, min_volume=0.7)
            ),
            vad_analyzer_enabled=True,
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        )

        transport = SmallWebRTCTransport(webrtc_connection=self.rtc_connection, params=transport_params)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                video_streamer,
                lipsync_processor,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        self.pipeline_task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                audio_in_sample_rate=self.audio_sample_rate,
                audio_out_sample_rate=self.audio_sample_rate,
            ),
        )

        # Set up event handlers
        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            logger.info(f"Client connected via WebRTC with video support - Character: {self.character_config['name']}")
            # Send greeting using TTS service directly
            greeting_text = self.character_config["greeting"]
            logger.info(f"Sending greeting: {greeting_text}")
            from pipecat.frames.frames import TTSSpeakFrame

            await self.pipeline_task.queue_frames([TTSSpeakFrame(text=greeting_text)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info("Client disconnected, cleaning up pipeline")
            await self.cleanup()

        @transport.event_handler("on_connection_error")
        async def on_connection_error(_transport, error):
            logger.error(f"WebRTC Connection error: {error}, cleaning up pipeline")
            await self.cleanup()

        # Run the pipeline
        self.pipeline_runner = PipelineRunner(handle_sigint=False)
        self.runner_task = asyncio.create_task(self.pipeline_runner.run(self.pipeline_task))
        logger.info("Pipeline runner started")

    async def cleanup(self):
        """Clean up resources"""
        if self.pipeline_task:
            await self.pipeline_task.cancel()

        if self.runner_task:
            try:
                await self.runner_task
                self.runner_task = None
            except asyncio.CancelledError:
                pass

        if self.rtc_connection:
            # SmallWebRTCConnection doesn't have a public close method in new version
            # It's cleaned up automatically when the connection goes out of scope
            self.rtc_connection = None

    async def handle_client(self, websocket):
        """Handle a WebSocket client connection"""
        self.websocket_connection = websocket
        logger.info("WebSocket client connected")

        try:
            async for message in websocket:
                await self.handle_websocket_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.cleanup()


def load_character_config(config_path: str) -> dict:
    """Load character configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Sidekick WebRTC Server - AI video call assistant")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--mlx", action="store_true", default=False, help="Use MLX for Whisper STT (default: True)")
    parser.add_argument("--character", type=str, required=True, help="Path to character configuration YAML file")
    parser.add_argument(
        "--audio-sample-rate", type=int, default=16000, help="Audio sample rate for lipsync (default: 16000)"
    )
    return parser.parse_args()


async def main():
    args = parse_arguments()

    # Load character configuration
    try:
        character_config = load_character_config(args.character)
        logger.info(f"Loaded character: {character_config['name']}")
    except Exception as e:
        logger.error(f"Failed to load character config: {e}")
        return

    # Create server instance
    server = SidekickWebRTCServer(character_config, use_mlx=args.mlx)

    logger.info(f"Starting Sidekick WebRTC server on {args.host}:{args.port}")
    logger.info(f"Character: {character_config['name']}")
    logger.info(f"Using MLX: {args.mlx}")
    logger.info("This server includes video output with lip synchronization!")

    async with websockets.serve(server.handle_client, args.host, args.port):
        logger.info(f"Server listening on ws://{args.host}:{args.port}")
        logger.info("Open client.html in a browser to connect")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
