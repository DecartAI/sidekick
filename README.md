# Sidekick 🤖

Real-time AI video call assistant with perfectly synchronized lip movements. Talk to historical figures, fictional characters, or create your own.

## What is this?

Sidekick lets you have face-to-face video conversations with AI characters. Unlike typical voice assistants, these characters have visual presence with realistic lip-sync, making conversations feel natural and engaging.

## Features

- **Live Video Conversations** - WebRTC-powered real-time video calls with AI characters
- **Perfect Lip Sync** - Powered by Decart's cutting-edge lipsync technology
- **Customizable Characters** - Define personality, voice, and appearance via simple YAML configs
- **Low Latency** - Optimized pipeline for natural conversation flow
- **Smart Interruptions** - Handles conversation turns naturally with VAD and smart turn detection

## Requirements

- Python 3.10+
- API keys for Groq, ElevenLabs, and Decart
- A character video file (static face video works best)
- Decent internet connection for real-time streaming

## Quick Start

1. **Clone and install**

```bash
git clone https://github.com/yourusername/sidekick-os.git
cd sidekick
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Set up your API keys**

```bash
cp .env.example .env
# Add your API keys:
# - GROQ_API_KEY (LLM)
# - ELEVENLABS_API_KEY (Voice)
# - DECART_API_KEY (Lipsync)
```

3. **Run with a character**

```bash
# Talk to Cleopatra
python sidekick.py --character cleopatra.yaml

# Or meet V1X3N, the sassy AI
python sidekick.py --character v1x3n.yaml
```

4. **Open `client.html` in your browser and hit Connect**

## Creating Your Own Characters

Characters are defined in YAML files. Here's the structure:

```yaml
name: YourCharacter
voice_id: elevenlabs_voice_id
video_path: videos/YourCharacter.mp4
greeting: "Your character's opening line"
system_prompt: |
  Detailed personality and behavior instructions
  for the LLM to roleplay as your character
```

See `cleopatra.yaml` and `v1x3n.yaml` for examples.

## Architecture

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) for pipeline orchestration:

- **STT**: Whisper (MLX optimized on Mac)
- **LLM**: Groq (Llama 3.3 70B)
- **TTS**: ElevenLabs
- **Lipsync**: Decart
- **Transport**: WebRTC via aiortc

## Command Line Options

```bash
python sidekick.py [options]

Options:
  --character PATH     Character config file (required)
  --host HOST         Server host (default: 0.0.0.0)
  --port PORT         Server port (default: 8080)
  --mlx               Use MLX for Whisper STT (Mac only)
  --audio-sample-rate Audio sample rate (default: 16000)
```

## License

MIT

## Contributing

PRs welcome! Please check existing issues first.

---

_Built with ❤️ for more natural AI interactions_
