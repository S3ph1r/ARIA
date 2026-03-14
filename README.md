# ARIA: AI Runtime Infrastructure for Applications

ARIA is a powerful, distributed node orchestrator designed to provide high-performance AI services (TTS, LLM, Computer Vision) via a centralized Redis-based queue system.

## 🚀 Key Features

- **Multi-Backend Orchestration**: Manage multiple AI models (Fish Speech, Qwen 3.5, Gemini) with JIT loading and unloading to optimize VRAM usage.
- **Hardware Agnostic**: Run local models on high-end GPUs while seamlessly failing over to Cloud APIs (Gemini Flash Lite) when local resources are busy or unavailable.
- **Thinking Mode Support**: Native extraction and handling of reasoning/thinking tokens for advanced LLMs like Qwen 3.5 MoE.
- **Advanced Rate Limiting**: Global pacing and quote management for cloud providers to prevent 429 errors across multi-node deployments.
- **Standardized API**: Simple Redis-based protocol for submitting tasks and receiving results, making it easy to integrate with any application.

## 🏗 Architecture

ARIA follows a distributed "Node & Orchestrator" pattern:
1. **Node Controller**: A Windows/Linux daemon that monitors Redis queues and spawns child processes for local AI backends.
2. **Model Backends**: Isolated environments for specialized tasks (TTS, LLM, STT).
3. **Cloud Manager**: A dedicated sequencial worker for handling third-party API calls without blocking local GPU resources.

## 📁 Project Structure

```
ARIA/
 aria_node_controller/   # Core Orchestrator logic
   ├── core/              # Main loop, QueueManager, RateLimiter
   ├── backends/          # Local backend connectors
   └── installer/         # Setup and environment scripts
 docs/                   # Full Technical Documentation (Blueprints, APIs)
 requirements/           # Component-specific dependencies
 scripts/                # Utility tools (Voice prepper, debug)
 node_settings.json.example  # Configuration template
```

## 🛠 Setup

1. **Clone the repository** (excluding assets):
   ```bash
   git clone https://github.com/S3ph1r/aria.git
   cd ARIA
   ```

2. **Configure Node**:
   Copy `node_settings.json.example` to `node_settings.json` and fill in your Redis internal IP and API keys.

3. **Install Core Dependencies**:
   ```bash
   pip install -r requirements/core.txt
   ```

4. **Install Backends**:
   Follow the guides in `docs/environments-setup.md` for specific model requirements (Fish, Qwen, etc.).

## 📖 Documentation

Detailed technical documentation can be found in the `docs/` directory:
- [ARIA Blueprint](docs/ARIA-blueprint.md)
- [Network Interface Spec](docs/ARIA-network-interface.md)
- [LLM Backend Details](docs/llm-backend.md)
- [Environment Setup](docs/environments-setup.md)

---
*Created by S3ph1r as part of the NH-Mini development framework.*
