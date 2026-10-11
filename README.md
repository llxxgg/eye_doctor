# eye_doctor

The ophthalmology diagnosis agent identifies the cause of the disease through ophthalmology images and provides medical advice.

A LangGraph-powered ophthalmology assistant that analyses fundus photographs, OCT scans, retinal images and slit-lamp photos through a Gradio chat UI. The agent can call a single image-processing tool to annotate or transform the uploaded image, then synthesises findings into a chat reply.

## Features

- Multimodal analysis: take an ophthalmic image as input and ask follow-up questions in natural language.
- Tool-augmented reasoning via LangGraph `process` / `execute` workflow nodes.
- Image annotation and re-visualisation through `ImageVisualizerTool` (matplotlib / scikit-image).
- Multi-turn chat with persistent thread memory (`MemorySaver` checkpointer).
- Gradio Blocks demo with image upload and per-thread reset.
- Pluggable OpenAI-compatible backend (works with `OPENAI_BASE_URL` pointing at any compatible provider).

## Installation

Requires Python 3.10-3.14.

```bash
git clone <repo-url> eye_doctor
cd eye_doctor
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable           | Required | Default                     | Description                                                  |
| ------------------ | -------- | --------------------------- | ------------------------------------------------------------ |
| `OPENAI_API_KEY`   | yes      | —                           | OpenAI-compatible API key. The app still launches without it, but agent calls will fail. |
| `OPENAI_BASE_URL`  | no       | `https://api.openai.com/v1` | Any OpenAI-compatible base URL (proxies, local LLMs, etc.).  |
| `EYE_DOCTOR_MODEL` | no       | `gpt-4o`                    | Model name passed to `ChatOpenAI`.                           |

## Usage

Launch the Gradio demo:

```bash
python main.py
```

The UI will start on `http://localhost:8585` (bound to `0.0.0.0:8585`). Upload an image, type a question, and the agent will respond with textual analysis plus any tool-generated visualisations.

Use the `Clear Chat` button to wipe history inside the current thread, or `New Thread` to start a fresh conversation.

## Architecture

The project is split into three layers:

- **Agent layer** (`eye_doctor/agent/`): a LangGraph `Agent` that orchestrates `process` and `execute` nodes, streams events back to the UI, and persists state via `MemorySaver`.
- **Tool layer** (`eye_doctor/tools/`): pluggable `BaseTool` implementations. Only `ImageVisualizerTool` is wired up today; new tools can be registered in `main.py`'s `tools_dict`.
- **Interface layer** (`interface.py`): a Gradio `Blocks` demo that handles image uploads, base64-encodes them for the multimodal model, and renders streamed agent events as chat messages.

System prompts for the agent (and a generic fallback) live in `eye_doctor/prompts/system_prompts.txt`.

## Limitations

- **Not a medical device.** Outputs are for educational and research exploration only; do not use them for clinical decisions.
- **Single tool.** Only `ImageVisualizerTool` is currently registered — the agent cannot, for example, run OCR, segmentation, or external knowledge lookups.
- **No DICOM support.** The UI exposes an "Upload OCT" button but the OCT/DICOM processing pipeline (a planned `DicomProcessorTool`) is intentionally not wired up; `.dcm` files are saved but not interpreted.
- **Requires your own OpenAI key.** The default `OPENAI_BASE_URL` points at the public OpenAI API, so a paid key is needed unless you re-point at another compatible provider.
- **In-memory state.** Conversation history lives in `MemorySaver`; restarting the process wipes it.

## Project structure

```
eye_doctor/
├── main.py                          # Entry point: agent init + Gradio launch
├── interface.py                     # Gradio Blocks demo
├── pyproject.toml                   # Package metadata & dependencies
├── .env.example                     # Environment variable template
├── assets/
│   └── medrax_logo.jpg              # Chatbot avatar
├── eye_doctor/
│   ├── agent/
│   │   └── agent.py                 # LangGraph Agent (process / execute)
│   ├── tools/
│   │   └── image_visualizer.py      # ImageVisualizerTool
│   ├── utils/
│   │   └── utils.py                 # load_prompts_from_file helpers
│   └── prompts/
│       └── system_prompts.txt       # MEDICAL_ASSISTANT & GENERAL_ASSISTANT
├── logs/                            # Agent tool-call log output
└── temp/                            # Upload staging directory
```

---

中文支持：聊天界面与系统提示兼容中英文，用户可在对话中直接使用中文提问与描述。
