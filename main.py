"""eye_doctor — entry point.

Initialises the Agent with the ImageVisualizerTool and launches the Gradio demo.
Requires an OpenAI-compatible API key supplied via OPENAI_API_KEY in the
environment (or in a local .env file).
"""

import os
import warnings
from typing import Any, Dict, Optional

import gradio as gr
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from interface import build_application
from eye_doctor.agent import Agent
from eye_doctor.tools import ImageVisualizerTool
from eye_doctor.utils import load_prompts_from_file

warnings.filterwarnings("ignore")
_ = load_dotenv()


def initialize_agent(
    prompt_file: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.95,
    openai_kwargs: Dict[str, Any] = None,
) -> Agent:
    """Initialise the eye_doctor agent with the ImageVisualizerTool.

    Args:
        prompt_file (str): Path to the system prompts file.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.
        top_p (float): Sampling top-p.
        openai_kwargs (dict, optional): extra kwargs forwarded to ``ChatOpenAI``
            (e.g. ``api_key``, ``base_url``). If ``None``, these are collected
            from the ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` environment
            variables.

    Returns:
        Agent: initialised agent.
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    tools = [
        ImageVisualizerTool(),
    ]

    if openai_kwargs is None:
        openai_kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            openai_kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = base_url

    checkpointer = MemorySaver()
    chat_model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)

    agent = Agent(
        chat_model,
        tools=tools,
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent


if __name__ == "__main__":
    print("Starting server...")
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    agent: Optional[Agent] = None
    try:
        agent = initialize_agent(
            prompt_file="eye_doctor/prompts/system_prompts.txt",
            model=model,
            temperature=0.7,
            top_p=0.95,
        )
    except (openai.OpenAIError, ValueError, KeyError) as e:
        print(
            "⚠️  No OPENAI_API_KEY found (or key is invalid). "
            "UI will load but agent calls will fail. "
            "Set OPENAI_API_KEY in .env or env vars and restart."
        )
        print(f"   Underlying error: {e}")

    demo = build_application(agent)
    demo.launch(server_name="0.0.0.0", server_port=8090, share=False, theme=gr.themes.Soft())
