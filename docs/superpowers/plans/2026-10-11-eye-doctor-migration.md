# eye_doctor MedRAX 迁移 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MedRAX 的 Agent 框架与 Gradio 前端迁移到 eye_doctor。仅迁移 `ImageVisualizerTool` 一个工具，使用眼科主题文案，依赖用户自备 OpenAI API key。

**Architecture:** 复制 MedRAX 的 `Agent` (LangGraph 工作流)、`ImageVisualizerTool`、`ChatInterface`、`create_demo` 与精简后的 `main.py`。包名 `medrax` → `eye_doctor`。系统提示与 UI 文案改眼科主题。删除 8 个 ML 工具（不动 torchxrayvision / transformers / pydicom）。所有 Python 文件用内置 `assert` 做最小冒烟，不引入 pytest。

**Tech Stack:** Python 3.14，langgraph, langchain-core, langchain-openai, gradio, pydantic, matplotlib, scikit-image, python-dotenv。

**Source spec:** `docs/superpowers/specs/2026-10-11-eye-doctor-migration-design.md`

---

## 文件总览

| 路径 | 状态 | 来源 / 责任 |
|---|---|---|
| `eye_doctor/__init__.py` | 新建 | 空，仅是包标识 |
| `eye_doctor/agent/__init__.py` | 新建 | 暴露 `Agent` |
| `eye_doctor/agent/agent.py` | 新建 | 来自 `MedRAX/medrax/agent/agent.py` |
| `eye_doctor/tools/__init__.py` | 新建 | 暴露 `ImageVisualizerTool` |
| `eye_doctor/tools/image_visualizer.py` | 新建 | 来自 `MedRAX/medrax/tools/utils.py` 的 ImageVisualizerTool |
| `eye_doctor/utils/__init__.py` | 新建 | 暴露 `load_prompts_from_file`, `load_tool_prompts` |
| `eye_doctor/utils/utils.py` | 新建 | 来自 `MedRAX/medrax/utils/utils.py` |
| `eye_doctor/docs/system_prompts.txt` | 新建 | 来自 `MedRAX/medrax/docs/system_prompts.txt` 并改眼科主题 |
| `assets/medrax_logo.jpg` | 新建 | 来自 `MedRAX/assets/medrax_logo.jpg` |
| `interface.py` | 新建 | 来自 `MedRAX/interface.py` 改眼科主题、删除 DICOM 分支 |
| `main.py` | 新建 | 来自 `MedRAX/main.py` 精简工具字典 |
| `pyproject.toml` | 新建 | 仅声明本次依赖 |
| `.env.example` | 新建 | OPENAI_API_KEY 模板 |
| `.gitignore` | 新建 | 标准 Python + `temp/`、`logs/`、`.gradio/` |
| `tests_smoke/test_imports.py` | 新建 | 一份冒烟测试（不是 pytest） |
| `README.md` | 修改 | 安装 + 配置 + 限制说明 |

**不在范围**：不引入 pytest、不创建 `tests/`、不创建 `tools.json`（一个工具硬编码到 prompt 里）。

---

## 执行顺序

按"基础设施 → 叶子模块 → 集成"的顺序。每个文件先验证缺失/失败，再实现，再验证通过。每个 Python 文件完成后立即 commit。

---

### Task 1: 目录骨架与空包标识

**Files:**
- Create: `eye_doctor/__init__.py`
- Create: `eye_doctor/agent/__init__.py`
- Create: `eye_doctor/tools/__init__.py`
- Create: `eye_doctor/utils/__init__.py`
- Create: `assets/.gitkeep`

- [ ] **Step 1: 创建目录与空文件**

```bash
mkdir -p eye_doctor/agent eye_doctor/tools eye_doctor/utils eye_doctor/docs assets
touch eye_doctor/__init__.py eye_doctor/agent/__init__.py eye_doctor/tools/__init__.py eye_doctor/utils/__init__.py assets/.gitkeep
```

- [ ] **Step 2: 用最小冒烟验证包可以被 import**

```bash
python -c "import eye_doctor; import eye_doctor.agent; import eye_doctor.tools; import eye_doctor.utils"
```

Expected: 无报错，进程退出码 0。

- [ ] **Step 3: Commit**

```bash
git add eye_doctor/__init__.py eye_doctor/agent/__init__.py eye_doctor/tools/__init__.py eye_doctor/utils/__init__.py assets/.gitkeep
git commit -m "feat: add empty eye_doctor package skeleton"
```

---

### Task 2: .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: 创建 `.gitignore`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/.gitignore`：

```
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
*.egg-info/
*.egg

# Environments
.env
.venv
env/
venv/

# Logs and runtime artifacts
logs/
temp/
.gradio/

# IDEs
.idea/
.vscode/
*.swp

# OS
.DS_Store
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for Python, secrets, runtime dirs"
```

---

### Task 3: pyproject.toml 与 .env.example

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: 创建 `pyproject.toml`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "eye-doctor"
version = "0.1.0"
description = "Ophthalmology diagnosis agent"
requires-python = ">=3.10,<3.15"
dependencies = [
  "langgraph>=0.2.0",
  "langchain-core>=0.3.0",
  "langchain-openai>=0.2.0",
  "gradio>=5.0",
  "pydantic>=2.5",
  "matplotlib>=3.8",
  "scikit-image>=0.22",
  "python-dotenv>=1.0",
]

[tool.setuptools.packages.find]
include = ["eye_doctor*"]
exclude = ["tests_smoke*", "docs*"]
```

- [ ] **Step 2: 创建 `.env.example`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/.env.example`：

```
# Required: your OpenAI-compatible credentials
OPENAI_API_KEY=sk-replace-me
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional override for the model name passed to ChatOpenAI
EYE_DOCTOR_MODEL=gpt-4o
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml .env.example
git commit -m "chore: add pyproject.toml and .env.example"
```

---

### Task 4: 迁移 assets/medrax_logo.jpg

**Files:**
- Create: `assets/medrax_logo.jpg`
- Remove: `assets/.gitkeep`（由实际文件替代）

- [ ] **Step 1: 复制 logo**

```bash
cp /Users/m11003005586/www/py_src/MedRAX/assets/medrax_logo.jpg assets/medrax_logo.jpg
rm -f assets/.gitkeep
ls -la assets/
```

Expected: `medrax_logo.jpg` 存在，大小大于 1 KB。

- [ ] **Step 2: Commit**

```bash
git add assets/medrax_logo.jpg
git commit -m "feat: add medrax logo asset for Gradio avatar"
```

---

### Task 5: eye_doctor/utils/utils.py（提示加载）

**Files:**
- Create: `eye_doctor/utils/utils.py`
- Modify: `eye_doctor/utils/__init__.py`

- [ ] **Step 1: 先验证函数尚未定义**

```bash
python -c "from eye_doctor.utils.utils import load_prompts_from_file"
```

Expected: `ModuleNotFoundError` 或 `ImportError`（确认现状）。

- [ ] **Step 2: 写入 `eye_doctor/utils/utils.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/utils/utils.py`：

```python
"""Prompt and tool description loading utilities."""

import json
import os
from typing import Dict, List


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file using the format::

        [PROMPT_NAME]
        Prompt content lines...

    Args:
        file_path (str): Path to the file containing prompts.

    Returns:
        Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
        FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts: Dict[str, str] = {}
    current_prompt: str = None
    current_content: List[str] = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts


def load_tool_prompts(tools: List[str], tools_json_path: str) -> str:
    """
    Load prompts for specified tools from the tools.json file.

    Args:
        tools (List[str]): List of tool names to load prompts for.
        tools_json_path (str): Path to the tools.json file.

    Returns:
        str: A string containing prompts for the specified tools.

    Raises:
        FileNotFoundError: If the tools.json file is not found.
    """
    if not os.path.exists(tools_json_path):
        raise FileNotFoundError(f"Tools JSON file not found: {tools_json_path}")

    with open(tools_json_path, "r") as file:
        tools_data = json.load(file)

    tool_prompts = []
    for tool in tools:
        if tool in tools_data:
            tool_info = tools_data[tool]
            tool_prompt = f"Tool: {tool}\n"
            tool_prompt += f"Description: {tool_info['description']}\n"
            tool_prompt += f"Usage: {tool_info['prompt']}\n"
            tool_prompt += f"Input type: {tool_info['input_type']}\n"
            tool_prompt += f"Return type: {tool_info['return_type']}\n\n"
            tool_prompts.append(tool_prompt)

    return "\n".join(tool_prompts)
```

- [ ] **Step 3: 改写 `eye_doctor/utils/__init__.py` 以公开符号**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/utils/__init__.py`：

```python
from .utils import load_prompts_from_file, load_tool_prompts
```

- [ ] **Step 4: 跑最小冒烟**

```bash
python -c "
from eye_doctor.utils import load_prompts_from_file, load_tool_prompts
import inspect, os
src = inspect.getsourcefile(load_prompts_from_file)
assert load_tool_prompts.__doc__ is not None, 'load_tool_prompts must have docstring'
assert src.endswith('utils/utils.py'), src
print('utils smoke OK')
"
```

Expected: 打印 `utils smoke OK`，退出码 0。

- [ ] **Step 5: Commit**

```bash
git add eye_doctor/utils/utils.py eye_doctor/utils/__init__.py
git commit -m "feat: add prompt loading utils (load_prompts_from_file, load_tool_prompts)"
```

---

### Task 6: eye_doctor/docs/system_prompts.txt

**Files:**
- Create: `eye_doctor/docs/system_prompts.txt`

- [ ] **Step 1: 写入眼科主题提示词**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/docs/system_prompts.txt`：

```
[MEDICAL_ASSISTANT]
You are an expert ophthalmology AI assistant who can analyze ophthalmic images
(fundus photographs, OCT scans, retinal images, slit-lamp photos) similarly to
an ophthalmologist.
Solve using your own visual reasoning and use tools to complement your reasoning.
Make multiple tool calls in parallel or sequence as needed for comprehensive answers.
Critically evaluate any tool outputs before forming an answer.
If you need to look up some information before asking a follow up question, you are allowed to do that.

[GENERAL_ASSISTANT]
You are a helpful AI assistant. Your role is to assist users with a wide range of tasks and questions, providing accurate and useful information on various topics.
```

- [ ] **Step 2: 用 utils 验证可读且含 MEDICAL_ASSISTANT**

```bash
python -c "
from eye_doctor.utils import load_prompts_from_file
prompts = load_prompts_from_file('eye_doctor/docs/system_prompts.txt')
assert 'MEDICAL_ASSISTANT' in prompts, prompts.keys()
assert 'GENERAL_ASSISTANT' in prompts
assert 'ophthalmology' in prompts['MEDICAL_ASSISTANT'].lower()
print('prompts OK')
"
```

Expected: 打印 `prompts OK`。

- [ ] **Step 3: Commit**

```bash
git add eye_doctor/docs/system_prompts.txt
git commit -m "feat: add ophthalmology-themed system prompts"
```

---

### Task 7: eye_doctor/tools/image_visualizer.py

**Files:**
- Create: `eye_doctor/tools/image_visualizer.py`
- Modify: `eye_doctor/tools/__init__.py`

- [ ] **Step 1: 验证工具类尚未导出**

```bash
python -c "from eye_doctor.tools.image_visualizer import ImageVisualizerTool"
```

Expected: `ImportError`（确认现状）。

- [ ] **Step 2: 写入 `eye_doctor/tools/image_visualizer.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/tools/image_visualizer.py`：

```python
"""Image Visualizer tool — displays annotated images and returns metadata."""

from pathlib import Path
from typing import Optional, Tuple, Type

import matplotlib.pyplot as plt
import skimage.io
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ImageVisualizerInput(BaseModel):
    """Input schema for the Image Visualizer Tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the image file to display, only supports JPG or PNG images"
    )
    title: Optional[str] = Field(None, description="Optional title to display above the image")
    description: Optional[str] = Field(
        None, description="Optional description to display below the image"
    )
    figsize: Optional[tuple] = Field(
        (10, 10), description="Optional figure size as (width, height) in inches"
    )
    cmap: Optional[str] = Field(
        "rgb", description="Optional colormap to use for displaying the image"
    )


class ImageVisualizerTool(BaseTool):
    """Tool for displaying medical images to users with annotations."""

    name: str = "image_visualizer"
    description: str = (
        "Displays images to users with optional titles and descriptions. "
        "Input: Path to image file and optional display parameters. "
        "Output: Dict with image path and metadata."
    )
    args_schema: Type[BaseModel] = ImageVisualizerInput

    def _display_image(
        self,
        image_path: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        figsize: tuple = (10, 10),
        cmap: str = "rgb",
    ) -> None:
        """Display an image with optional annotations."""
        plt.figure(figsize=figsize)

        img = skimage.io.imread(image_path)
        if len(img.shape) > 2 and cmap != "rgb":
            img = img[..., 0]

        plt.imshow(img, cmap=None if cmap == "rgb" else cmap)
        plt.axis("off")

        if title:
            plt.title(title, pad=15, fontsize=12)

        if description:
            plt.figtext(
                0.5, 0.01, description, wrap=True, horizontalalignment="center", fontsize=10
            )

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
        plt.show()

    def _run(
        self,
        image_path: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        figsize: tuple = (10, 10),
        cmap: str = "rgb",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            output = {"image_path": image_path}
            metadata = {
                "image_path": image_path,
                "title": bool(title),
                "description": bool(description),
                "figsize": figsize,
                "cmap": cmap,
                "analysis_status": "completed",
            }
            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {
                    "image_path": image_path,
                    "visualization_status": "failed",
                    "note": "An error occurred during image visualization",
                },
            )

    async def _arun(
        self,
        image_path: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        figsize: tuple = (10, 10),
        cmap: str = "rgb",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[dict, dict]:
        """Async version of _run."""
        return self._run(image_path, title, description, figsize, cmap)
```

- [ ] **Step 3: 导出 `eye_doctor/tools/__init__.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/tools/__init__.py`：

```python
"""Tools for the eye_doctor Agent."""

from .image_visualizer import ImageVisualizerTool, ImageVisualizerInput
```

- [ ] **Step 4: 用临时 PNG 跑最小冒烟**

```bash
python -c "
from eye_doctor.tools import ImageVisualizerTool
from PIL import Image
import tempfile, os

tool = ImageVisualizerTool()

# 错误路径：文件不存在
out, meta = tool.invoke({'image_path': '/nonexistent.png'})
assert 'error' in out, out
assert meta.get('visualization_status') == 'failed', meta

# 正常路径：使用临时 PNG
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    Image.new('RGB', (8, 8), color=(255, 0, 0)).save(f.name)
    out, meta = tool.invoke({'image_path': f.name, 'title': 'demo', 'description': 'red square'})
    assert out.get('image_path') == f.name, out
    assert meta.get('analysis_status') == 'completed', meta
    assert meta.get('title') is True
    assert meta.get('description') is True
    os.unlink(f.name)

print('image_visualizer smoke OK')
"
```

Expected: 打印 `image_visualizer smoke OK`。

- [ ] **Step 5: Commit**

```bash
git add eye_doctor/tools/image_visualizer.py eye_doctor/tools/__init__.py
git commit -m "feat: add ImageVisualizerTool"
```

---

### Task 8: eye_doctor/agent/agent.py

**Files:**
- Create: `eye_doctor/agent/agent.py`
- Modify: `eye_doctor/agent/__init__.py`

- [ ] **Step 1: 验证 Agent 尚未导出**

```bash
python -c "from eye_doctor.agent import Agent"
```

Expected: `ImportError`。

- [ ] **Step 2: 写入 `eye_doctor/agent/agent.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/agent/agent.py`：

```python
"""Agent — LangGraph workflow that processes requests and executes tools.

Originally authored in MedRAX (https://github.com/bowang-lab/MedRAX). Migrated
verbatim to eye_doctor with no behavioural changes.
"""

import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """A log entry for a single tool call."""

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """Agent state — the only field is the message list with operator.add semantics."""

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """An agent that processes requests and executes tools based on LLM responses."""

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
```

- [ ] **Step 3: 导出 `eye_doctor/agent/__init__.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/eye_doctor/agent/__init__.py`：

```python
from .agent import Agent, AgentState, ToolCallLog
```

- [ ] **Step 4: 验证 Agent 类可构造（用假 LLM）**

```bash
python -c "
from eye_doctor.agent import Agent, AgentState, ToolCallLog
from eye_doctor.tools import ImageVisualizerTool

class FakeModel:
    def bind_tools(self, tools): return self
    def invoke(self, msgs): return msgs[-1] if msgs else None

agent = Agent(model=FakeModel(), tools=[ImageVisualizerTool()], system_prompt='hi', checkpointer=None, log_tools=False)
assert hasattr(agent, 'workflow'), 'workflow must be compiled'
assert 'image_visualizer' in agent.tools, list(agent.tools)
assert agent.system_prompt == 'hi'
print('agent smoke OK')
"
```

Expected: 打印 `agent smoke OK`。注意：未传真实 checkpointer 但 StateGraph 编译需要它为 None 或 MemorySaver——`checkpointer=None` 是合法值。

- [ ] **Step 5: Commit**

```bash
git add eye_doctor/agent/agent.py eye_doctor/agent/__init__.py
git commit -m "feat: add Agent (LangGraph workflow)"
```

---

### Task 9: interface.py（Gradio 前端）

**Files:**
- Create: `interface.py`

- [ ] **Step 1: 验证 interface 尚未导出**

```bash
python -c "from interface import create_demo, ChatInterface"
```

Expected: `ImportError`。

- [ ] **Step 2: 写入 `interface.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/interface.py`：

```python
import base64
import re
import shutil
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import gradio as gr
from gradio import ChatMessage

from eye_doctor.utils import load_prompts_from_file  # noqa: F401  (kept for future use)


class ChatInterface:
    """A chat interface for interacting with an ophthalmology AI agent through Gradio."""

    def __init__(self, agent, tools_dict):
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        self.original_file_path = None
        self.display_file_path = None

    def handle_upload(self, file_path: str) -> Optional[str]:
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)
        self.original_file_path = str(saved_path)

        # DICOM processing is intentionally not wired up in eye_doctor
        # (no DicomProcessorTool in scope). If a .dcm is uploaded, we still
        # save the original and let the frontend show the raw path.
        self.display_file_path = str(saved_path)
        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self,
        message: str,
        display_image: Optional[str],
        chat_history: List[ChatMessage],
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        chat_history = chat_history or []

        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self.original_file_path or display_image

        if image_path is not None:
            messages.append({"role": "user", "content": f"image_path: {image_path}"})
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        }
                    ],
                }
            )

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self.display_file_path, ""

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            tool_result = eval(message.content)[0]

                            if tool_result:
                                metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                formatted_result = " ".join(
                                    line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    )
                                )

                            if tool_name == "image_visualizer":
                                self.display_file_path = tool_result["image_path"]
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content={"path": self.display_file_path},
                                    )
                                )

                            yield chat_history, self.display_file_path, ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path


def create_demo(agent, tools_dict):
    """Create the Gradio Blocks demo for the ophthalmology agent."""

    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 👁️ eye_doctor
            Ophthalmology Diagnosis Agent
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the ophthalmic image...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=700, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload Fundus",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload OCT",
                            file_types=["image"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file):
            return interface.handle_upload(file.name)

        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)
        dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
```

- [ ] **Step 3: 用冒烟测试验证 create_demo 产物**

```bash
python -c "
import gradio as gr
from interface import create_demo, ChatInterface

class FakeModel:
    def bind_tools(self, tools): return self
    def invoke(self, msgs): return None

from eye_doctor.agent import Agent
from eye_doctor.tools import ImageVisualizerTool

agent = Agent(model=FakeModel(), tools=[ImageVisualizerTool()], log_tools=False, checkpointer=None)
demo = create_demo(agent, {'ImageVisualizerTool': ImageVisualizerTool()})
assert isinstance(demo, gr.Blocks)
print('interface smoke OK')
"
```

Expected: 打印 `interface smoke OK`。注意：Gradio 在某些环境下默认会立即 launch；我们没有调用 `.launch()`，所以不会。

- [ ] **Step 4: Commit**

```bash
git add interface.py
git commit -m "feat: add Gradio interface for eye_doctor agent"
```

---

### Task 10: main.py（应用入口）

**Files:**
- Create: `main.py`

- [ ] **Step 1: 验证 main 尚未运行**

```bash
python main.py --help 2>&1 | head -5
```

Expected: 由于 main.py 不存在，会出现 `No such file or directory` 或 `Can't find adapter` 之类的报错。

- [ ] **Step 2: 写入 `main.py`**

写入 `/Users/m11003005586/www/py_src/eye_doctor/main.py`：

```python
"""eye_doctor — entry point.

Initialises the Agent with the ImageVisualizerTool and launches the Gradio demo.
Requires an OpenAI-compatible API key supplied via OPENAI_API_KEY in the
environment (or in a local .env file).
"""

import os
import warnings
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from interface import create_demo
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
) -> Tuple[Agent, Dict[str, Any]]:
    """Initialise the eye_doctor agent with the ImageVisualizerTool.

    Args:
        prompt_file (str): Path to the system prompts file.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.
        top_p (float): Sampling top-p.
        openai_kwargs (dict, optional): Extra kwargs forwarded to ``ChatOpenAI``
            (e.g. ``api_key``, ``base_url``). If ``None``, these are collected
            from the ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` environment
            variables.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialised agent and tool dictionary.
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    tools_dict = {
        "ImageVisualizerTool": ImageVisualizerTool(),
    }

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
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    print("Starting server...")
    model = os.getenv("EYE_DOCTOR_MODEL", "gpt-4o")
    agent, tools_dict = initialize_agent(
        prompt_file="eye_doctor/docs/system_prompts.txt",
        model=model,
        temperature=0.7,
        top_p=0.95,
    )
    demo = create_demo(agent, tools_dict)
    demo.launch(server_name="0.0.0.0", server_port=8585, share=False)
```

- [ ] **Step 3: 验证 import 不报错（不启动 Gradio）**

```bash
python -c "import main; print(main.initialize_agent.__doc__ is not None)"
```

Expected: 打印 `True`。**这步不会启动 Gradio**，因为 `__name__ == '__main__'` 在 import 时为 `'main'`。

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add main.py entry point with simplified tool registry"
```

---

### Task 11: 安装依赖并跑端到端冒烟

**Files:** 无新增（仅运行命令）

- [ ] **Step 1: 安装包**

```bash
pip install -e .
```

Expected: 退出码 0；确认未安装 `torchxrayvision`、`transformers`、`pydicom`。

```bash
pip list | grep -iE "torchxrayvision|transformers|pydicom"
```

Expected: 该命令输出为空（nothing matched）。

- [ ] **Step 2: 全量导入冒烟**

```bash
python -c "
import eye_doctor
from eye_doctor.agent import Agent, AgentState, ToolCallLog
from eye_doctor.tools import ImageVisualizerTool
from eye_doctor.utils import load_prompts_from_file, load_tool_prompts
from interface import create_demo, ChatInterface
print('all imports OK')
"
```

Expected: 打印 `all imports OK`。

- [ ] **Step 3: 启动冒烟（无 API key 应能起服务，发消息时失败）**

```bash
OPENAI_API_KEY="" timeout 5 python main.py 2>&1 | head -30 || true
```

Expected: 服务能在 5 秒内启动；超时后被 kill；输出中包含 `Running on local URL`。

- [ ] **Step 4: 不需要 commit（验证步骤）**

---

### Task 12: 更新 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 写入新 README**

写入 `/Users/m11003005586/www/py_src/eye_doctor/README.md`：

````markdown
# eye_doctor

Ophthalmology Diagnosis Agent — built on a LangGraph-driven agent and a Gradio
chat UI. Forked from [MedRAX](https://github.com/bowang-lab/MedRAX) and trimmed
to focus on ophthalmology image analysis with a single, model-free tool
(`ImageVisualizerTool`).

## What this project contains

- `eye_doctor/agent/` — `Agent` class: LangGraph `process` → `execute` workflow
  with thread-based checkpointing.
- `eye_doctor/tools/image_visualizer.py` — `ImageVisualizerTool`, a
  `langchain_core.tools.BaseTool` that returns image metadata.
- `eye_doctor/utils/utils.py` — `load_prompts_from_file`,
  `load_tool_prompts`.
- `interface.py` — Gradio `ChatInterface` and `create_demo`.
- `main.py` — entry point: `initialize_agent(...)` then `demo.launch(...)`.

## What it does NOT contain (deliberately)

This repo does **not** include the eight medical ML tools that ship with
MedRAX (chest X-ray classification, segmentation, VQA, report generation,
phrase grounding, generation, Llava-Med, DICOM processing). To stay lean, we
ship only `ImageVisualizerTool`. Pulling the others in would force a `torch` /
`transformers` / `torchxrayvision` / `pydicom` install.

## Install

Python 3.10 ≤ x < 3.15 (verified on 3.14).

```bash
pip install -e .
cp .env.example .env
# edit .env to set OPENAI_API_KEY (and optionally OPENAI_BASE_URL)
```

## Run

```bash
python main.py
```

Gradio listens on `http://0.0.0.0:8585`.

## Known limitations

- Uploading a `.dcm` file: the file is preserved on disk, but the browser
  cannot preview DICOM natively and there is no `DicomProcessorTool` to convert
  it on-the-fly. Use JPG/PNG fundus or OCT snapshots.
- The Gradio avatar image (`assets/medrax_logo.jpg`) is reused from MedRAX;
  replace it with your own if you rebrand.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README to describe eye_doctor scope and limits"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement | Task |
|---|---|
| Migrate Agent class | T8 |
| Migrate ImageVisualizerTool | T7 |
| Migrate `load_prompts_from_file` + `load_tool_prompts` | T5 |
| Migrate ChatInterface + create_demo, ophthalmology theme, no DICOM branch | T9 |
| Migrate main.py with simplified tool registry | T10 |
| Migrate system prompts with ophthalmology theme | T6 |
| Migrate medrax_logo.jpg | T4 |
| pyproject.toml with minimal deps | T3 |
| .env.example | T3 |
| .gitignore | T2 |
| Install / import / no-key startup smoke | T11 |
| README updated with scope & limits | T12 |

**2. Placeholder scan**: No "TBD/TODO/implement later" — every step has full code or commands.

**3. Type consistency**:
- `Agent.__init__(model, tools, checkpointer=None, system_prompt="", log_tools=True, log_dir="logs")` — consistent across T8 definition and T11 fake-model smoke.
- `ImageVisualizerTool._run(...)` returns `(dict, dict)` — T7 implementation matches the interface.py `eval(message.content)[0]` usage in T9.
- `load_prompts_from_file` reads `[NAME]` markers — T6 prompts written with correct syntax and verified by T5's smoke test pattern adapted in T6.
- `tools_dict["ImageVisualizerTool"]` — used in T10 default and referenced from `interface.py` only as a parameter (not dereferenced after DICOM removal).
- Port `8585` — used consistently in T10 and README.

**4. Out-of-plan cleanup**: T9 deletes the DICOM branch and `tools_dict["DicomProcessorTool"]` lookup; T10's `tools_dict` no longer contains DicomProcessorTool — no dangling references remain.
