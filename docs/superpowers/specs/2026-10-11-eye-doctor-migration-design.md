# eye_doctor — MedRAX 迁移设计文档

**日期**：2026-10-11  
**范围**：将 MedRAX 的 Agent 框架与 Gradio 前端迁移到 eye_doctor 项目  
**非范围**：不迁移 8 个医学 ML 工具（见「范围说明」）

---

## 1. 背景与目标

`eye_doctor` 是一个全新的 Python 项目（当前只有 README.md 和 `.git/`、`.idea/`），
README 中说明目标是"通过眼科图像识别病因并给出医疗建议"。需要引入 MedRAX
（位于 `../MedRAX`）中的两类能力：

1. **Agent 框架**——基于 LangGraph 的 process / execute 工作流，能调用工具、与 LLM 对话。
2. **Gradio 前端**——`ChatInterface` 和 `create_demo`，提供图像上传 + 流式对话 UI。

迁移后 eye_doctor 可以独立 `pip install -e . && python main.py` 启动，
**不引入** MedRAX 中的 9 个医学工具所依赖的 torchxrayvision / transformers / pydicom 等重型库。

---

## 2. 范围说明（保留 / 不保留）

### 保留并迁移

| 模块 | 来源 | 处理 |
|---|---|---|
| `Agent` 类（含 `AgentState`、`ToolCallLog`） | `MedRAX/medrax/agent/agent.py` | 原样迁移 |
| `ImageVisualizerTool` | `MedRAX/medrax/tools/utils.py` 中定义 | 原样迁移到一个独立文件 |
| `load_prompts_from_file`、`load_tool_prompts` | `MedRAX/medrax/utils/utils.py` | 原样迁移 |
| `interface.py` 中的 `ChatInterface` 和 `create_demo` | `MedRAX/interface.py` | 迁移并替换眼科主题文案 |
| `main.py` 中的 `initialize_agent` 入口 | `MedRAX/main.py` | 迁移，**移除 8 个工具的初始化分支**，只注册 ImageVisualizerTool |
| `system_prompts.txt` | `MedRAX/medrax/docs/system_prompts.txt` | 文本改写：MEDICAL_ASSISTANT 改为眼科图像（OCT、眼底图、视网膜图像） |

### 不迁移

- `ChestXRayClassifierTool`、`ChestXRaySegmentationTool`、`LlavaMedTool`
  `XRayVQATool`、`ChestXRayReportGeneratorTool`、`XRayPhraseGroundingTool`
  `ChestXRayGeneratorTool`、`DicomProcessorTool` 八个工具类（依赖重型 ML 库）。
- `medrax/llava/` 目录（含模型加载逻辑）。
- `benchmark/`、`experiments/`、`data/`、`demo/`、`quickstart.py`。
- `assets/medrax_logo.jpg` 单独看待，见第 4.5 节备注。

---

## 3. 目标结构

```
eye_doctor/
├── README.md                              已存在
├── main.py                                新建（迁移自 MedRAX main.py）
├── interface.py                           新建（迁移自 MedRAX interface.py）
├── eye_doctor/                            新建 Python 包（替代原 medrax 命名空间）
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py                       来自 MedRAX/medrax/agent/agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── image_visualizer.py            来自 MedRAX/medrax/tools/utils.py（仅工具部分）
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py                       来自 MedRAX/medrax/utils/utils.py
│   └── docs/
│       └── system_prompts.txt             改写为眼科主题
├── assets/                                新建（仅 logo）
│   └── medrax_logo.jpg                    来自 MedRAX/assets/medrax_logo.jpg
├── pyproject.toml                         新建
├── .env.example                           新建
├── .gitignore                             新建
└── docs/                                  已创建（本次规格文档所在）
```

---

## 4. 模块设计

### 4.1 `eye_doctor/agent/agent.py`

完整保留 MedRAX 的 `Agent` 实现：

- `AgentState`：`TypedDict`，含 `messages: Annotated[List[AnyMessage], operator.add]`
- `ToolCallLog`：`TypedDict`，五个字段
- `Agent.__init__`：构造 LangGraph `StateGraph`，节点 `process` 与 `execute`，
  条件边 `has_tool_calls`，`compile(checkpointer=...)`。
- `process_request`：`model.invoke(messages)`，system prompt 在最前面注入。
- `has_tool_calls`：判断最后一条消息是否有 `tool_calls`。
- `execute_tools`：遍历 `tool_calls`，调用 `self.tools[name].invoke(args)`，
  把结果包装为 `ToolMessage` 写回 state。
- `_save_tool_calls`：把 tool 调用的元数据落盘到 `logs/tool_calls_<timestamp>.json`。

差异：仅 import 路径从 `medrax.agent` 改为 `eye_doctor.agent`，**无逻辑修改**。

### 4.2 `eye_doctor/tools/image_visualizer.py`

只搬 `ImageVisualizerInput` + `ImageVisualizerTool`。其余工具代码不出现于本文件。

`ImageVisualizerTool._run` 行为保持不变：
- 校验 `image_path` 文件存在
- 返回 `({"image_path": ...}, metadata_dict)` 双元组
- 异常路径返回 `({"error": ...}, {"visualization_status": "failed", ...})`

### 4.3 `eye_doctor/utils/utils.py`

迁移 `load_prompts_from_file` 和 `load_tool_prompts` 两个函数。
`load_system_prompt` 暂时不需要（多工具时才需要动态拼装工具说明，目前只有一个工具
硬编码到 `system_prompts.txt` 即可），**不迁移以减少死代码**。

`load_tool_prompts` 读取 `tools.json` 中的工具元信息。本次迁移**不创建**该 JSON
文件，因为只有一个工具，硬编码进 `system_prompts.txt` 已经足够。
`load_tool_prompts` 作为扩展点保留，便于后续添加工具时直接使用。

### 4.4 `eye_doctor/docs/system_prompts.txt`

```
[MEDICAL_ASSISTANT]
You are an expert ophthalmology AI assistant who can analyze ophthalmic images
(fundus photographs, OCT scans, retinal images) similarly to an ophthalmologist.
Solve using your own visual reasoning and use tools to complement your reasoning.
Make multiple tool calls in parallel or sequence as needed.
Critically evaluate any tool outputs before forming an answer.

[GENERAL_ASSISTANT]
You are a helpful AI assistant. Your role is to assist users with a wide range
of tasks and questions, providing accurate and useful information.
```

### 4.5 `interface.py`

保留 `ChatInterface` 的全部行为，仅做以下替换：

| 项 | MedRAX 原文 | eye_doctor 替换 |
|---|---|---|
| Markdown 标题 | `# 🏥 MedRAX\nMedical Reasoning Agent for Chest X-ray` | `# 👁️ eye_doctor\nOphthalmology Diagnosis Agent` |
| Textbox placeholder | `Ask about the X-ray...` | `Ask about the ophthalmic image...` |
| 上传按钮 | `📎 Upload X-Ray` / `📄 Upload DICOM` | `📎 Upload Fundus` / `📄 Upload OCT` |
| Avatar image path | `assets/medrax_logo.jpg`（MedRAX 原文件） | 暂时迁过来，作为 `assets/medrax_logo.jpg` 在 eye_doctor 中复用 |
| 工具键引用 | `tools_dict["DicomProcessorTool"]` | 删除该分支（眼科场景下 DICOM 不是必须） |

> **保留** `DicomProcessorTool` 的 DICOM 转换分支对应的代码会被删除：
> 当 `suffix == ".dcm"` 时使用 `tools_dict["DicomProcessorTool"]`，迁移后
> 该工具不存在，所以简化为：`.dcm` 文件仍走相同目录复制，但 `display_file_path = original`，
> 不调用工具；若用户上传 DICOM，前端显示原路径（Gradio 不可读 DICOM，由用户自决）。
> 这是文档化的退化行为，README 中会注明。

### 4.6 `main.py`

迁移 `initialize_agent` 函数。`all_tools` 字典从 MedRAX 的 9 条记录精简为 1 条：

```python
all_tools = {
    "ImageVisualizerTool": lambda: ImageVisualizerTool(),
}
```

入口默认 `selected_tools = ["ImageVisualizerTool"]`，device / model_dir / temp_dir
等参数保留可配置但**默认值改为 `cpu` / 本地路径**，因为：
- 没有 GPU 模型加载
- `/model-weights` 在普通用户机器上不存在

调用 `ChatOpenAI(model=model, temperature=temperature, ...)` 时 `openai_kwargs` 从
环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 收集（与 MedRAX 一致）。

`Agent` 构造参数：`log_tools=True`、`log_dir="logs"`、`system_prompt=prompt`、
`checkpointer=MemorySaver()`，与 MedRAX 一致。

### 4.7 `pyproject.toml`

```toml
[project]
name = "eye-doctor"
version = "0.1.0"
description = "Ophthalmology diagnosis agent"
requires-python = ">=3.10,<3.15"     # 用户已确认可以使用 3.14
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

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["eye_doctor*"]
exclude = ["tests*", "docs*"]
```

### 4.8 `.env.example`

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 5. 数据流

```
用户上传图片 (Gradio UploadButton)
   ↓
ChatInterface.handle_upload(file)
   ├─ 拷贝到 temp/upload_<ts>.<ext>
   └─ 记录 self.original_file_path / self.display_file_path
   ↓
用户输入文本 → add_message 写入 chat_history → 提交
   ↓
process_message 异步生成器
   ├─ 把图片 base64 编码后塞进 messages（多模态）
   ├─ agent.workflow.stream({"messages": ...}, {"configurable": {"thread_id": ...}})
   │     ├─ process 节点 → model.invoke(messages) → ChatOpenAI.bind_tools(tools)
   │     ├─ has_tool_calls → 真：execute；假：END
   │     └─ execute 节点 → 逐个调用工具 → ToolMessage → 回 process
   └─ 流式 yield 给 Gradio Chatbot + image_display + textbox
   ↓
用户看到对话 + 工具返回的图像元数据（image_visualizer 工具 metadata）
```

---

## 6. 错误处理

| 场景 | 行为 |
|---|---|
| 上传空文件 | `handle_upload` 返回 `None`，前端不显示图 |
| 缺失 OPENAI_API_KEY | `ChatOpenAI` 构造时即报错；`main.py` 不捕获，让用户立即看到错误 |
| 工具调用异常 | `execute_tools` 中调用 `invoke` 时抛出，**会上抛到 Gradio**；前端 `process_message` 的 `try/except` 捕获后以 `❌ Error: ...` 显示 |
| 工具名无效（LLM hallucinate 了一个未注册的 tool） | MedRAX 已有处理：`result = "invalid tool, please retry"`，**保留** |
| LangGraph 状态 | `MemorySaver()` 在进程内持久化，`thread_id` 切换时（New Thread 按钮）刷新 |

---

## 7. 测试 / 验证

迁移完成必须通过下列检查：

1. **安装**
   - `pip install -e .` 不报缺包（不安装 transformers / torchxrayvision / pydicom）
2. **导入**
   - `python -c "import eye_doctor; from eye_doctor.agent import Agent, AgentState; from eye_doctor.tools.image_visualizer import ImageVisualizerTool; from eye_doctor.utils.utils import load_prompts_from_file; from interface import create_demo, ChatInterface"`
3. **空 key 启动**
   - `unset OPENAI_API_KEY && python main.py`：Gradio 启动成功；
     发送消息时立即报错（合理预期，未配 key 不应静默）
4. **带 key 联通（手工验证）**
   - 配置 `OPENAI_API_KEY` 后启动，发"显示这个图像"，ImageVisualizer 返回元数据
5. **日志**
   - `logs/tool_calls_*.json` 在执行工具后生成

不引入自动化测试套件（MedRAX 项目本身也没有 pytest 套件，遵循现状）。

---

## 8. 风险与限制

- **ImageVisualizer 工具输出**：MedRAX 中它的 `_run` 实际**没把图像渲染到屏幕**
  （`plt.show()` 被注释掉），只返回路径与 metadata。迁移后保持该行为，**图像在前端通过
  `image_display` gr.Image 组件显示**，由 gradio 直接读取路径。
- **LangGraph 版本**：0.2 起步，行为与 MedRAX 当前用法兼容（`add_node` / `add_edge`
  / `add_conditional_edges` / `compile` API 稳定）。
- **DICOM 用户**：迁移版不内置 DICOM 处理。若上传 `.dcm`，前端会显示原路径但
  浏览器不能预览。README 中标注为限制。
- **avatar 图片**：暂时复用 MedRAX 的 `assets/medrax_logo.jpg`，后续如有版权 / 品牌诉求再替换为 eye_doctor 自有 logo。

---

## 9. 不在本范围内（明确划出）

- 不实现 mock LLM（用户决定保留 ChatOpenAI 与真实链路）
- 不实现图像分类 / 分割 / VQA 等医学能力
- 不接入 LangSmith / LangFuse 等观测平台
- 不引入数据库持久化（用 `MemorySaver` 内存 checkpoint）
- 不写自动化测试或 CI
