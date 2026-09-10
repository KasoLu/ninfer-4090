# serve 中的工具调用解析（tool-call）

本文基于仓库代码，完整分析 HTTP 服务层（`src/serve`，即 `ninfer-serve` 应用）的
parser 逻辑，重点是工具调用（tool call）部分：各协议如何在请求侧解析工具定义与
工具调用历史，引擎输出如何在输出侧被解析为结构化工具调用，以及各协议如何将其渲染
回线上格式。所有结论均附带文件位置引用。

## 1. 范围：请求侧与输出侧两层解析

工具调用处理被拆分为**两个独立的解析层**：

1. **请求侧 —— `src/serve` 中的协议 parser。** 一个 HTTP 监听器上服务三种线上协议：
   OpenAI Chat Completions、OpenAI Responses、Anthropic Messages。每个协议 parser
   把各自的工具定义、tool_choice 模式、工具调用历史归一化为一个线上传输中立的请求模型
   `ninfer::serve::GenerationRequest`（`src/serve/request.h`）。引擎从不接触协议相关的
   工具语法：它只看到渲染后的文本，加上每个工具一份的 JSON 列表（`options.tool_jsons`）。

2. **输出侧 —— 目标前端 parser。** 模型以纯 XML 文本（`<tool_call>…</tool_call>`，见 §4）
   发出工具调用。把该文本解析成 `name` + `arguments_json` 结构体的逻辑位于 Qwen 目标前端：
   `src/targets/qwen3_6/impl/frontend/tool_call_parser.{h,cpp}`
   （命名空间 `ninfer::targets::qwen3_6::frontend_internal`）。一次性 parser
   （`parse_qwen_tool_call_output`）与流式过滤器（`ToolCallOutputDecoder`）共享同一套规则。
   引擎请求生命周期从输出会话中取走解析结果（`src/runtime/engine/engine_core.h:1035`，
   `complete_success` 中 `result.tool_calls = request->output.take_tool_calls()`），
   随后 `src/serve` 中的协议**响应**渲染器把结构化调用映射回各自的线上格式。

```
协议请求 parser                                  协议响应渲染器
 openai_chat_request.cpp                          openai_chat_response.cpp
 openai_responses_request.cpp                     openai_responses_response.cpp
 anthropic_messages_request.cpp                   anthropic_messages_response.cpp
          │  (校验 + 归一化)                              ▲  (id、字段形状、结束/停止原因)
          ▼                                              │
      GenerationRequest (src/serve/request.h)            │
          │                                              │
          ▼                                              │
      translate.cpp: to_prompt_input / to_request_options
        options.tool_jsons (每个工具一份 OpenAI 形式 JSON)
        output.preserve_special_tokens / tool_name_max_length
          │
          ▼
      GenerationService (src/serve/generation_service.cpp)
        prepare → ninfer::Engine (PreparedPromptData 携带输出契约)
          │
          ▼
      Qwen 前端 (src/targets/qwen3_6/impl/frontend/frontend.cpp)
        OutputSession + ToolCallOutputDecoder (tool_call_parser.cpp)
        文本流 → 可见文本增量 + 结构化 GeneratedToolCall 列表
          │
          ▼
      GenerationOutcome { text, reasoning, tool_calls, finish_reason, … }
          │
          ▼
      ServiceOutputSink (只流式发送 Reasoning/Content 增量)
```

### 关键文件

| 职责 | 文件 |
|---|---|
| 线上传输中立请求模型 | `src/serve/request.h` |
| 共享校验 / 错误辅助 | `src/serve/request_validation.cpp` |
| 翻译为引擎输入 / 选项 | `src/serve/translate.cpp` |
| 服务层（prepare/run、sink、outcome） | `src/serve/generation_service.{h,cpp}` |
| ID 生成 | `src/serve/openai_common.cpp` |
| OpenAI Chat 请求 / 响应 | `src/serve/openai_chat_request.cpp` / `src/serve/openai_chat_response.cpp` |
| OpenAI Responses 请求 / 响应 | `src/serve/openai_responses_request.cpp` / `src/serve/openai_responses_response.cpp` |
| Anthropic Messages 请求 / 响应 | `src/serve/anthropic_messages_request.cpp` / `src/serve/anthropic_messages_response.cpp` |
| Prompt 渲染（工具语法、历史） | `src/targets/qwen3_6/impl/frontend/chat_template.cpp` |
| 工具调用输出 parser + 流式解码器 | `src/targets/qwen3_6/impl/frontend/tool_call_parser.{h,cpp}` |
| 输出会话 / 解码器接线 | `src/targets/qwen3_6/impl/frontend/frontend.cpp` |
| 引擎取回解析结果 | `src/runtime/engine/engine_core.h:1035` |
| 行为测试 | `tests/test_tool_call_parser.cpp` |

## 2. 线上传输中立的请求模型（`src/serve/request.h`）

`GenerationRequest` 中与工具相关的字段：

- `messages` —— `std::vector<ChatTurn>`；每个 `ChatTurn` 携带角色、文本/部件；
  assistant 轮携带 `tool_calls`（`std::vector<ToolCall>`）；工具结果轮携带
  `tool_call_id`、`content`、`tool_result_is_error`、`cache_boundary_after`。
- `tools` —— `std::vector<ToolDefinition>`；每个 `ToolDefinition` 携带 `name`、
  可选 `description`、`input_schema_json`（原始 JSON Schema 对象）、
  可选 `input_examples_json`、`cache_boundary_after`。
- `tool_name_max_length` —— 请求模型中默认 **64**；Anthropic parser 会覆盖为 **128**
  （见 §6.1）。它约束输出 parser 对模型发出的函数名所应用的文法。
- `tool_choice` —— `ToolChoice`，模式只有 `Auto` 和 `None`。各线上协议更丰富的
  tool_choice 形式，要么被降维到这两个模式，要么直接拒绝（§5、§6、§7）。
- 其他字段：`stop_strings`、`stop_strings_apply_to_reasoning`、`max_tokens`、
  `enable_thinking`、`thinking_budget`、`reasoning_effort`、`preserve_thinking`、
  `continuation`、`SamplingParams`。
- `uses_tools()`：`tools` 非空且模式不是 `None` 时为真；`has_tool_history()`：
  `messages` 中存在工具轮。

引擎侧类型（`include/ninfer/types.h`）：

- `ToolCall`（315 行）：`id`、`name`、`arguments_json` —— 协议历史形状
  （id 归线上协议所有）。
- `GeneratedToolCall`（325 行）：`name` + `arguments_json`。这是前端 parser 产生的、
  模型起源的结构体；注释明确“任何线上层面的调用标识符归协议适配器所有”，
  即引擎从不分配 `call_xxx` 之类的 id。
- `OutputOptions`（286–293 行）：`raw`（默认 `false`）、`preserve_special_tokens`
  （默认 `false`）、`tool_name_max_length`（默认 `128`；实际会被 serve 层按请求
  覆盖 —— 它“只约束 Qwen 发出的函数名文法”，不要求名称与已声明工具匹配）。

`GenerationOutcome`（`src/serve/generation_service.h`）携带 `text`、`reasoning`、
`tool_calls`（`std::vector<GeneratedToolCall>`）、`finish_reason`、
`matched_stop_string`。

## 3. 翻译为引擎输入（`src/serve/translate.cpp`）

`to_prompt_input` 通过目标 chat template 渲染对话。当 `uses_tools()` 为真时，
`render_tool_definition` 把每个 `ToolDefinition` 序列化为**OpenAI 形式 JSON 字符串**，
追加进 `input.options.tool_jsons`：

```json
{"type":"function","function":{"name":"...","parameters":<input_schema_json>,"strict":false}}
```

可含可选的 `"description"` 与 `"input_examples"` 成员。这些字符串即前端用来
(a) 渲染 prompt 工具段、(b) 构建输出解析参数编码契约（§8.1）的输入。

`to_request_options` 的设置：

- `options.output.raw = false`（serve 一律走展示解码器）；
- `options.output.preserve_special_tokens = uses_tools() || has_tool_history()`
  （思考控制后缀等特殊 token 必须在解码后存活）；
- `options.output.tool_name_max_length = request.tool_name_max_length`。

`tool_result_is_error` 为真的工具结果轮，会在结果内容前插入文本部件
`[tool_error]` + 换行，让模型在上下文中看到显式的错误信号。

## 4. 工具调用的线上语法（prompt 侧）

`scripts/chat_template_v22_4.jinja` 声明了 `tool_call_format` 变量（`json` 或 `xml`），
但 C++ 渲染器**只实现了 XML 分支** —— `src/targets/qwen3_6/impl/frontend/chat_template.cpp:371-373`
带有明确注释：“json 分支刻意不实现，xml 分支无条件启用”。
`docs/cli.md:52` 对 CLI 也作同样说明。

输出格式为：

```
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

系统提示词指示模型（内嵌于 `chat_template.cpp:224-239` 与 `505-537`）：

- 函数块必须是嵌套在 `<tool_call></tool_call>` 内部的 `<function=…></function>`；
- `<tool_call>` 块必须**紧跟 thinking 之后输出，前面不得有任何对话文本**；
- `<tool_call>`/`<function>` 标签必须位于新行行首；
- 每个函数拥有自己一个完全闭合的 `<tool_call></tool_call>` 块，禁止嵌套。

assistant **历史**中的工具调用由 `render_tool_call`（`chat_template.cpp:277-300`）
重渲染为同样的 XML；空参数调用渲染为裸的 `<function=NAME>` 开标签 + 闭标签。
这些块插入位置在 `chat_template.cpp:863-871` 与 `1211-1221`。

Prompt 缓存交互：`Frontend::prepare`
（`src/targets/qwen3_6/impl/frontend/frontend.cpp:1362-1388`）从 `options.tool_jsons`
构建输出契约；当允许引擎自动共享前缀且存在工具时，额外注册一个
`PromptCacheMarker{SharedStablePrefix, EngineStructural, ToolBoundary,
after_tool_count=N}` —— 渲染后的工具定义之后的缓存边界，使长系统提示 + 工具前缀
可跨会话共享。

## 5. OpenAI Chat Completions 解析（`src/serve/openai_chat_request.cpp`）

### 5.1 共享校验

`src/serve/request_validation.cpp`：

- `valid_tool_name(name, max)`：非空、`name.size() <= max`、每个字符为字母数字
  或 `_` 或 `-`。
- `bad_request(message, param, code)` → 抛出 `ApiException`，状态 **400**、
  类型 `invalid_request_error`，并附带协议错误 `code`。

### 5.2 `parse_tools`（589–637 行）

- 每个工具必须是 `type` 恰好为 `"function"` 的对象，否则
  `400 tool_type_not_supported`。
- `function.name` 必须匹配 `[A-Za-z0-9_-]{1,64}`（上限 64）。
- `function.description` 可选。
- `function.parameters` 可选对象；缺省时默认
  `{"type":"object","properties":{}}`。
- `strict: true` 被拒绝：`400 strict_tools_not_supported`。

### 5.3 `parse_tool_choice`（693–739 行）与 `allowed_tools`（639–690 行）

| 请求形式 | 结果 |
|---|---|
| `"auto"` / `"none"` | 接受（模式 `Auto` / `None`） |
| `"required"` | `400 tool_choice_not_supported` |
| 对象 `{"type":"function",…}` | `400 tool_choice_not_supported`（无法保证一定发起该调用） |
| 对象 `{"type":"allowed_tools","tools":[{type,name}…]}` | 模式必须为 `"auto"`（`"required"` → 400）；条目必须为 `type:"function"` 且名称已在 `tools` 中声明（否则 400）；`generation.tools` 被过滤到所选名称；模式 `Auto` |
| 其他对象类型（如 `"custom"`） | `400` |

### 5.4 `parse_parallel_tool_calls`

在 `uses_tools()` 为真时取 `false` → `400 parallel_tool_calls_not_supported`
（无工具时为中性，不报错）。

### 5.5 消息历史

- `parse_assistant_tool_calls`：assistant 的 `tool_calls` 条目需要字符串 `id`、
  `type:"function"`、`function` 中的**字符串** `arguments` 以及合法名称。
  旧版 assistant `function_call` 对象被接受并降维为 `id` 为空的 `ToolCall`。
- `role:"function"` → `ChatRole::Tool`。
- 工具消息（`role:"tool"`）必须带 `tool_call_id` + `content`，且禁止携带
  `tool_calls`。工具消息上的 `name` 字段被接受但**忽略**（兼容提示，
  永远不会到达引擎）。
- 直接拒绝：旧版顶层 `functions` / 强制 `function_call`、非零 `logit_bias`、
  `logprobs`、非文本 `response_format`、`grammar`/`structured_outputs`/`guided_*`
  约束解码字段、`store: true`。

`parse_stop`：至多 4 个字符串；设置 `stop_strings_apply_to_reasoning = true`。

### 5.6 编排顺序

`parse_chat_completion_request` 依次执行：对象检查 → 标准输出控制 → 约束解码拒绝项
→ 兼容提示 → `model` → OpenAI prompt 缓存策略 → `parse_tools` → `parse_tool_choice`
→ `parse_parallel_tool_calls` → `parse_messages` → `parse_stop` → 采样 →
流式选项（`stream` + `stream_options.include_usage`）→ 输出上限
（`max_completion_tokens` 或 `max_tokens`，否则默认 **8192**）→ 推理强度 →
模板选项（`enable_thinking`、`preserve_thinking`、`reasoning_effort`，来自顶层与
`chat_template_kwargs`，冲突 → 400）→ 应用 prompt 缓存策略。

## 6. Anthropic Messages 解析（`src/serve/anthropic_messages_request.cpp`）

### 6.1 名称上限

`kMaxToolNameLength = 128`（第 22 行；模式 `[A-Za-z0-9_-]{1,128}`）。
`tool_name_max_length = 128` 在 messages 端点与 count-tokens 端点的生成请求上
均被设置（1043、1071 行）—— 因此该协议下输出 parser 接受 128 字符的函数名。

### 6.2 tool_use / tool_result 块

- `parse_tool_use`（assistant 的 `tool_use`）：要求 `id`、`name` 以及**对象**
  类型的 `input`；`arguments_json = input.dump()`。
- `parse_tool_result`（user 的 `tool_result`）：`tool_use_id` + `content`
  （字符串或 `text`/`image` 块数组，否则 `400 content_block_not_supported`）、
  可选 `is_error` 布尔、可选 `cache_control`。

`normalize_tool_history` 强制严格的对话形状：

- `tool_result` 块必须位于同一 user 消息中所有 `text`/`image` 块**之前**；
- 携带可见 `tool_use` 的 assistant 消息必须**紧接着**是携带结果的 user 消息；
- 每个可见 `tool_use` id 必须在那条 user 消息中有且仅有一个 `tool_result`
  —— 未知 id → `400 invalid_tool_history`（“unknown tool_result id … after
  visible assistant tool_use”），缺失 → `400`（“missing tool_result for
  visible tool_use id …”），两个方向都拒绝重复；首条 user 消息允许携带
  未匹配的 result；
- 结果按原始调用顺序重排，`lower_messages` 把每个结果降维为
  `ChatRole::Tool` 轮 `{tool_call_id, content, tool_result_is_error,
  cache_boundary_after}`。

### 6.3 工具定义（`parse_tool_definitions`）

可选 `type` 字段选择 `ToolSource`：`"custom"` → `UserDefined`、`"toolset"` →
`Toolset`、其他 → `AnthropicProvided`。`UserDefined` 工具要求 1–128 字符的名称
（重名 → 400 “duplicate tool name”）与对象类型 `input_schema`；可选
`description`、`input_examples` 数组、`cache_control`。

### 6.4 `lower_tools` —— tool_choice 与特性拒绝矩阵

| 特性 | 结果 |
|---|---|
| `tool_choice` `auto` / `none` | 接受（模式 `Auto`/`None`） |
| `tool_choice` `tool`（按名指定） | `400 tool_choice_not_supported`（先做未知名称检查） |
| `tool_choice` `any` | `400` |
| toolset 工具 | `400 toolsets_not_supported` |
| Anthropic 托管工具 | `400 anthropic_tools_not_supported` |
| `strict: true` | `400` |
| `defer_loading: true` | `400 deferred_tools_not_supported` |
| `allowed_callers` 不含 `"direct"` | `400 tool_caller_not_supported` |
| 带工具时 `disable_parallel_tool_use: true` | `400 parallel_tool_use_not_supported` |

当降维后模式为 `None` 时，只有 `UserDefined` 工具会被转发。

### 6.5 Thinking 与其他限制

Thinking `type` ∈ `disabled` | `adaptive` | `enabled`；`enabled` 要求
`budget_tokens >= 1024` 且 `< max_tokens`（Messages 用途）。历史中的 thinking 块
必须通过签名校验器（否则 `400 invalid_thinking_signature`）；`redacted_thinking`
块被跳过。`max_tokens: 0` → `400 cache_prewarm_not_supported`。

## 7. OpenAI Responses 解析（`src/serve/openai_responses_request.cpp`）

### 7.1 函数标识与命名空间

`function_identity` = `name`（`[A-Za-z0-9_-]{1,64}`）+ 可选 `wire_namespace`
（同一模式）。`lower_function_identity` 把带命名空间的函数扁平化为引擎名
**`namespace__name`（双下划线）**；引擎名本身必须匹配 `[A-Za-z0-9_-]{1,64}`
（否则 `400 invalid_tool_name`）。`tool_identities` 映射表记录
`engine_name → identity`；两个不同标识冲突时 → `400 duplicate_tool_name`。
该映射保留在 `OpenAIResponsesCreateRequest`（`src/serve/openai_responses.h`）上，
供响应渲染器从引擎名还原线上 `namespace`。

### 7.2 `parse_function_tool`

允许成员：`type`、`name`、`description`、`parameters`、`strict`、
`allowed_callers`、`defer_loading`、`output_schema`。`parameters` 缺省为空对象
schema。拒绝项：`strict: true` → `400 strict_tools_not_supported`；
`defer_loading: true` → `400 deferred_tools_not_supported`；`allowed_callers`
不含 `"direct"` → `400 tool_caller_not_supported`；非空 `output_schema`
→ `400 tool_output_schema_not_supported`。命名空间的描述会被前缀到每个
成员的描述上。

`parse_tools`：条目为 `type:"function"` 或 `type:"namespace"`（namespace 携带
`name`（1–64）、可选 `description`、嵌套 `tools` 且全部必须为 `type:"function"`）；
其他类型 → `400 tool_type_not_supported`；引擎名重复 → `400 duplicate_tool_name`。
请求保留规范线上条目（`wire_tools`）。

### 7.3 `parse_tool_choice`

字符串 `auto`/`none`；`"required"` → `400`。对象形式必须为
`{"type":"allowed_tools",…}`（named 或 hosted → `400 tool_choice_not_supported`）；
其模式必须为 `auto`；条目是 `type:"function"`，成员仅限 `type`/`name`/`namespace`；
未声明名称 → `400 invalid_tool_choice`（错误信息中显示点分线上名，如
`ns.name`）。`generation.tools` 被过滤到所选引擎名。

### 7.4 历史条目

- `parse_function_call_item`（`function_call`）：`call_id` 非空必需；`name`/
  `namespace` 经 identities 映射降为引擎名；`arguments` 必须是能解码为 JSON
  **对象**的 JSON 字符串（否则 400，信息为 “function_call arguments must encode
  a JSON object”）；`status` 非 `completed` → `400 partial_tool_call_not_supported`；
  `caller` → `400 tool_relationship_not_supported`。
- `parse_function_call_output_item`（`function_call_output`）：`call_id`；`output`
  为字符串或 `input_text`/`input_image` 部件数组（`input_file` 被拒）。带
  `namespace` 时必须同时有 `name`（否则 `400 invalid_tool_history`）。断言的
  name 记为 `turn.tool_result_name`（已降维）；按 `src/serve/request.h` 的注释，
  调用图归一化会在**引擎看到历史之前**，用它与 `tool_call_id` 所指向的函数
  进行校验。
- `AssistantInputRun`（assistant 输入条目）：顺序必须为 reasoning → 消息内容
  → 函数调用；违反 → `400 invalid_assistant_history`。

### 7.5 顶层形状

`parse_openai_responses_create_request` 强制顶层字段白名单（未知字段 →
`400 unknown_parameter`），并拒绝平台字段 `conversation`、`prompt`、
`context_management`、`moderation`。带可用工具时 `parallel_tool_calls: false`
→ `400`；`max_tool_calls` 必须为非负整数，作为 hosted-tool 空操作保留；
`store` 默认 `true`；`metadata` 限 16 条、键 ≤ 64 字符、字符串值 ≤ 512 字符。
保留的请求模型含 `prompt`、`wire_tools`、`wire_tool_choice`、`tool_identities`、
`parallel_tool_calls`、`max_tool_calls`。

## 8. 工具调用输出解析（核心）

`src/targets/qwen3_6/impl/frontend/tool_call_parser.{h,cpp}`，命名空间
`ninfer::targets::qwen3_6::frontend_internal`。头文件用途注释
（`tool_call_parser.h:15-18`）：Qwen 语法把每个顶层参数作为 `parameter` 标签间的
文本承载；契约只记录显式 JSON Schema 类型是“接受字符串”还是“需要 JSON 解码”
——“有意不做完整 Schema 校验”；声明为非 string 的值若 JSON 解码失败，退化为
原始文本而不是拒绝调用。

### 8.1 契约类型（`tool_call_parser.h:19-50`）

```cpp
struct ToolArgumentTypeContracts {
    enum class Encoding : uint8_t { Json, String };
    struct Parameter { std::string name; Encoding encoding = Json; };
    struct Tool { std::string name; std::vector<Parameter> parameters;
                  bool unambiguous = true; };
    std::vector<Tool> tools;
    bool enforce_declared_names = false;
};
struct ToolCallOutputContract { ToolArgumentTypeContracts argument_types; };
struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
};
```

`build_tool_call_output_contract(tool_jsons, enabled)`
（`tool_call_parser.cpp:375-405`）：禁用时返回 `nullptr`；否则设置
`enforce_declared_names = true`，并对每个 `tool_json` 做
`compile_tool_contract`（146–168 行）—— 要求 OpenAI 形式：带字符串 `name` 的
`function` 对象 + 含 `properties` 对象的 `parameters` 对象。对每个属性，
`explicit_parameter_encoding`（120–145 行）决定 `Encoding`：

| 属性 schema 中的 `type` 值 | Encoding |
|---|---|
| `"string"` | `String` |
| `"integer"`、`"number"`、`"boolean"`、`"object"`、`"array"`、`"null"` | `Json` |
| schema 类型数组（所有成员均合法） | 任一成员为 `"string"` 则 `String`，否则 `Json` |
| 缺 `type`、类型不合规范、属性非对象 | **跳过** → 该参数回退到 legacy 推断 |

`append_tool_contract`（181–196 行）：同一工具名以首个契约为准；若**重名**工具
带着*不同*的参数契约到来（`same_contract`，169–180 行），则清空已有工具的参数
并置 `unambiguous = false` —— 该工具随后对每个参数都回退到 legacy 逐参数推断。
`find_parameter_contract`（197–209 行）仅在工具 `unambiguous` 时返回契约；
`declares_tool`（210–214 行）：除非 `enforce_declared_names` 且名称不在已声明
集合中，否则恒为真。

### 8.2 一次性解析：`parse_qwen_tool_call_output`（407–477 行）

```
文本中不含 "<tool_call>" 标记
    → 回退：content = 全文，is_tool_call_response = false

否则：
    content = 第一个 "<tool_call>" 之前的文本，右裁剪
    扫描块：
        块间非空白文本 → 追加到 content，继续扫描
        某个已开块缺少 "</tool_call>" → 该块原文追加到 content，扫描结束
        每块执行 parse_one_tool_call(...)：
            成功 → 追加结构化调用
            失败 → 该块原文（含标记）追加到 content，继续扫描下一块
    成功解析的调用数为 0 → 回退：content = 全文
    否则 is_tool_call_response = true（content 可能同时非空）
```

`parse_one_tool_call`（304–365 行）：

1. 跳过空白；块必须以 `<function=` 开头；
2. 函数名到下一个 `>` 为止，必须非空，通过 `valid_function_name`
   （107–113 行：非空、`<= max_tool_name_length`、字符限 `[A-Za-z0-9_-]`），
   且通过 `declares_tool`（调用**未声明**工具视为不合法 —— 已声明工具集被强制）；
3. 必须有 `</function>` 闭标签（缺失 → 不合法）；
4. 参数区（名称行与 `</function>` 之间）是 `skip_ws` + `parse_parameter` 的循环；
   最后一个参数之后的剩余必须是空白，`</function>` 之后的剩余也必须是空白 ——
   否则该块不合法；
5. `out.name = name; out.arguments_json = args.dump()`（nlohmann JSON 对象，
   插入顺序 = 参数顺序）。

`parse_parameter`（232–302 行）：必须以 `<parameter=` 开头；键到 `>` 为止
（非空）；值到 `</parameter>` 为止。闭标签查找走 `find_closing_tag`（97–105 行）：
优先取行首出现（`\n</parameter>`），找不到才退回第一个普通出现 —— 值内部不在
行首的字面标签文本（例如 summary 里内嵌的 `</parameter>`）不会提前截断值；
`</function>`、`</tool_call>` 用同一规则。值解码规则：

| 参数情形 | 解码方式 |
|---|---|
| 契约为 `String` | 剥掉**一个**前导 + **一个**尾随框架换行（`remove_parameter_framing_newlines`，216–230 行，兼容 CRLF 与 LF），值按**纯字符串**保留 —— 绝不做 JSON 嗅探 |
| 契约为 `Json` | 剥框架换行后 JSON 解析；解析失败则**退化为裁剪后的原始字符串**（不做类型强转，也不拒绝调用） |
| 无契约条目（legacy） | `trim_ascii` 后 JSON 解析；解析被丢弃则保留裁剪后的**原始字符串** |

隔离粒度是逐块：结构性违规（坏标记、坏名称、未声明工具、缺闭标签、坏参数、
尾部垃圾）只让**该块**退化为逐字原文并留在 `content`，其余块照常产出调用；
只有**零调用**时才回退为整段原文（`is_tool_call_response = false`）。

### 8.3 流式过滤器：`ToolCallOutputDecoder`（头文件 62–83 行；`feed` 位于 `tool_call_parser.cpp:479-523`，`finish` 位于 525–548 行）

状态：`contract_`、`trailing_whitespace_`、`tool_region_`、`marker_prefix_bytes_`、
`saw_tool_marker_`、`finished_`。

`feed(text)` —— “增量发布那些可证明不可能属于终端 Qwen 工具调用后缀的字节”
（头文件注释，56–58 行）：

- 已 `finished_` → 抛 `std::logic_error`（“tool-call output decoder is
  already finished”）；
- 空输入 → 空输出；
- 无契约 → 直通（拷贝）；
- `saw_tool_marker_` 已置位 → 整块追加进 `tool_region_`，不输出任何内容
  （整个工具区被扣住）；
- 否则逐字节扫描，跟踪 11 字符 `"<tool_call>"` 标记的前缀匹配：
  - 空白字节被缓冲进 `trailing_whitespace_`（其后可能紧跟标记，故先扣住）；
  - 非空白且非标记字节：冲刷空白缓冲，该字节作为可见输出发出；
  - **完整匹配标记**时：`tool_region_ = 缓冲空白 + "<tool_call>" + 本块剩余`；
    `saw_tool_marker_ = true`；停止扫描（本块剩余全部被扣住）；
  - **前缀失配**时：缓冲空白 + 已匹配前缀字节冲刷为可见输出，前缀状态复位。

`finish()`：

- 无契约 → 空 `Terminal`；
- 用 `parse_qwen_tool_call_output` 解析 `tool_region_`；
- **有调用**（`saw_tool_marker_` 且 `is_tool_call_response`）：返回
  `Terminal{content = 降级块原文, tool_calls = 解析结果}` —— 被扣住的区域由
  结构体替代；正常路径下降级块为空，缓冲的尾随空白/残缺前缀被丢弃（它们属于
  被扣区域，而可见前缀已经流出）；
- **零调用**（从未见到标记，或解析回退）：返回
  `Terminal{content = 尾随空白 + 残缺前缀 + tool_region_（逐字）,
  tool_calls = {}}` —— 原始字节被精确还原，因此失败的工具响应会以普通文本
  形式浮出，而不是被吞掉。

### 8.4 前端与引擎接线

- `OutputSession::Impl` 构造函数
  （`src/targets/qwen3_6/impl/frontend/frontend.cpp:953-958`）：当
  `output.raw` 为真时解码器以**空契约**构造（raw 模式完全绕开工具调用解析），
  否则携带契约与 `output.tool_name_max_length`。
- `Frontend::prepare`（1362–1388 行）：契约 =
  `build_tool_call_output_contract(options.tool_jsons, !options.tool_jsons.empty())`；
  外加 §4 的 `ToolBoundary` prompt 缓存标记。契约经由
  `PreparedPromptData.tool_call_output`
  （`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/prepared_prompt.h:150`）
  随行传递。
- `OutputSession::commit_preview`（1255–1282 行）：preview 被接受后，
  **Content** 通道上的每个 `OutputDelta` 都经过 `decoder.feed`
  （可能返回更短的可见前缀）；到达终端时执行 `decoder.finish()`，
  `terminal.tool_calls` 移入会话，非空 `terminal.content` 追加到最后一个
  content 增量（或新推一个）。`take_tool_calls()`（1284 行）暴露累积的调用。
- 引擎取回：`src/runtime/engine/engine_core.h:1035`（`complete_success`）：
  `result.tool_calls = request->output.take_tool_calls()`。
- 另一个展示解码器（`feed_token_bytes`/`terminalize`，`frontend.cpp:502-660`，
  think 闭标记位于第 40 行）把解码后的原始字节按 `</think>` 标记切分为
  Reasoning/Content 通道，先于工具调用过滤器运行。

## 9. 流式可见性（服务层）

`ServiceOutputSink`（`src/serve/generation_service.cpp`）只把 **Reasoning** 与
**Content** 增量发布给流 sink（`StreamSink{on_start, on_content, on_reasoning,
is_cancelled}`）。工具调用区**从不以文本形式流式下发**：解码器从第一个
`<tool_call>` 标记起将其扣住，因此客户端看到标记前的文本（前缀，空白已裁剪），
结构化调用随终端 outcome 到达。`docs/serving.md:537` 记录了用户可见行为：
输出“立即流式下发；只有模糊的 `<tool_call>` 后缀或结构化工具区被扣住”。
`GenerationOutcome` 上的结构化 `tool_calls` 即各响应渲染器（§10）转成线上
形状的输入。

## 10. 响应渲染（工具调用发射）

ID（`src/serve/openai_common.cpp`，线程局部 `mt19937_64`）：chat 补全
`"chatcmpl-" + 16 位十六进制`；chat 工具调用 `"call_" + 16 位十六进制`；
response `"resp_" + 32 位十六进制`；response 条目 `prefix + "_" + 32 位十六进制`。

### 10.1 OpenAI Chat（`src/serve/openai_chat_response.cpp`）

- 每个 `GeneratedToolCall` 获得全新的 `call_…` id。
- 非流式 `message`：`role:"assistant"`、`content`（为空时 null）、
  `tool_calls: [{id, type:"function", function:{name, arguments}}]`，其中
  `arguments` 是**原始 `arguments_json` 字符串**；`reasoning_content` 可选；
  `refusal: null`。`finish_reason` 在存在调用时为 `"tool_calls"`，否则为
  `length`（OutputLimit/ContextCapacity）或 `stop`。
- 流式：起始块 `{role:"assistant", content:""}` → reasoning/content 增量块
  （只有解码器可见的文本）→ 结束时，已流式文本必须是终端文本的前缀
  （否则 `std::logic_error "streamed content does not match terminal
  output"`），冲刷后缀；有工具调用时：一个携带 `tool_calls`（每条目含
  `index`）的增量块，随后一个 `finish_reason:"tool_calls"` 块；无调用时
  为普通 `finish_reason` 块。可选 usage 块（含 timings/slot 标识），
  最后 `data: [DONE]`。

### 10.2 Anthropic Messages（`src/serve/anthropic_messages_response.cpp`）

- 工具 id 为 `"toolu_" + 16 位十六进制`。
- `parse_tool_input` 要求 `arguments_json` 能解码为 JSON **对象**；否则
  `std::logic_error "Engine produced non-object Anthropic tool input"`
  （该协议无法表示非对象输入）。
- 非流式 `content`：`thinking`（带签名）/ `text` / `tool_use {id, name, input}`
  块，`input` 为解析后的对象。`stop_reason` 存在调用时为 `"tool_use"`，
  否则为 `max_tokens`、`model_context_window_exceeded`、`stop_sequence`
  （带匹配到的序列）或 `end_turn`；被取消的结果抛异常。
- 流式：每个调用 `content_block_start`（tool_use 且 `input:{}`）→
  `content_block_delta`（`input_json_delta.partial_json` 在**一个 delta** 中
  携带整个 `arguments_json`）→ `content_block_stop`；随后 `message_delta`
  （stop_reason + usage）与 `message_stop`。

### 10.3 OpenAI Responses（`src/serve/openai_responses_response.cpp`）

- 每个调用生成一个输出条目 `{id: "fc_"…, type:"function_call",
  status:"completed", call_id: "call_"…, arguments: <字符串>, name}` ——
  可选 `namespace` 通过 `add_wire_function_identity` 重新挂回：按引擎名查询
  `request.tool_identities`（撤销 §7.1 的 `namespace__name` 扁平化）。
- 状态映射：OutputLimit/ContextCapacity → `"incomplete"`（带
  `incomplete_details.reason:"max_output_tokens"`）；Cancelled → `"cancelled"`；
  其余 → `"completed"`。
- 存在以下情况之一时发出 `message` 输出条目：文本非空；或既无调用也无
  reasoning 且状态为 `completed`。
- `output_history` 保留一条 assistant `ChatTurn`，其 `ToolCall` id 即
  `call_` id（可作为 `function_call` 条目原路往返）。
- 流式：每个调用 `response.output_item.added`（状态 `in_progress`，
  `arguments:""`）→ `response.function_call_arguments.delta`
  （**一个** delta 内携带整个 `arguments_json`）→
  `response.function_call_arguments.done` → `response.output_item.done`；
  终端事件为 `response.completed`、`response.incomplete` 或
  `response.failed`，携带完整响应体。

## 11. 错误码汇总（均为 HTTP 400，类型 `invalid_request_error`）

| 协议 | 错误码 | 触发条件 |
|---|---|---|
| Chat | `tool_type_not_supported` | 工具 `type` 非 `function` |
| Chat | `strict_tools_not_supported` | 工具 `strict: true` |
| Chat | `tool_choice_not_supported` | `required`，或对象形式 `function`/custom 工具选择 |
| Chat | `parallel_tool_calls_not_supported` | 带工具时 `parallel_tool_calls:false` |
| Anthropic | `content_block_not_supported` | `tool_result` content 块非 text/image |
| Anthropic | `invalid_tool_history` | 未知/重复 tool_result id，或带 `namespace` 的输出缺 name |
| Anthropic | `tool_choice_not_supported` | 按名（`tool`）或 `any` 工具选择 |
| Anthropic | `toolsets_not_supported` / `anthropic_tools_not_supported` | toolset / 托管工具 |
| Anthropic | `deferred_tools_not_supported` / `tool_caller_not_supported` / `parallel_tool_use_not_supported` | `defer_loading`、非 direct 调用方、禁用并行 |
| Anthropic | `invalid_thinking_signature` / `cache_prewarm_not_supported` | 历史 thinking 签名无效 / `max_tokens:0` |
| Responses | `invalid_tool_name` / `duplicate_tool_name` | 引擎名超出 `[A-Za-z0-9_-]{1,64}` / 扁平化名冲突 |
| Responses | `strict_tools_not_supported` / `deferred_tools_not_supported` / `tool_caller_not_supported` / `tool_output_schema_not_supported` | 不受支持的 function 工具成员 |
| Responses | `tool_type_not_supported` | 非 function/非 namespace 工具条目 |
| Responses | `tool_choice_not_supported` / `invalid_tool_choice` | named/hosted 选择 / 未声明的 `allowed_tools` 名称 |
| Responses | `partial_tool_call_not_supported` / `tool_relationship_not_supported` | 非 completed 调用 / 历史中的 `caller` |
| Responses | `invalid_assistant_history` | assistant 条目顺序违规 |
| Responses | `unknown_parameter` | 顶层字段不在白名单内 |

各协议的函数名长度上限：Chat **64**、Responses **64**（引擎名）、
Anthropic **128** —— 这正是输出 parser 对模型发出名称所应用的
`tool_name_max_length`。

## 12. 行为测试规格（`tests/test_tool_call_parser.cpp`）

| 测试 | 断言 |
|---|---|
| `test_single_call` | 标记前文本作为右裁剪后的 `content` 流出；名称 + 参数被解析 |
| `test_multiple_calls_and_json_values` | 多个块按顺序解析 |
| `test_malformed_falls_back_to_text` | 缺 `</function>` → 逐字回退，`is_tool_call_response=false` |
| `test_suffix_after_tool_keeps_call_and_text` | `</tool_call>` 后出现非空白 → 调用保留，后缀文本进入 content |
| `test_configured_name_limit` | 128 字符名称在上限 128 下合法（Anthropic）；上限 64（Chat）下被拒；129 字符在 128 下被拒 |
| `test_declared_strings_are_not_json_sniffed` | string 类型参数逐字保留 |
| `test_declared_non_string_values_are_json_decoded` | 非 string 类型按 JSON 解码 |
| `test_declared_type_mismatches_are_forwarded_without_coercion` | 布尔类型参数收到字符串 True → 值退化为原始字符串，调用保留 |
| `test_invalid_json_value_degrades_to_string` | 数组参数收到坏 JSON → 调用保留，值退化为原始字符串，兄弟参数不受影响 |
| `test_embedded_tag_literals_do_not_truncate` | 值内嵌 `</parameter>`/`</function>` 字面量（不在行首）→ 不截断，正常解码 |
| `test_failed_block_degrades_but_other_calls_survive` | 未声明工具块退化为 content 原文，前后两个合法调用保留 |
| `test_incremental_filter_keeps_calls_next_to_degraded_block` | 流式：可见前缀不受影响；终端同时给出有效调用与降级块原文 |
| `test_unknown_schema_keeps_legacy_inference` | 无类型参数 → legacy 裁剪 + JSON 嗅探 |
| `test_parser_enforces_active_tool_set` | 未声明工具名 → 回退；原始 `<function=other>` 保留在 content 中 |
| `test_incremental_filter_valid_tool` | 三段 feed（首段为 “Calling weather.” + 两个空格 + 换行 + `<tool_`，次段为 `call>` + 换行 + `<function=get_weather>`，末段为 换行 + `</function>` + 换行 + `</tool_call>`）→ 可见输出恰为 `Calling weather.`；终端持有结构化调用 |
| `test_incremental_filter_fallback` | 不合法区域精确还原原始字节；普通文本保留尾随空白；残缺标记失配（首段 `两个空格 + <too`，次段 `l_x then <tool_`）保留原始字节 |

## 13. 周边路径（非 serve HTTP parser）

- `src/product/prompt_input/prompt_input.cpp:142-217`：CLI JSON prompt 路径
  —— `tool_calls` 仅限 assistant 消息、`tool_call_id` 仅限工具消息 —— 共享同一
  线上传输中立模型，但不是 HTTP 协议。
- `src/serve/openai_responses_store.cpp`：Responses `store` 请求的持久化；
  `src/serve/request_events`/请求日志记录所选 tool_choice，但不改变解析。
- 展示解码器主体（`feed_token_bytes`/`terminalize`，`frontend.cpp:502-660`）
  围绕 `</think>` 标记把解码字节切分为 Reasoning/Content 通道；它先于工具调用
  过滤器运行，与其正交。

## 14. 设计要点

1. **引擎与工具无关。** 工具被渲染为 prompt 文本，模型发出纯文本 XML；结构在
   前端事后恢复。这让引擎保持协议中立，一个目标 parser 即可服务全部三种线上协议。
2. **逐块隔离的回退。** 结构违规只让**该块**退化为逐字保留的原文并留在
   `content`，其余块照常产出调用；值解码失败退化为原始字符串而不拒绝调用。
   只有零调用时才回退为整段文本。流式过滤器在整段回退时精确还原字节，客户端
   不会因单个字节错而丢掉整轮工具调用。
3. **契约驱动的解码。** 已声明的参数类型决定字符串 vs JSON 解码；未声明/有歧义
   的参数回退到 legacy 的“裁剪后嗅探”推断。不做 schema 校验、不做类型强转，
   因此类型不匹配表现为原始字符串，而不是静默强转的值。
4. **已声明名称强制。** 工具激活时，对未声明函数的调用被视为不合法（prompt
   已告知模型存在哪些工具），防止幻觉工具名被当作结构化调用浮出。
5. **流式只扣住模糊部分。** 只有“仍可能构成 `<tool_call>` 标记”或“属于工具区”
   的字节才被扣住；可证明在其之外的内容立即流出。