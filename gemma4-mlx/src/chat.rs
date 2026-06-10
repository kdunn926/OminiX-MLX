use std::{collections::HashMap, path::Path};

use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::{load_model, load_tokenizer, Error, Generate, KVCache, Model, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gemma4Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Gemma4Role {
    /// Returns the role token as it appears in the rendered prompt.
    ///
    /// **Important:** Gemma 4 uses `model` (not `assistant`) for assistant
    /// turns — every shipped `chat_template.jinja` translates `assistant` →
    /// `model` before emitting the `<|turn>...` marker (search for
    /// `set role = 'model' if message['role'] == 'assistant'` in the
    /// templates). The model was trained on `<|turn>model\n`; feeding it
    /// `<|turn>assistant\n` is observably worse (notably it stops respecting
    /// the empty-channel pre-fill suppression and starts emitting an
    /// unsolicited reasoning channel that leaks `"thought\n..."` plaintext
    /// once the tokenizer eats the surrounding special-token wrappers).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "model",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Gemma4ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ToolCall {
    pub name: String,
    pub arguments: Value,
    /// Opaque call id matching the OpenAI/Anthropic tool-call → tool-result
    /// protocol. Used by the Jinja renderer to link an assistant tool call
    /// to its follow-up `role: tool` message via `tool_call_id`. `None` when
    /// the caller doesn't track ids (the Rust fallback renderer ignores it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl Gemma4ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
            id: None,
        }
    }

    /// Builder: attach the OpenAI-style `call_…` id so the Jinja renderer
    /// can pair this call with its follow-up `role: tool` message.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Message {
    pub role: Gemma4Role,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
    /// Name of the tool that produced this `role: tool` message (used by
    /// both the Rust fallback renderer and the Jinja `name` lookup).
    pub tool_name: Option<String>,
    /// Opaque id linking a `role: tool` message back to a prior assistant
    /// `tool_calls[*].id`. The Jinja templates' forward-scan match this
    /// against `tc.get('id')` to recover the tool name; leaving it `None`
    /// falls back to matching by `tool_name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Gemma4Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::System,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::User,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::Assistant,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<Gemma4ToolCall>,
    ) -> Self {
        Self {
            role: Gemma4Role::Assistant,
            content: content.into(),
            thinking: None,
            tool_calls,
            tool_name: None,
            tool_call_id: None,
        }
    }

    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::Tool,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: Some(name.into()),
            tool_call_id: None,
        }
    }

    /// Builder: attach the OpenAI-style `tool_call_id` so the Jinja
    /// renderer can match this `role: tool` message back to the originating
    /// assistant `tool_calls[*].id`.
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ParsedAssistantResponse {
    pub raw_text: String,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4ToolResult {
    pub name: String,
    pub content: Value,
    pub success: bool,
    pub error: Option<String>,
}

impl Gemma4ToolResult {
    pub fn success(name: impl Into<String>, content: Value) -> Self {
        Self {
            name: name.into(),
            content,
            success: true,
            error: None,
        }
    }

    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: Value::Null,
            success: false,
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4ChatResponse {
    pub raw_text: String,
    pub text: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
    pub tool_results: Vec<Gemma4ToolResult>,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedAssistantTurn {
    parsed: Gemma4ParsedAssistantResponse,
    tokens_generated: usize,
}

impl GeneratedAssistantTurn {
    fn new(parsed: Gemma4ParsedAssistantResponse, tokens_generated: usize) -> Self {
        Self {
            parsed,
            tokens_generated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Gemma4Conversation {
    pub messages: Vec<Gemma4Message>,
}

impl Gemma4Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_user(&mut self, content: impl Into<String>) {
        self.messages.push(Gemma4Message::user(content));
    }

    pub fn add_assistant_message(&mut self, message: Gemma4Message) {
        self.messages.push(message);
    }

    pub fn add_tool_result(&mut self, result: &Gemma4ToolResult) {
        let content = if result.success {
            result.content.clone()
        } else {
            serde_json::json!({ "error": result.error.clone().unwrap_or_default() })
        };
        self.messages
            .push(Gemma4Message::tool(&result.name, content.to_string()));
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ChatConfig {
    pub temperature: f32,
    pub max_new_tokens: usize,
    pub max_tool_iterations: usize,
}

impl Default for Gemma4ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_new_tokens: 2048,
            max_tool_iterations: 4,
        }
    }
}

pub trait Gemma4Tool: Send + Sync {
    fn spec(&self) -> Gemma4ToolSpec;
    fn execute(&self, arguments: &Value) -> Result<Value>;
}

pub struct Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    spec: Gemma4ToolSpec,
    handler: F,
}

impl<F> Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    pub fn new(spec: Gemma4ToolSpec, handler: F) -> Self {
        Self { spec, handler }
    }
}

impl<F> Gemma4Tool for Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    fn spec(&self) -> Gemma4ToolSpec {
        self.spec.clone()
    }

    fn execute(&self, arguments: &Value) -> Result<Value> {
        (self.handler)(arguments)
    }
}

#[derive(Default)]
pub struct Gemma4ToolRegistry {
    tools: HashMap<String, Box<dyn Gemma4Tool>>,
}

impl Gemma4ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Gemma4Tool + 'static,
    {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), Box::new(tool));
    }

    pub fn specs(&self) -> Vec<Gemma4ToolSpec> {
        let mut specs: Vec<_> = self.tools.values().map(|tool| tool.spec()).collect();
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        specs
    }

    pub fn execute_call(&self, call: &Gemma4ToolCall) -> Gemma4ToolResult {
        match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(&call.arguments) {
                Ok(content) => Gemma4ToolResult::success(&call.name, content),
                Err(error) => Gemma4ToolResult::failure(&call.name, error.to_string()),
            },
            None => Gemma4ToolResult::failure(&call.name, format!("Unknown tool: {}", call.name)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4SpecialTokens {
    pub bos: String,
    pub eos: String,
    pub turn_start: String,
    pub turn_end: String,
    pub channel_start: String,
    pub channel_end: String,
    pub tool_start: String,
    pub tool_end: String,
    pub tool_call_start: String,
    pub tool_call_end: String,
    pub tool_response_start: String,
    pub tool_response_end: String,
    /// Gemma's "escape" / string-delimiter token (default `<|"|>`). Used as
    /// the quote character inside the model's native tool-call DSL — e.g.
    /// `call:read_file{path:<|"|>/foo/bar<|"|>}`. Strip + replace with `"`
    /// when normalising a DSL payload into JSON for parsing (Bug 1 fix).
    pub escape: String,
}

impl Default for Gemma4SpecialTokens {
    fn default() -> Self {
        Self {
            bos: "<bos>".to_string(),
            eos: "<eos>".to_string(),
            turn_start: "<|turn>".to_string(),
            turn_end: "<turn|>".to_string(),
            channel_start: "<|channel>".to_string(),
            channel_end: "<channel|>".to_string(),
            tool_start: "<|tool>".to_string(),
            tool_end: "<tool|>".to_string(),
            tool_call_start: "<|tool_call>".to_string(),
            tool_call_end: "<tool_call|>".to_string(),
            tool_response_start: "<|tool_response>".to_string(),
            tool_response_end: "<tool_response|>".to_string(),
            escape: "<|\"|>".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ChatTemplate {
    pub tokens: Gemma4SpecialTokens,
    pub default_system_prompt: String,
    /// Raw `chat_template.jinja` source bundled with the model, if found.
    /// When present, `render_prompt` runs this via [`minijinja`] in
    /// preference to the hand-rolled Rust renderer — the Jinja is what
    /// the model was trained on (role token `model` not `assistant`, the
    /// `<|tool>declaration:...{description:<|"|>...<|"|>}<tool|>` tool
    /// declaration grammar, the in-turn `<|tool_response>...<tool_response|>`
    /// embedding rather than a separate `<|turn>tool` turn, etc.).
    ///
    /// If Jinja rendering errors at runtime, `render_prompt` falls back to
    /// the Rust path so a malformed template never wedges generation. The
    /// fallback now matches the Jinja on the role string (`model`), but
    /// still differs on tool-call/response grammar — set
    /// `OMINIX_GEMMA4_DISABLE_JINJA=1` if you need to force the fallback
    /// path for A/B comparison.
    pub jinja_source: Option<String>,
}

impl Default for Gemma4ChatTemplate {
    fn default() -> Self {
        Self {
            tokens: Gemma4SpecialTokens::default(),
            default_system_prompt: "You are a helpful assistant.".to_string(),
            jinja_source: None,
        }
    }
}

impl Gemma4ChatTemplate {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let defaults = Gemma4SpecialTokens::default();
        let model_dir = model_dir.as_ref();
        let tokenizer_config_path = model_dir.join("tokenizer_config.json");
        let json = std::fs::read_to_string(tokenizer_config_path)?;
        let config: TokenizerConfigTokens = serde_json::from_str(&json)?;

        // `chat_template.jinja` is optional — pre-Gemma4 era models, DFlash
        // draft dirs, and some quant variants ship without it. When absent,
        // we fall back to the hand-rolled Rust renderer. Read failures (e.g.
        // permission denied) are non-fatal for the same reason.
        let jinja_path = model_dir.join("chat_template.jinja");
        let jinja_source = std::fs::read_to_string(&jinja_path).ok();

        Ok(Self {
            tokens: Gemma4SpecialTokens {
                bos: config.bos_token.unwrap_or(defaults.bos),
                eos: config.eos_token.unwrap_or(defaults.eos),
                turn_start: config.sot_token.unwrap_or(defaults.turn_start),
                turn_end: config.eot_token.unwrap_or(defaults.turn_end),
                channel_start: config.soc_token.unwrap_or(defaults.channel_start),
                channel_end: config.eoc_token.unwrap_or(defaults.channel_end),
                tool_start: config.std_token.unwrap_or(defaults.tool_start),
                tool_end: config.etd_token.unwrap_or(defaults.tool_end),
                tool_call_start: config.stc_token.unwrap_or(defaults.tool_call_start),
                tool_call_end: config.etc_token.unwrap_or(defaults.tool_call_end),
                tool_response_start: config.str_token.unwrap_or(defaults.tool_response_start),
                tool_response_end: config.etr_token.unwrap_or(defaults.tool_response_end),
                escape: config.escape_token.unwrap_or(defaults.escape),
            },
            default_system_prompt: "You are a helpful assistant.".to_string(),
            jinja_source,
        })
    }

    pub fn render_prompt(
        &self,
        messages: &[Gemma4Message],
        tools: &[Gemma4ToolSpec],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // Prefer the model's bundled `chat_template.jinja` — it's what the
        // model was trained on. Fall back to the hand-rolled Rust renderer
        // if no Jinja was loaded, or if Jinja execution errors at runtime
        // (so a malformed template never wedges generation), or if the user
        // sets `OMINIX_GEMMA4_DISABLE_JINJA=1` for A/B comparison.
        let jinja_disabled = std::env::var_os("OMINIX_GEMMA4_DISABLE_JINJA").is_some();
        if !jinja_disabled {
            if let Some(src) = self.jinja_source.as_deref() {
                match self.render_via_jinja(src, messages, tools, add_generation_prompt) {
                    Ok(rendered) => return Ok(rendered),
                    Err(e) => {
                        eprintln!(
                            "[gemma4-mlx] chat_template.jinja render failed ({}); \
                             falling back to Rust renderer",
                            e
                        );
                    }
                }
            }
        }
        self.render_prompt_rust(messages, tools, add_generation_prompt)
    }

    /// Render the prompt by executing the model's `chat_template.jinja`
    /// directly via [`minijinja`]. Messages and tools are serialized to
    /// JSON so the template sees the same shape a Python `transformers`
    /// caller would: `messages[i]['role'|'content'|'tool_calls'|...]`,
    /// `tools[i]['function']['name'|'description'|'parameters']`,
    /// `bos_token`, `add_generation_prompt`.
    fn render_via_jinja(
        &self,
        source: &str,
        messages: &[Gemma4Message],
        tools: &[Gemma4ToolSpec],
        add_generation_prompt: bool,
    ) -> std::result::Result<String, minijinja::Error> {
        let mut env = minijinja::Environment::new();
        // `.get()`, `.split()`, `.startswith()` etc. — the Gemma4 templates
        // assume the HuggingFace transformers Python-string-method surface.
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        env.add_template("gemma4", source)?;
        let template = env.get_template("gemma4")?;

        // Preserve the Rust fallback's long-standing behavior of always
        // emitting a system turn: if no `system`/`developer` is at index 0,
        // prepend a synthetic one with `default_system_prompt`. Without it,
        // the Jinja's first `if` would skip the system turn entirely
        // (`messages[0]['role'] in ['system','developer']` is its trigger),
        // changing the prompt shape for tool-less chats with no system.
        let needs_default_system = !self.default_system_prompt.is_empty()
            && messages
                .first()
                .map(|m| !matches!(m.role, Gemma4Role::System))
                .unwrap_or(true);
        let mut messages_json: Vec<serde_json::Value> = Vec::with_capacity(
            messages.len() + if needs_default_system { 1 } else { 0 },
        );
        if needs_default_system {
            messages_json.push(message_to_jinja_value(&Gemma4Message::system(
                self.default_system_prompt.clone(),
            )));
        }
        messages_json.extend(messages.iter().map(message_to_jinja_value));
        // The Jinja iterates `for tool in tools` and indexes
        // `tool_data['function']['name'|'description'|'parameters']` — wrap
        // our flat `Gemma4ToolSpec` to match.
        let tools_json: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })
            })
            .collect();
        // The 12B / 26B templates emit the empty-channel pre-fill only when
        // `enable_thinking` is falsy (the default). We pass `false` explicitly
        // so the pre-fill kicks in — matching what the Rust fallback always
        // did. e4b's template omits the pre-fill entirely (it ignores
        // `enable_thinking` for `add_generation_prompt`), so this flag has no
        // effect on e4b — which is the correct behavior per its training.
        let ctx = minijinja::context! {
            messages => messages_json,
            tools => if tools_json.is_empty() { None } else { Some(tools_json) },
            bos_token => self.tokens.bos,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => false,
        };
        template.render(ctx)
    }

    fn render_prompt_rust(
        &self,
        messages: &[Gemma4Message],
        tools: &[Gemma4ToolSpec],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut prompt = String::new();
        prompt.push_str(&self.tokens.bos);

        let (system_message, remaining_messages) = match messages.first() {
            Some(message) if message.role == Gemma4Role::System => {
                (message.content.clone(), &messages[1..])
            }
            _ => (self.default_system_prompt.clone(), messages),
        };

        let mut system_body = system_message;
        if !tools.is_empty() {
            if !system_body.is_empty() {
                system_body.push_str("\n\n");
            }
            system_body.push_str(&self.render_tool_instructions(tools)?);
        }
        prompt.push_str(&self.render_turn(Gemma4Role::System, &system_body)?);

        for message in remaining_messages {
            prompt.push_str(&self.render_message(message)?);
        }

        if add_generation_prompt {
            prompt.push_str(&self.tokens.turn_start);
            prompt.push_str(Gemma4Role::Assistant.as_str());
            prompt.push('\n');
            // Suppress thinking channel (matches Jinja template `enable_thinking | default(false)`).
            // Without this prefix, the model generates <|channel>thought\n...<channel|> as its first
            // tokens; the tokenizer then strips the special-token delimiters, leaking "thought\n"
            // into visible output.
            prompt.push_str(&self.tokens.channel_start);
            prompt.push_str("thought\n");
            prompt.push_str(&self.tokens.channel_end);
        }

        Ok(prompt)
    }

    pub fn render_tool_instructions(&self, tools: &[Gemma4ToolSpec]) -> Result<String> {
        let mut body = String::from("Available tools:\n");
        for tool in tools {
            body.push_str(&self.tokens.tool_start);
            body.push_str(&serialize_tool_spec(tool)?);
            body.push_str(&self.tokens.tool_end);
            body.push('\n');
        }
        body.push_str(
            "When you call a tool, respond with one or more tool calls formatted exactly as:\n",
        );
        body.push_str(&self.tokens.tool_call_start);
        body.push_str("call:tool_name{\"arg\":\"value\"}");
        body.push_str(&self.tokens.tool_call_end);
        Ok(body)
    }

    fn render_message(&self, message: &Gemma4Message) -> Result<String> {
        match message.role {
            Gemma4Role::System | Gemma4Role::User => {
                self.render_turn(message.role, &message.content)
            }
            Gemma4Role::Assistant => {
                let mut body = String::new();
                if let Some(thinking) = message.thinking.as_deref() {
                    body.push_str(&self.tokens.channel_start);
                    body.push_str("thought\n");
                    body.push_str(thinking);
                    body.push_str(&self.tokens.channel_end);
                }
                if !message.content.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(&message.content);
                }
                if !message.tool_calls.is_empty() {
                    for tool_call in &message.tool_calls {
                        if !body.is_empty() {
                            body.push('\n');
                        }
                        body.push_str(&self.tokens.tool_call_start);
                        body.push_str("call:");
                        body.push_str(&tool_call.name);
                        body.push_str(&serde_json::to_string(&tool_call.arguments)?);
                        body.push_str(&self.tokens.tool_call_end);
                    }
                }
                self.render_turn(Gemma4Role::Assistant, &body)
            }
            Gemma4Role::Tool => {
                let tool_name = message.tool_name.as_deref().ok_or_else(|| {
                    Error::Model("Gemma4 tool messages must include a tool_name".to_string())
                })?;
                let tool_content = parse_json_or_string(&message.content);
                let mut body = String::new();
                body.push_str(tool_name);
                body.push('\n');
                body.push_str(&self.tokens.tool_response_start);
                body.push_str(&serialize_tool_response(tool_name, tool_content)?);
                body.push_str(&self.tokens.tool_response_end);
                self.render_turn(Gemma4Role::Tool, &body)
            }
        }
    }

    fn render_turn(&self, role: Gemma4Role, body: &str) -> Result<String> {
        if body.contains(&self.tokens.turn_start) {
            return Err(Error::Model(format!(
                "Gemma4 turn body for role {} contains the turn start token",
                role.as_str()
            )));
        }

        Ok(format!(
            "{}{}\n{}\n{}\n",
            self.tokens.turn_start,
            role.as_str(),
            body,
            self.tokens.turn_end
        ))
    }

    pub fn has_complete_tool_call(&self, text: &str) -> bool {
        text.contains(&self.tokens.tool_call_start) && text.contains(&self.tokens.tool_call_end)
    }

    pub fn parse_assistant_response(
        &self,
        raw_text: &str,
    ) -> Result<Gemma4ParsedAssistantResponse> {
        let mut remaining = raw_text.to_string();
        let thinking = self.extract_thinking(&mut remaining)?;
        let tool_calls = self.extract_tool_calls(&mut remaining)?;
        let mut content = remaining.trim().to_string();
        loop {
            let trimmed = content.trim_end();
            if let Some(next) = trimmed.strip_suffix(&self.tokens.turn_end) {
                content = next.trim_end().to_string();
                continue;
            }
            if let Some(next) = trimmed.strip_suffix(&self.tokens.eos) {
                content = next.trim_end().to_string();
                continue;
            }
            content = trimmed.to_string();
            break;
        }

        Ok(Gemma4ParsedAssistantResponse {
            raw_text: raw_text.to_string(),
            content,
            thinking,
            tool_calls,
        })
    }

    /// Extract `<|channel>LABEL\nBODY<channel|>` blocks from `remaining`.
    ///
    /// Mirrors the Jinja template's `strip_thinking` macro semantics:
    /// **every** channel block is stripped from `remaining`, regardless of
    /// label. The first block whose label is `thought` becomes the
    /// returned reasoning text; other labels (`analysis`, `final`, etc.)
    /// are dropped silently — they shouldn't be user-visible either way,
    /// and the reasoning-vs-content split in the OpenAI ChatCompletions
    /// surface only has room for one channel.
    ///
    /// Behavior:
    /// - **No `<|channel>` markers**: returns `Ok(None)`, `remaining`
    ///   unchanged.
    /// - **One or more closed blocks**: every block removed; the first
    ///   `thought`-labeled body returned.
    /// - **Unclosed block** (model hit max_tokens mid-channel): drop from
    ///   `<|channel>` to EOF rather than erroring the whole response;
    ///   return the captured tail as a courtesy if it was a `thought`
    ///   block, so the client still sees partial reasoning.
    /// - **Channel with no newline after the label**: treat the entire
    ///   pre-`<channel|>` span as body, label = "" (won't match `thought`,
    ///   so won't surface as reasoning, but still strips from content).
    fn extract_thinking(&self, remaining: &mut String) -> Result<Option<String>> {
        let start_marker = self.tokens.channel_start.as_str();
        let end_marker = self.tokens.channel_end.as_str();
        let mut thinking: Option<String> = None;

        loop {
            let Some(start_idx) = remaining.find(start_marker) else {
                return Ok(thinking);
            };
            let after_open = start_idx + start_marker.len();
            // Split label from body: first `\n` after `<|channel>` ends
            // the label. No newline → empty label, body begins immediately.
            let (label, body_start) = match remaining[after_open..].find('\n') {
                Some(nl) => {
                    let label = remaining[after_open..after_open + nl].to_string();
                    (label, after_open + nl + 1)
                }
                None => (String::new(), after_open),
            };

            // Find close. If missing (truncated generation), drop from
            // `<|channel>` to EOF and surface the captured tail as
            // partial reasoning when the label was `thought`.
            let close_rel = remaining[body_start..].find(end_marker);
            let (body_end, remove_end) = match close_rel {
                Some(rel) => {
                    let body_end = body_start + rel;
                    let mut remove = body_end + end_marker.len();
                    if remaining[remove..].starts_with('\n') {
                        remove += 1;
                    }
                    (body_end, remove)
                }
                None => {
                    // Unclosed — everything to EOF is the body.
                    let body_end = remaining.len();
                    let body = remaining[body_start..body_end].to_string();
                    if thinking.is_none() && label.trim() == "thought" {
                        thinking = Some(body);
                    }
                    remaining.truncate(start_idx);
                    return Ok(thinking);
                }
            };

            let body = remaining[body_start..body_end].to_string();
            if thinking.is_none() && label.trim() == "thought" {
                thinking = Some(body);
            }
            remaining.replace_range(start_idx..remove_end, "");
        }
    }

    fn extract_tool_calls(&self, remaining: &mut String) -> Result<Vec<Gemma4ToolCall>> {
        let mut tool_calls = Vec::new();

        // Wrapped form: `<|tool_call>call:name{json}<tool_call|>`. This is
        // what we see when the tokenizer kept the special-token wrappers
        // (uncommon — only when `skip_special_tokens=false` or the model
        // emitted the bracket sequences as plain text by accident).
        while let Some(start_idx) = remaining.find(&self.tokens.tool_call_start) {
            let payload_start = start_idx + self.tokens.tool_call_start.len();
            let Some(end_rel) = remaining[payload_start..].find(&self.tokens.tool_call_end) else {
                // No matching end marker — tokenizer stripped only `<tool_call|>`
                // (asymmetric special-token stripping) or generation was truncated.
                // Strip the orphaned start token here so it doesn't leak into
                // content; the bare-form scanner below will still extract the
                // `call:name{...}` payload that follows it.
                remaining.replace_range(start_idx..payload_start, "");
                break;
            };
            let payload_end = payload_start + end_rel;
            let payload = remaining[payload_start..payload_end].trim();
            let tool_call = parse_tool_call_payload(payload, &self.tokens.escape)?;
            tool_calls.push(tool_call);

            let mut remove_end = payload_end + self.tokens.tool_call_end.len();
            if remaining[remove_end..].starts_with('\n') {
                remove_end += 1;
            }
            remaining.replace_range(start_idx..remove_end, "");
        }

        // Bare form: just `call:name{json}` with no wrappers. This is the
        // *common* case under our text path — `tokenizer.decode(_, true)`
        // strips `<|tool_call>` / `<tool_call|>` as special tokens, so the
        // payload reaches us unwrapped. Without this scan, the call leaks
        // into `parsed.content` (observed: gemma-4-12B via Hermes-gateway
        // emitting `call:terminal{...}` as visible text).
        tool_calls.extend(extract_bare_call_payloads(remaining, &self.tokens.escape));

        // Safety: strip any surviving control-token stubs that neither the
        // wrapped-form nor the bare-form path consumed. These are always
        // parser artefacts — never user-visible content. Handles the
        // complementary case: tokenizer stripped `<|tool_call>` but kept
        // `<tool_call|>` (or vice versa, handled above), leaving an orphaned
        // end marker after bare-form spliced out the call payload.
        for marker in [&self.tokens.tool_call_start, &self.tokens.tool_call_end] {
            while let Some(idx) = remaining.find(marker.as_str()) {
                remaining.replace_range(idx..idx + marker.len(), "");
            }
        }

        Ok(tool_calls)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigTokens {
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    sot_token: Option<String>,
    #[serde(default)]
    eot_token: Option<String>,
    #[serde(default)]
    soc_token: Option<String>,
    #[serde(default)]
    eoc_token: Option<String>,
    #[serde(default)]
    std_token: Option<String>,
    #[serde(default)]
    etd_token: Option<String>,
    #[serde(default)]
    stc_token: Option<String>,
    #[serde(default)]
    etc_token: Option<String>,
    #[serde(default)]
    str_token: Option<String>,
    #[serde(default)]
    etr_token: Option<String>,
    #[serde(default)]
    escape_token: Option<String>,
}

fn parse_json_or_string(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

/// Convert a [`Gemma4Message`] into the JSON shape the Jinja templates
/// expect. Keys mirror what HuggingFace `transformers` would hand the
/// template:
///
/// - `role` — raw role string (`assistant`, not `model`; the template
///   itself rewrites it).
/// - `content` — string body.
/// - `tool_calls[*].id` / `tool_calls[*].function.{name,arguments}` —
///   matches the OpenAI ChatCompletions shape so `tc.get('id') ==
///   follow.get('tool_call_id')` works in the forward-scan.
/// - `name` / `tool_call_id` — populated on `role: tool` follow-ups.
/// - `reasoning_content` — preserved when `Gemma4Message.thinking` is set
///   so the template's reasoning-channel re-render path is reachable.
///
/// Empty optional fields are omitted so `.get(...)` returns `Undefined`
/// (the template branches on it).
fn message_to_jinja_value(msg: &Gemma4Message) -> serde_json::Value {
    let role_str = match msg.role {
        Gemma4Role::System => "system",
        Gemma4Role::User => "user",
        // The Jinja templates rewrite `assistant` → `model` themselves
        // (`set role = 'model' if message['role'] == 'assistant'`). Don't
        // pre-translate here — the template needs the raw OpenAI-shape
        // role to drive its `last_user_idx` and continuation logic.
        Gemma4Role::Assistant => "assistant",
        Gemma4Role::Tool => "tool",
    };
    let mut m = serde_json::Map::new();
    m.insert("role".into(), serde_json::Value::String(role_str.into()));
    m.insert(
        "content".into(),
        serde_json::Value::String(msg.content.clone()),
    );
    if let Some(t) = &msg.thinking {
        // Jinja looks up `message.get('reasoning') or message.get('reasoning_content')`.
        // Set the `_content` variant — that's the OpenAI-API spelling our
        // upstream callers use.
        m.insert(
            "reasoning_content".into(),
            serde_json::Value::String(t.clone()),
        );
    }
    if !msg.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = msg
            .tool_calls
            .iter()
            .map(|tc| {
                let mut c = serde_json::Map::new();
                if let Some(id) = &tc.id {
                    c.insert("id".into(), serde_json::Value::String(id.clone()));
                }
                c.insert("type".into(), serde_json::Value::String("function".into()));
                c.insert(
                    "function".into(),
                    serde_json::json!({
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }),
                );
                serde_json::Value::Object(c)
            })
            .collect();
        m.insert("tool_calls".into(), serde_json::Value::Array(calls));
    }
    if let Some(name) = &msg.tool_name {
        m.insert("name".into(), serde_json::Value::String(name.clone()));
    }
    if let Some(id) = &msg.tool_call_id {
        m.insert(
            "tool_call_id".into(),
            serde_json::Value::String(id.clone()),
        );
    }
    serde_json::Value::Object(m)
}

fn serialize_tool_spec(tool: &Gemma4ToolSpec) -> Result<String> {
    let mut map = Map::new();
    map.insert(
        "description".to_string(),
        Value::String(tool.description.clone()),
    );
    map.insert("name".to_string(), Value::String(tool.name.clone()));
    map.insert("parameters".to_string(), tool.parameters.clone());
    serde_json::to_string(&Value::Object(map)).map_err(Into::into)
}

fn serialize_tool_response(tool_name: &str, content: Value) -> Result<String> {
    let mut map = Map::new();
    map.insert("content".to_string(), content);
    map.insert("name".to_string(), Value::String(tool_name.to_string()));
    serde_json::to_string(&Value::Object(map)).map_err(Into::into)
}

/// Scan `remaining` for bare `call:name{json}` payloads — the post-decode
/// form of `<|tool_call>call:name{json}<tool_call|>` after the tokenizer
/// strips the wrappers (`skip_special_tokens=true`, the default on every
/// text path in this workspace). Each extracted call is spliced out of
/// `remaining` (along with one trailing newline) so the residual
/// `content` doesn't double-report the call as visible text.
///
/// Validation:
///   - `call:` must start at a word boundary (start of text or after a
///     non-identifier char), so prose like `Recall: budget {2024}` is
///     never treated as a tool call.
///   - The character run after `call:` and before `{` must be a valid tool
///     name (`[A-Za-z0-9_.\-]+`). `call: see also` and `call: alone` are
///     not picked up — the scan cursor advances past the prefix without
///     touching the text, so non-matches stay verbatim in `content`.
///   - The arguments object is bracket-matched with a string-aware
///     depth counter (handles nested objects, arrays, and escaped quotes).
///   - If generation truncated mid-args (no matching close brace before
///     EOF), the call still dispatches with `{}` rather than leaving
///     `call:name{partial...` in `content`.
///   - Native-DSL arg strings (the Jinja-trained `{path:<|"|>/foo<|"|>}`
///     form, which decodes to `{path:/foo}` once the `<|"|>` escape token
///     is dropped as special) are normalised into JSON via
///     [`normalize_gemma4_dsl_to_json`] before parsing. Tool calls used to
///     dispatch with **empty args** in this case — the dominant failure
///     mode for `read_file`, `Bash`, `Edit` and other path-bearing tools
///     against gemma-4-12B-it. (Bug 1 fix.)
fn extract_bare_call_payloads(remaining: &mut String, escape_token: &str) -> Vec<Gemma4ToolCall> {
    const PREFIX: &str = "call:";
    let mut out = Vec::new();
    // Scan cursor: non-matches advance the cursor instead of mutating
    // `remaining`, so prose containing `call:` is never altered.
    let mut cursor = 0;
    while let Some(rel_start) = remaining[cursor..].find(PREFIX) {
        let start = cursor + rel_start;
        // Word-boundary check: `Recall:`/`recall:` etc. are prose.
        let at_boundary = remaining[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if !at_boundary {
            cursor = start + PREFIX.len();
            continue;
        }
        let name_start = start + PREFIX.len();
        let Some(brace_offset) = remaining[name_start..].find('{') else {
            // No `{` anywhere after this `call:` — nothing to parse.
            break;
        };
        let name = remaining[name_start..name_start + brace_offset].trim();
        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.');
        if !valid_name {
            // Not a real tool call (`call: see notes`, …). Scan past the
            // prefix and leave the text untouched.
            cursor = start + PREFIX.len();
            continue;
        }
        let name = name.to_string();
        let json_start = name_start + brace_offset;

        // String-aware brace-depth matcher: `{` and `}` outside string
        // literals adjust depth; `"` toggles in_string; `\\` inside a
        // string escapes the next byte (so `"foo\"bar"` doesn't close).
        let mut depth: i32 = 0;
        let mut end: Option<usize> = None;
        let mut in_string = false;
        let mut escape_next = false;
        for (i, b) in remaining[json_start..].bytes().enumerate() {
            if escape_next {
                escape_next = false;
                continue;
            }
            match b {
                b'\\' if in_string => escape_next = true,
                b'"' => in_string = !in_string,
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(json_start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let (arguments, splice_end) = match end {
            Some(e) => {
                let json_str = &remaining[json_start..e];
                // Try direct JSON parse first — fast path for the model
                // emitting (or having been prompted with) plain JSON. On
                // failure, normalise Gemma's native DSL into JSON
                // (`<|"|>` → `"`, then quote bare-identifier keys) before
                // retrying. Only after both attempts fail do we fall back
                // to empty args (truly malformed payload).
                let args = serde_json::from_str::<Value>(json_str)
                    .or_else(|_| {
                        let normalised = normalize_gemma4_dsl_to_json(json_str, escape_token);
                        serde_json::from_str::<Value>(&normalised)
                    })
                    .unwrap_or_else(|_| Value::Object(Map::new()));
                (args, e)
            }
            None => {
                // Truncated mid-args (model hit max_tokens). Dispatch with
                // empty args rather than leak `call:name{partial` as text.
                (Value::Object(Map::new()), remaining.len())
            }
        };
        out.push(Gemma4ToolCall::new(name, arguments));

        // Splice the whole `call:name{...}` block out, plus one trailing
        // newline so we don't leave a stranded blank line between
        // surrounding prose lines.
        let mut end_cut = splice_end;
        if remaining[end_cut..].starts_with('\n') {
            end_cut += 1;
        }
        remaining.replace_range(start..end_cut, "");
        // Text after the splice shifted left to `start`; rescan from there.
        cursor = start;
    }
    out
}

/// Parse Gemma's native tool-call argument DSL into a `serde_json::Value`.
///
/// The chat template (`models/gemma-4-*-it-*/chat_template.jinja`,
/// `format_argument` macro at line 118-147) trains the model to emit:
///
/// - **string values** wrapped in the `escape_token` (default `<|"|>`),
///   e.g. `path:<|"|>/foo/bar<|"|>`. After `tokenizer.decode(_, true)`
///   strips that special token, the wire form decodes to `path:/foo/bar`
///   with no quote characters at all — `serde_json` can't parse it;
/// - **dict keys** as bare identifiers (`escape_keys=False` in the
///   assistant-call branch at line 252), so `{path:...,encoding:...}`
///   instead of JSON's `{"path":...,"encoding":...}`;
/// - booleans, numbers and arrays as plain JSON literals.
///
/// The parser pre-substitutes `escape_token` → `"`, then walks the input
/// recursive-descent style. Both string forms are accepted: `"foo"` (a
/// proper JSON string after substitution) and bare values that run until
/// the next structural delimiter (`,`, `}`, `]`, EOF). Barewords parsed
/// as JSON `true` / `false` / `null` / number become those literals;
/// anything else becomes a JSON string with surrounding whitespace
/// trimmed. Returns `None` on a malformed payload — callers fall back to
/// empty args, matching pre-fix behaviour on unrecoverable input.
fn normalize_gemma4_dsl_to_json(payload: &str, escape_token: &str) -> String {
    parse_gemma4_dsl_to_value(payload, escape_token)
        .map(|v| serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|| "{}".to_string())
}

fn parse_gemma4_dsl_to_value(payload: &str, escape_token: &str) -> Option<Value> {
    let escaped = if escape_token.is_empty() {
        payload.to_string()
    } else {
        payload.replace(escape_token, "\"")
    };
    let chars: Vec<char> = escaped.chars().collect();
    let mut i = 0;
    let v = dsl_parse_value(&chars, &mut i)?;
    dsl_skip_ws(&chars, &mut i);
    if i == chars.len() {
        Some(v)
    } else {
        // Trailing garbage — accept it (matches lenient prior behaviour).
        Some(v)
    }
}

fn dsl_skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn dsl_parse_value(chars: &[char], i: &mut usize) -> Option<Value> {
    dsl_skip_ws(chars, i);
    if *i >= chars.len() {
        return None;
    }
    match chars[*i] {
        '{' => dsl_parse_object(chars, i),
        '[' => dsl_parse_array(chars, i),
        '"' => dsl_parse_string(chars, i).map(Value::String),
        _ => dsl_parse_bareword(chars, i),
    }
}

fn dsl_parse_object(chars: &[char], i: &mut usize) -> Option<Value> {
    debug_assert!(chars[*i] == '{');
    *i += 1;
    let mut map = Map::new();
    loop {
        dsl_skip_ws(chars, i);
        if *i >= chars.len() {
            return None;
        }
        if chars[*i] == '}' {
            *i += 1;
            return Some(Value::Object(map));
        }
        let key = dsl_parse_key(chars, i)?;
        dsl_skip_ws(chars, i);
        if *i >= chars.len() || chars[*i] != ':' {
            return None;
        }
        *i += 1;
        let value = dsl_parse_value(chars, i)?;
        map.insert(key, value);
        dsl_skip_ws(chars, i);
        if *i >= chars.len() {
            return None;
        }
        match chars[*i] {
            ',' => {
                *i += 1;
            }
            '}' => {
                *i += 1;
                return Some(Value::Object(map));
            }
            _ => return None,
        }
    }
}

fn dsl_parse_array(chars: &[char], i: &mut usize) -> Option<Value> {
    debug_assert!(chars[*i] == '[');
    *i += 1;
    let mut arr = Vec::new();
    loop {
        dsl_skip_ws(chars, i);
        if *i >= chars.len() {
            return None;
        }
        if chars[*i] == ']' {
            *i += 1;
            return Some(Value::Array(arr));
        }
        arr.push(dsl_parse_value(chars, i)?);
        dsl_skip_ws(chars, i);
        if *i >= chars.len() {
            return None;
        }
        match chars[*i] {
            ',' => {
                *i += 1;
            }
            ']' => {
                *i += 1;
                return Some(Value::Array(arr));
            }
            _ => return None,
        }
    }
}

fn dsl_parse_string(chars: &[char], i: &mut usize) -> Option<String> {
    debug_assert!(chars[*i] == '"');
    *i += 1;
    let mut s = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        if c == '\\' {
            *i += 1;
            if *i >= chars.len() {
                return None;
            }
            // Honour the JSON escape set; unknown escapes pass through.
            let esc = match chars[*i] {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => '\u{0008}',
                'f' => '\u{000c}',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            };
            s.push(esc);
            *i += 1;
        } else if c == '"' {
            *i += 1;
            return Some(s);
        } else {
            s.push(c);
            *i += 1;
        }
    }
    None
}

fn dsl_parse_key(chars: &[char], i: &mut usize) -> Option<String> {
    dsl_skip_ws(chars, i);
    if *i < chars.len() && chars[*i] == '"' {
        return dsl_parse_string(chars, i);
    }
    let start = *i;
    while *i < chars.len()
        && (chars[*i].is_ascii_alphanumeric() || matches!(chars[*i], '_' | '-' | '.'))
    {
        *i += 1;
    }
    if start == *i {
        None
    } else {
        Some(chars[start..*i].iter().collect())
    }
}

/// Consume a bareword that runs until the next structural delimiter
/// (`,`, `}`, `]`, EOF). Interprets the trimmed result as `true`/`false`/
/// `null`/number when it matches, otherwise returns a JSON string.
fn dsl_parse_bareword(chars: &[char], i: &mut usize) -> Option<Value> {
    let start = *i;
    while *i < chars.len() && !matches!(chars[*i], ',' | '}' | ']') {
        *i += 1;
    }
    let raw: String = chars[start..*i].iter().collect();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(match trimmed {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        other => {
            if let Ok(n) = other.parse::<i64>() {
                Value::Number(n.into())
            } else if let Ok(n) = other.parse::<u64>() {
                Value::Number(n.into())
            } else if let Ok(f) = other.parse::<f64>() {
                match serde_json::Number::from_f64(f) {
                    Some(n) => Value::Number(n),
                    None => Value::String(other.to_string()),
                }
            } else {
                Value::String(other.to_string())
            }
        }
    })
}

fn parse_tool_call_payload(payload: &str, escape_token: &str) -> Result<Gemma4ToolCall> {
    if let Some(rest) = payload.strip_prefix("call:") {
        let brace_idx = rest.find('{').ok_or_else(|| {
            Error::Model("Gemma4 tool call payload is missing JSON arguments".to_string())
        })?;
        let name = rest[..brace_idx].trim();
        if name.is_empty() {
            return Err(Error::Model(
                "Gemma4 tool call payload is missing a tool name".to_string(),
            ));
        }
        let body = &rest[brace_idx..];
        // JSON first, DSL on miss (mirrors `extract_bare_call_payloads`).
        let arguments = match serde_json::from_str::<Value>(body) {
            Ok(v) => v,
            Err(_) => {
                let normalised = normalize_gemma4_dsl_to_json(body, escape_token);
                serde_json::from_str(&normalised)?
            }
        };
        return Ok(Gemma4ToolCall::new(name, arguments));
    }

    let value: Value = serde_json::from_str(payload)?;
    parse_json_tool_call(&value)
}

fn parse_json_tool_call(value: &Value) -> Result<Gemma4ToolCall> {
    let obj = value.as_object().ok_or_else(|| {
        Error::Model("Gemma4 tool call payload must be a JSON object".to_string())
    })?;

    if let (Some(name), Some(arguments)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("arguments"),
    ) {
        return Ok(Gemma4ToolCall::new(name, arguments.clone()));
    }
    if let (Some(name), Some(arguments)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("parameters"),
    ) {
        return Ok(Gemma4ToolCall::new(name, arguments.clone()));
    }
    if let Some(function) = obj.get("function").and_then(Value::as_object) {
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::Model("Gemma4 nested tool call is missing function.name".to_string())
            })?;
        let arguments = function.get("arguments").ok_or_else(|| {
            Error::Model("Gemma4 nested tool call is missing function.arguments".to_string())
        })?;
        let arguments = match arguments {
            Value::String(raw) => serde_json::from_str(raw)?,
            other => other.clone(),
        };
        return Ok(Gemma4ToolCall::new(name, arguments));
    }

    Err(Error::Model(
        "Gemma4 tool call JSON must contain name+arguments, name+parameters, or function.{name,arguments}".to_string(),
    ))
}

fn run_tool_loop_with_generator<F>(
    _template: &Gemma4ChatTemplate,
    conversation: &mut Gemma4Conversation,
    tools: &Gemma4ToolRegistry,
    config: &Gemma4ChatConfig,
    mut generate: F,
) -> Result<Gemma4ChatResponse>
where
    F: FnMut(&[Gemma4Message], &[Gemma4ToolSpec]) -> Result<GeneratedAssistantTurn>,
{
    let mut tokens_generated = 0;
    let mut all_tool_calls = Vec::new();
    let mut all_tool_results = Vec::new();
    let mut tool_iterations = 0usize;

    loop {
        let tool_specs = tools.specs();
        let generated = generate(&conversation.messages, &tool_specs)?;
        tokens_generated += generated.tokens_generated;

        let assistant_message = Gemma4Message {
            role: Gemma4Role::Assistant,
            content: generated.parsed.content.clone(),
            thinking: generated.parsed.thinking.clone(),
            tool_calls: generated.parsed.tool_calls.clone(),
            tool_name: None,
            tool_call_id: None,
        };
        conversation.add_assistant_message(assistant_message);

        if generated.parsed.tool_calls.is_empty() {
            return Ok(Gemma4ChatResponse {
                raw_text: generated.parsed.raw_text,
                text: generated.parsed.content,
                thinking: generated.parsed.thinking,
                tool_calls: all_tool_calls,
                tool_results: all_tool_results,
                tokens_generated,
            });
        }

        if tool_iterations >= config.max_tool_iterations {
            return Err(Error::Model(
                "Gemma4 chat reached max_tool_iterations before producing a final assistant response"
                    .to_string(),
            ));
        }
        tool_iterations += 1;

        let round_calls = generated.parsed.tool_calls;
        let round_results: Vec<_> = round_calls
            .iter()
            .map(|call| tools.execute_call(call))
            .collect();
        for result in &round_results {
            conversation.add_tool_result(result);
        }
        all_tool_calls.extend(round_calls);
        all_tool_results.extend(round_results);
    }
}

pub struct Gemma4ChatPipeline {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub template: Gemma4ChatTemplate,
    pub config: Gemma4ChatConfig,
    pub tools: Gemma4ToolRegistry,
}

impl Gemma4ChatPipeline {
    pub fn load(model_dir: impl AsRef<Path>, config: Gemma4ChatConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Ok(Self {
            model: load_model(model_dir)?,
            tokenizer: load_tokenizer(model_dir)?,
            template: Gemma4ChatTemplate::load(model_dir)?,
            config,
            tools: Gemma4ToolRegistry::new(),
        })
    }

    pub fn register_tool<T>(&mut self, tool: T)
    where
        T: Gemma4Tool + 'static,
    {
        self.tools.register(tool);
    }

    pub fn chat(&mut self, conversation: &mut Gemma4Conversation) -> Result<Gemma4ChatResponse> {
        let template = self.template.clone();
        let config = self.config.clone();
        let tokenizer = self.tokenizer.clone();

        run_tool_loop_with_generator(
            &template,
            conversation,
            &self.tools,
            &config,
            |messages, tool_specs| {
                generate_assistant_turn(
                    &mut self.model,
                    &tokenizer,
                    &template,
                    messages,
                    tool_specs,
                    &config,
                )
            },
        )
    }
}

fn generate_assistant_turn(
    model: &mut Model,
    tokenizer: &Tokenizer,
    template: &Gemma4ChatTemplate,
    messages: &[Gemma4Message],
    tool_specs: &[Gemma4ToolSpec],
    config: &Gemma4ChatConfig,
) -> Result<GeneratedAssistantTurn> {
    let prompt = template.render_prompt(messages, tool_specs, true)?;
    let raw_text = generate_raw_text(model, tokenizer, &prompt, config, template)?;
    let parsed = template.parse_assistant_response(&raw_text.0)?;
    Ok(GeneratedAssistantTurn::new(parsed, raw_text.1))
}

/// EOS token IDs from Gemma4 generation_config.json.
/// Includes: eos (1), turn_end (106), pad (50).
pub const EOS_TOKEN_IDS: &[u32] = &[1, 106, 50];

fn generate_raw_text(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &Gemma4ChatConfig,
    _template: &Gemma4ChatTemplate,
) -> Result<(String, usize)> {
    let encoding = tokenizer.encode(prompt, false)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(model, &mut cache, config.temperature, &prompt_tokens);

    let mut generated_ids = Vec::new();
    for token in generator.take(config.max_new_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();
        generated_ids.push(token_id);

        // Token-level stop: EOS (1), turn_end (106), pad (50)
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
    }

    let decoded = tokenizer.decode(&generated_ids, false)?;
    Ok((decoded, generated_ids.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Inline Jinja that mirrors the *shape* of the shipped Gemma4
    /// `chat_template.jinja` for the bits we care about: role-rewriting
    /// (`assistant` → `model`), system-turn emission, user/assistant
    /// turn markers, and the empty-channel pre-fill on
    /// `add_generation_prompt`. Keeps the test self-contained and avoids
    /// depending on the developer's local model dirs.
    const JINJA_FIXTURE_LIKE_12B: &str = "\
{{- bos_token -}}
{%- if messages[0]['role'] in ['system', 'developer'] -%}
{{- '<|turn>system\n' -}}{{- messages[0]['content'] | trim -}}{{- '\n<turn|>\n' -}}
{%- set loop_messages = messages[1:] -%}
{%- else -%}
{%- set loop_messages = messages -%}
{%- endif -%}
{%- for m in loop_messages -%}
{%- set role = 'model' if m['role'] == 'assistant' else m['role'] -%}
{{- '<|turn>' + role + '\n' -}}{{- m['content'] | trim -}}{{- '\n<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\n' -}}{{- '<|channel>thought\n<channel|>' -}}
{%- endif -%}";

    fn fixture_template(jinja: Option<&str>) -> Gemma4ChatTemplate {
        Gemma4ChatTemplate {
            tokens: Gemma4SpecialTokens::default(),
            default_system_prompt: "You are a helpful assistant.".to_string(),
            jinja_source: jinja.map(str::to_string),
        }
    }

    #[test]
    fn render_via_jinja_rewrites_assistant_role_to_model() {
        let template = fixture_template(Some(JINJA_FIXTURE_LIKE_12B));
        let messages = [
            Gemma4Message::user("hi"),
            Gemma4Message::assistant("hello back"),
            Gemma4Message::user("again"),
        ];
        let rendered = template.render_prompt(&messages, &[], true).unwrap();
        // assistant turn must render with the trained role token `model`
        assert!(
            rendered.contains("<|turn>model\nhello back"),
            "rendered output should rewrite assistant→model: {rendered:?}"
        );
        // and final generation prompt must also be `model`, not `assistant`
        assert!(rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn render_via_jinja_injects_default_system_when_missing() {
        let template = fixture_template(Some(JINJA_FIXTURE_LIKE_12B));
        let messages = [Gemma4Message::user("hi")];
        let rendered = template.render_prompt(&messages, &[], true).unwrap();
        // We prepend the default system prompt so the Jinja's system-turn
        // branch fires (the template only emits one when messages[0] is
        // system/developer or tools/enable_thinking is set).
        assert!(rendered.contains("<|turn>system\nYou are a helpful assistant."));
    }

    #[test]
    fn render_via_jinja_preserves_caller_supplied_system_prompt() {
        let template = fixture_template(Some(JINJA_FIXTURE_LIKE_12B));
        let messages = [
            Gemma4Message::system("Custom rules."),
            Gemma4Message::user("hi"),
        ];
        let rendered = template.render_prompt(&messages, &[], true).unwrap();
        assert!(rendered.contains("<|turn>system\nCustom rules."));
        assert!(!rendered.contains("You are a helpful assistant."));
    }

    #[test]
    fn jinja_render_failure_falls_back_to_rust_renderer() {
        // Syntactically invalid Jinja → render errors → Rust fallback runs.
        let template = fixture_template(Some("{{ unclosed"));
        let messages = [Gemma4Message::user("hi")];
        let rendered = template.render_prompt(&messages, &[], true).unwrap();
        // The Rust path emits `<|turn>model\n…` (post-fix), so confirm we
        // got the fallback's full system+user+gen-prompt structure.
        assert!(rendered.starts_with("<bos><|turn>system\n"));
        assert!(rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    /// Smoke-test the real shipped templates. Skips if the developer's
    /// local model dirs aren't present (CI may not have them); when they
    /// are, asserts each template renders a trivial chat without erroring
    /// and emits `<|turn>model\n` for the assistant turn.
    #[test]
    fn real_shipped_chat_templates_render_without_error() {
        let candidates = [
            // (model-dir name, expects empty-channel pre-fill on generation)
            ("gemma-4-12B-it-4bit", true),
            ("gemma-4-e4b-it-4bit", false), // e4b's Jinja omits the pre-fill
            ("gemma4-26B-a4b-it-UD-MLX-4bit", true),
        ];
        // Model roots, most specific first: explicit override, the
        // workspace-local ./models dir, then the shared ~/.OminiX cache.
        // The test skips (loudly, see below) when none contain a candidate.
        let parents = [
            std::env::var_os("OMINIX_MODELS_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_default(),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"),
            std::env::var_os("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".OminiX/models"))
                .unwrap_or_default(),
        ];
        let mut any_seen = false;
        for (name, expect_pre_fill) in candidates {
            let Some(dir) = parents
                .iter()
                .map(|p| p.join(name))
                .find(|p| p.join("chat_template.jinja").exists())
            else {
                continue;
            };
            any_seen = true;
            let template = Gemma4ChatTemplate::load(&dir)
                .unwrap_or_else(|e| panic!("load {name}: {e}"));
            assert!(
                template.jinja_source.is_some(),
                "{name}: jinja_source should be populated from chat_template.jinja"
            );
            let messages = [
                Gemma4Message::user("ping"),
                Gemma4Message::assistant("pong"),
                Gemma4Message::user("ping again"),
            ];
            let rendered = template
                .render_prompt(&messages, &[], true)
                .unwrap_or_else(|e| panic!("render {name}: {e}"));
            assert!(
                rendered.contains("<|turn>model\npong"),
                "{name}: assistant turn must render as <|turn>model\\n (got {rendered:?})"
            );
            let has_pre_fill =
                rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>");
            assert_eq!(
                has_pre_fill, expect_pre_fill,
                "{name}: empty-channel pre-fill mismatch (rendered tail: {:?})",
                &rendered[rendered.len().saturating_sub(80)..]
            );
        }
        if !any_seen {
            eprintln!(
                "skipping real-template smoke test: no local Gemma4 model dirs found \
                 under $OMINIX_MODELS_DIR, <workspace>/models, or ~/.OminiX/models"
            );
        }
    }

    /// Regression for the "thought" leak that survived the
    /// `skip_special_tokens=false` decode fix: extract_thinking required
    /// the channel label to be EXACTLY `thought\n`. If the model emitted
    /// a channel with any other label (e.g. `analysis`, `final`,
    /// `thought ` with a trailing space), the entire `<|channel>…<channel|>`
    /// block was left in `content` and the client saw it as plain text.
    /// The Jinja template's own `strip_thinking` macro is label-agnostic
    /// — it splits on `<channel|>` and drops everything from the preceding
    /// `<|channel>` regardless of label. The parser must do the same.
    #[test]
    fn parse_assistant_response_strips_channel_block_with_unknown_label() {
        let template = Gemma4ChatTemplate::default();
        let raw = "<|channel>analysis\nweighing options...\n<channel|>\nThe answer is 42.<turn|>";
        let parsed = template.parse_assistant_response(raw).unwrap();
        // The `analysis` block isn't reasoning we surface (Gemma4's
        // canonical reasoning channel is `thought`), so we don't expose
        // it as `thinking`. But we MUST strip it from content — leaking
        // `analysis\nweighing options...\n` to the user is a quality bug.
        assert!(
            !parsed.content.contains("analysis"),
            "non-`thought` channel must be stripped from content, got {:?}",
            parsed.content,
        );
        assert!(
            !parsed.content.contains("weighing options"),
            "non-`thought` channel body must be stripped, got {:?}",
            parsed.content,
        );
        assert_eq!(parsed.content, "The answer is 42.");
    }

    /// Same case but the model emits multiple channel blocks in one
    /// response. All blocks must be stripped from content; the first
    /// `thought` block (if any) becomes `thinking`.
    #[test]
    fn parse_assistant_response_strips_all_channel_blocks() {
        let template = Gemma4ChatTemplate::default();
        let raw = concat!(
            "<|channel>thought\nplanning\n<channel|>\n",
            "<|channel>analysis\nchecking work\n<channel|>\n",
            "The final answer is 42.<turn|>",
        );
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(
            parsed.thinking.as_deref().map(str::trim),
            Some("planning"),
            "first `thought` channel becomes thinking"
        );
        assert_eq!(parsed.content, "The final answer is 42.");
        assert!(!parsed.content.contains("<|channel>"));
        assert!(!parsed.content.contains("analysis"));
    }

    /// Truncated channel (no close marker — model hit max_tokens
    /// mid-reasoning): drop everything from `<|channel>` to EOF rather
    /// than erroring out the whole response.
    #[test]
    fn parse_assistant_response_drops_unclosed_channel_to_eof() {
        let template = Gemma4ChatTemplate::default();
        let raw = "Some preamble.<|channel>thought\nstill thinking...";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.content, "Some preamble.");
        // No close marker → no reliable thinking text; we surface what
        // we captured as a courtesy so the client can see partial
        // reasoning rather than nothing.
        assert!(
            parsed
                .thinking
                .as_deref()
                .map(|t| t.contains("still thinking"))
                .unwrap_or(false),
            "partial reasoning should be surfaced when channel is unclosed; got {:?}",
            parsed.thinking,
        );
    }

    /// Demonstrates *why* callers of `parse_assistant_response` must hand
    /// it text decoded with `skip_special_tokens=false`. When the
    /// tokenizer ate the `<|channel>` / `<channel|>` wrappers, this
    /// parser has nothing to match on — the inner `thought\n…` plaintext
    /// leaks straight into `content`. OminiX-API's
    /// `finalize_generate_output` now decodes the Gemma4 path with
    /// `skip_special_tokens=false` so the surface markers survive and
    /// `extract_thinking` can do its job.
    #[test]
    fn parse_assistant_response_cannot_recover_post_special_token_strip() {
        let template = Gemma4ChatTemplate::default();
        // What `tokenizer.decode(_, skip_special=true)` produces when the
        // model emitted `<|channel>thought\nweighing…\n<channel|>\nThe
        // answer is 42.` — wrappers gone, inner text survives.
        let raw = "thought\nweighing the options\n\nThe answer is 42.";
        let parsed = template.parse_assistant_response(raw).unwrap();
        // No channel markers → no thinking extracted.
        assert!(
            parsed.thinking.is_none(),
            "parser cannot recover thinking from stripped text"
        );
        // Plaintext leak — this is exactly what the user reports.
        assert!(
            parsed.content.starts_with("thought\n"),
            "leak: stripped `thought\\n` plaintext lands in content. \
             Decode with skip_special_tokens=false to avoid this. \
             Actual content: {:?}",
            parsed.content,
        );
    }

    /// Companion: when the caller decodes with `skip_special_tokens=false`
    /// (as `finalize_generate_output` does now), the wrappers survive and
    /// extract_thinking cleanly partitions reasoning from visible content.
    /// No `thought\n…` ever lands in `content`.
    #[test]
    fn parse_assistant_response_extracts_cleanly_when_decode_keeps_special_tokens() {
        let template = Gemma4ChatTemplate::default();
        let raw = "<|channel>thought\nweighing the options\n<channel|>\nThe answer is 42.<turn|>";
        let parsed = template.parse_assistant_response(raw).unwrap();
        // `extract_thinking` slices on the literal `\n<channel|>` close; the
        // trailing newline of the inner block lands in the `thinking` field
        // unless the model puts content immediately after the close. Match
        // either spelling.
        let thinking = parsed.thinking.as_deref().unwrap_or("");
        assert!(
            thinking.trim() == "weighing the options",
            "thinking must be extracted (got {:?})",
            thinking
        );
        assert_eq!(parsed.content, "The answer is 42.");
        assert!(parsed.tool_calls.is_empty());
    }

    /// Regression: when the tokenizer decodes with `skip_special_tokens=true`
    /// (which it does on every text path), the `<|tool_call>` / `<tool_call|>`
    /// wrappers around an emitted call are stripped — what reaches our
    /// parser is just `call:name{json}`. Before the fix,
    /// `extract_tool_calls` only matched the wrapper literals and returned
    /// `[]`, so the `call:terminal{...}` payload leaked into
    /// `parsed.content` and the client saw it as plain text instead of a
    /// dispatched tool call (observed via Hermes-gateway with Gemma4-12B).
    #[test]
    fn parse_assistant_response_extracts_bare_call_after_special_token_strip() {
        let template = Gemma4ChatTemplate::default();
        let raw = "Let me check the directory.\ncall:terminal{\"cmd\":\"ls -la\"}";
        let parsed = template
            .parse_assistant_response(raw)
            .expect("parse should not error on a bare call payload");
        assert_eq!(
            parsed.tool_calls.len(),
            1,
            "bare `call:` payload must be extracted, not left in content; got content={:?}",
            parsed.content,
        );
        assert_eq!(parsed.tool_calls[0].name, "terminal");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"cmd": "ls -la"}),
        );
        assert_eq!(parsed.content, "Let me check the directory.");
    }

    /// Same shape with nested-object args (catches naive `find('}')`
    /// implementations that would terminate at the first close-brace).
    #[test]
    fn parse_assistant_response_extracts_bare_call_with_nested_object_args() {
        let template = Gemma4ChatTemplate::default();
        let raw = "call:create_file{\"path\":\"a.txt\",\"meta\":{\"size\":4,\"flags\":[1,2]}}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "create_file");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"path": "a.txt", "meta": {"size": 4, "flags": [1, 2]}}),
        );
        assert_eq!(parsed.content, "");
    }

    /// Multiple bare calls in one response, separated by newlines —
    /// matches the Jinja's loop that emits one tool-call block per call.
    #[test]
    fn parse_assistant_response_extracts_multiple_bare_calls() {
        let template = Gemma4ChatTemplate::default();
        let raw = "Doing two things.\ncall:ls{\"path\":\"/\"}\ncall:pwd{}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "ls");
        assert_eq!(parsed.tool_calls[1].name, "pwd");
        assert_eq!(parsed.tool_calls[1].arguments, json!({}));
        assert_eq!(parsed.content, "Doing two things.");
    }

    /// Don't false-positive on prose that happens to contain `call:`.
    /// "callback:" must not be picked up as a tool call, and a bare
    /// "call:" with no `{...}` is just text.
    #[test]
    fn parse_assistant_response_ignores_non_tool_call_uses_of_call_prefix() {
        let template = Gemma4ChatTemplate::default();
        let raw = "I'll use callback:foo soon. (call: see notes)";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 0);
        assert_eq!(parsed.content, raw);
    }

    /// Truncated bare call (model hit max_tokens mid-args) — the call
    /// should still dispatch with `{}` rather than leak `call:name{partial`
    /// into the visible response.
    #[test]
    fn parse_assistant_response_extracts_truncated_bare_call_as_empty_args() {
        let template = Gemma4ChatTemplate::default();
        let raw = "call:terminal{\"cmd\":\"ls -la --color";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "terminal");
        assert_eq!(parsed.tool_calls[0].arguments, json!({}));
        assert_eq!(parsed.content, "");
    }

    /// Bug: hermes-gateway saw `<|tool_call>` leaking into content. Cause:
    /// tokenizer stripped `<tool_call|>` (end) but kept `<|tool_call>` (start),
    /// or generation was truncated before the close. The wrapped-form scan
    /// previously just `break`ed when no end marker was found, leaving the
    /// orphaned start in `remaining`. Bare-form then spliced `call:name{...}`
    /// out of that, leaving only `<|tool_call>` which is neither `turn_end`
    /// nor `eos` and fell through to `content`.
    #[test]
    fn parse_assistant_response_no_leak_when_tool_call_end_absent() {
        let template = Gemma4ChatTemplate::default();
        // `<tool_call|>` end marker absent — asymmetric strip or truncation.
        let raw = "<|tool_call>call:terminal{\"cmd\":\"ls -la\"}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1, "call must still be extracted");
        assert_eq!(parsed.tool_calls[0].name, "terminal");
        assert!(
            !parsed.content.contains("<|tool_call>"),
            "<|tool_call> must not leak into content; got {:?}",
            parsed.content,
        );
        assert_eq!(parsed.content, "", "no visible text should remain");
    }

    /// Complementary: tokenizer stripped `<|tool_call>` (start) but kept
    /// `<tool_call|>` (end). Bare-form extracts `call:name{...}`, leaving the
    /// orphaned end marker. It must not leak.
    #[test]
    fn parse_assistant_response_no_leak_when_tool_call_start_absent() {
        let template = Gemma4ChatTemplate::default();
        // `<|tool_call>` start absent — only end marker survives decode.
        let raw = "call:pwd{}<tool_call|>";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1, "call must still be extracted");
        assert_eq!(parsed.tool_calls[0].name, "pwd");
        assert!(
            !parsed.content.contains("<tool_call|>"),
            "<tool_call|> must not leak into content; got {:?}",
            parsed.content,
        );
        assert_eq!(parsed.content, "");
    }

    /// Bug 1 regression. `tokenizer.decode(_, skip_special_tokens=true)` drops
    /// every `<|"|>` (Gemma's `escape_token`), so a wire-form
    /// `<|tool_call>call:read_file{path:<|"|>/Users/foo/bar.txt<|"|>}<tool_call|>`
    /// reaches us as a bare call with **un-quoted** values:
    /// `call:read_file{path:/Users/foo/bar.txt}`. Pre-fix that JSON-parsed
    /// as a `serde_json::Error` and the call dispatched with empty args —
    /// every path-bearing tool (`Read`, `Bash`, `Edit`) then failed at the
    /// client with "missing required parameter".
    #[test]
    fn parse_assistant_response_recovers_path_from_native_dsl_after_special_token_strip() {
        let template = Gemma4ChatTemplate::default();
        let raw = "call:read_file{path:/Users/kyle/repos/foo/bar.rs}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"path": "/Users/kyle/repos/foo/bar.rs"}),
            "the path argument must survive Gemma's native-DSL → JSON normalisation",
        );
        assert_eq!(parsed.content, "");
    }

    /// Same Gemma DSL but with the `<|"|>` escape tokens still present
    /// (what we see when `skip_special_tokens=false`, or when the model
    /// emitted the bracket sequences as literal text).
    #[test]
    fn parse_assistant_response_recovers_path_from_native_dsl_with_escape_tokens() {
        let template = Gemma4ChatTemplate::default();
        let raw = "call:read_file{path:<|\"|>/Users/kyle/with space/file.txt<|\"|>}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"path": "/Users/kyle/with space/file.txt"}),
        );
    }

    /// Multi-arg DSL with a mix of strings, an int, and an array — covers
    /// the `format_argument` macro's full menu (chat_template.jinja:118).
    #[test]
    fn parse_assistant_response_recovers_mixed_native_dsl_args() {
        let template = Gemma4ChatTemplate::default();
        let raw = "call:Edit{file_path:/tmp/x.rs,old_string:foo,new_string:bar,replace_all:true}";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "Edit");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({
                "file_path": "/tmp/x.rs",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": true,
            }),
        );
    }

    /// `<|tool_call>...<tool_call|>` wrapped DSL — exercises the
    /// `extract_tool_calls` wrapped branch (line 806).
    #[test]
    fn parse_assistant_response_recovers_wrapped_native_dsl() {
        let template = Gemma4ChatTemplate::default();
        let raw = "<|tool_call>call:Bash{command:ls -la,timeout:5000}<tool_call|>";
        let parsed = template.parse_assistant_response(raw).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "Bash");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"command": "ls -la", "timeout": 5000}),
        );
    }

    /// The normaliser must be a no-op on already-quoted JSON: identifiers
    /// inside quoted strings must NOT get re-quoted, and the structure
    /// must round-trip byte-equal (modulo serde reformatting).
    #[test]
    fn normalize_gemma4_dsl_to_json_passes_through_valid_json() {
        let valid = r#"{"path":"/foo","note":"call:not_a_tool"}"#;
        let out = super::normalize_gemma4_dsl_to_json(valid, "<|\"|>");
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed, json!({"path": "/foo", "note": "call:not_a_tool"}));
    }

    /// Don't re-quote identifiers that appear *inside* string values — a
    /// substring like `"description":"path of file"` must survive intact.
    #[test]
    fn normalize_gemma4_dsl_to_json_leaves_in_string_identifiers_alone() {
        let mixed = r#"{key:<|"|>value with key: inside<|"|>}"#;
        let out = super::normalize_gemma4_dsl_to_json(mixed, "<|\"|>");
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed, json!({"key": "value with key: inside"}));
    }

    #[test]
    fn render_prompt_uses_default_system_prompt_and_generation_prompt() {
        let template = Gemma4ChatTemplate::default();
        let messages = [Gemma4Message::user("Hello Gemma")];

        let rendered = template.render_prompt(&messages, &[], true).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<bos><|turn>system\n",
                "You are a helpful assistant.\n",
                "<turn|>\n",
                "<|turn>user\n",
                "Hello Gemma\n",
                "<turn|>\n",
                "<|turn>model\n",
                "<|channel>thought\n<channel|>",
            )
        );
    }

    #[test]
    fn render_prompt_includes_tools_assistant_tool_calls_and_tool_responses() {
        let template = Gemma4ChatTemplate::default();
        let tools = [Gemma4ToolSpec::new(
            "weather",
            "Lookup weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )];
        let messages = [
            Gemma4Message::system("You are a weather assistant."),
            Gemma4Message::user("What is the weather in Paris?"),
            Gemma4Message::assistant_with_tool_calls(
                "Let me check.",
                vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))],
            ),
            Gemma4Message::tool(
                "weather",
                json!({"temp_c": 20, "condition": "sunny"}).to_string(),
            ),
        ];

        let rendered = template.render_prompt(&messages, &tools, true).unwrap();

        assert!(rendered.starts_with("<bos><|turn>system\nYou are a weather assistant.\n"));
        assert!(rendered.contains(
            "Available tools:\n<|tool>{\"description\":\"Lookup weather\",\"name\":\"weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}}<tool|>\n"
        ));
        assert!(rendered
            .contains("<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>\n<turn|>\n"));
        assert!(rendered.contains(
            "<|turn>tool\nweather\n<|tool_response>{\"content\":{\"condition\":\"sunny\",\"temp_c\":20},\"name\":\"weather\"}<tool_response|>\n<turn|>\n"
        ));
        assert!(rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn parse_assistant_response_extracts_thinking_content_and_tool_calls() {
        let template = Gemma4ChatTemplate::default();
        let raw = concat!(
            "<|channel>thought\n",
            "Need to look this up",
            "<channel|>\n",
            "Let me check.\n",
            "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>",
            "<turn|>"
        );

        let parsed = template.parse_assistant_response(raw).unwrap();

        assert_eq!(parsed.thinking.as_deref(), Some("Need to look this up"));
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(
            parsed.tool_calls,
            vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))]
        );
    }

    #[test]
    fn parse_assistant_response_accepts_json_tool_call_payloads() {
        let template = Gemma4ChatTemplate::default();
        let raw = concat!(
            "<|tool_call>",
            "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}",
            "<tool_call|><turn|>"
        );

        let parsed = template.parse_assistant_response(raw).unwrap();

        assert_eq!(parsed.content, "");
        assert_eq!(
            parsed.tool_calls,
            vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))]
        );
    }

    #[test]
    fn has_complete_tool_call_requires_end_marker() {
        let template = Gemma4ChatTemplate::default();

        assert!(template
            .has_complete_tool_call("<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>"));
        assert!(!template.has_complete_tool_call("<|tool_call>call:weather{\"city\":\"Paris\"}"));
    }

    #[test]
    fn parse_assistant_response_strips_trailing_eos_token() {
        let template = Gemma4ChatTemplate::default();

        let parsed = template.parse_assistant_response("Done.<eos>").unwrap();

        assert_eq!(parsed.content, "Done.");
    }

    #[test]
    fn tool_loop_executes_tools_and_continues_to_final_answer() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("What is the weather in Paris?");

        let mut tools = Gemma4ToolRegistry::new();
        tools.register(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new(
                "weather",
                "Lookup weather",
                json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            ),
            |arguments| {
                Ok(json!({
                    "city": arguments["city"].clone(),
                    "temp_c": 20,
                    "condition": "sunny"
                }))
            },
        ));

        let config = Gemma4ChatConfig {
            max_tool_iterations: 2,
            ..Default::default()
        };
        let mut generation_count = 0;

        let response = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |messages, tool_specs| {
                generation_count += 1;
                match generation_count {
                    1 => {
                        assert_eq!(tool_specs.len(), 1);
                        assert!(messages.iter().any(|message| message.role == Gemma4Role::User));
                        Ok(GeneratedAssistantTurn::new(
                            Gemma4ParsedAssistantResponse {
                                raw_text: "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|><turn|>"
                                    .to_string(),
                                content: String::new(),
                                thinking: Some("Need weather data".to_string()),
                                tool_calls: vec![Gemma4ToolCall::new(
                                    "weather",
                                    json!({ "city": "Paris" }),
                                )],
                            },
                            7,
                        ))
                    }
                    2 => {
                        assert!(messages.iter().any(|message| {
                            message.role == Gemma4Role::Tool
                                && message.content.contains("\"temp_c\":20")
                        }));
                        Ok(GeneratedAssistantTurn::new(
                            Gemma4ParsedAssistantResponse {
                                raw_text: "It is 20C and sunny.<turn|>".to_string(),
                                content: "It is 20C and sunny.".to_string(),
                                thinking: None,
                                tool_calls: Vec::new(),
                            },
                            5,
                        ))
                    }
                    _ => panic!("unexpected generation count"),
                }
            },
        )
        .unwrap();

        assert_eq!(response.text, "It is 20C and sunny.");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_results.len(), 1);
        assert_eq!(response.tool_results[0].name, "weather");
        assert_eq!(response.tool_results[0].content["temp_c"], 20);
        assert_eq!(response.tokens_generated, 12);
    }

    #[test]
    fn tool_loop_returns_plain_assistant_response_without_tools() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("Say hello");
        let tools = Gemma4ToolRegistry::new();
        let config = Gemma4ChatConfig::default();

        let response = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |messages, tool_specs| {
                assert_eq!(messages.len(), 1);
                assert!(tool_specs.is_empty());
                Ok(GeneratedAssistantTurn::new(
                    Gemma4ParsedAssistantResponse {
                        raw_text: "Hello there!<turn|>".to_string(),
                        content: "Hello there!".to_string(),
                        thinking: None,
                        tool_calls: Vec::new(),
                    },
                    3,
                ))
            },
        )
        .unwrap();

        assert_eq!(response.text, "Hello there!");
        assert!(response.tool_calls.is_empty());
        assert!(response.tool_results.is_empty());
        assert_eq!(conversation.messages.len(), 2);
        assert_eq!(conversation.messages[1].role, Gemma4Role::Assistant);
    }

    #[test]
    fn tool_loop_errors_after_max_tool_iterations() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("Keep calling the tool");

        let mut tools = Gemma4ToolRegistry::new();
        tools.register(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new("weather", "Lookup weather", json!({"type": "object"})),
            |_| Ok(json!({"ok": true})),
        ));

        let config = Gemma4ChatConfig {
            max_tool_iterations: 1,
            ..Default::default()
        };
        let mut generation_count = 0;

        let error = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |_messages, _tool_specs| {
                generation_count += 1;
                Ok(GeneratedAssistantTurn::new(
                    Gemma4ParsedAssistantResponse {
                        raw_text: "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|><turn|>"
                            .to_string(),
                        content: String::new(),
                        thinking: None,
                        tool_calls: vec![Gemma4ToolCall::new(
                            "weather",
                            json!({ "city": "Paris" }),
                        )],
                    },
                    2,
                ))
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("max_tool_iterations"));
        assert_eq!(generation_count, 2);
    }
}
