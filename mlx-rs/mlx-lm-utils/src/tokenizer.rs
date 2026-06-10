// Port of `transformers` `PreTrainedTokenizerBase.apply_chat_template` /
// `render_jinja_template`. For the reference semantics (tools, documents,
// continue_final_message, assistant-token masks), see:
// https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py

use std::{
    collections::HashMap,
    fs::read_to_string,
    ops::{Deref, DerefMut},
    path::Path,
    str::FromStr,
};

use minijinja::{context, Environment, Template};
use serde::Serialize;
use tokenizers::Encoding;

use crate::error::Error;

/// Wrapper around [`tokenizers::Tokenizer`] and [`minijinja::Environment`]
/// providing more utilities.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
}

impl FromStr for Tokenizer {
    type Err = tokenizers::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        tokenizers::Tokenizer::from_str(s).map(Self::from_tokenizer)
    }
}

impl Tokenizer {
    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        Self {
            inner: tokenizer,
            env,
        }
    }

    pub fn from_file(file: impl AsRef<Path>) -> tokenizers::Result<Self> {
        tokenizers::Tokenizer::from_file(file).map(Self::from_tokenizer)
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> tokenizers::Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes).map(Self::from_tokenizer)
    }

    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        model_template: String,
        args: ApplyChatTemplateArgs<'a, I, R, T>,
    ) -> Result<Vec<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + ToString + 'a,
    {
        apply_chat_template(&mut self.env, model_template, args)
    }

    pub fn apply_chat_template_and_encode<'a, I, R, T>(
        &mut self,
        model_template: String,
        args: ApplyChatTemplateArgs<'a, I, R, T>,
    ) -> Result<Vec<Encoding>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + ToString + 'a,
    {
        let Self { inner, env } = self;

        let rendered_chats = apply_chat_template(env, model_template, args)?;
        inner
            .encode_batch(rendered_chats, false)
            .map_err(Into::into)
    }
}

impl Deref for Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize)]
pub enum Content {
    String(String),
    Map(HashMap<String, String>),
}

#[derive(Debug, Clone, Serialize)]
pub struct Conversation<R, T> {
    pub role: R,
    pub content: T,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Chat<'a, R, T> {
    Borrowed(&'a [Conversation<R, T>]),
    Owned(Vec<Conversation<R, T>>),
}

impl<R, T> Deref for Chat<'_, R, T> {
    type Target = [Conversation<R, T>];

    fn deref(&self) -> &Self::Target {
        match self {
            Chat::Borrowed(conversations) => conversations,
            Chat::Owned(conversations) => conversations,
        }
    }
}

impl<R, T> From<Vec<Conversation<R, T>>> for Chat<'_, R, T> {
    fn from(value: Vec<Conversation<R, T>>) -> Self {
        Chat::Owned(value)
    }
}

impl<'a, R, T> From<&'a [Conversation<R, T>]> for Chat<'a, R, T> {
    fn from(value: &'a [Conversation<R, T>]) -> Self {
        Chat::Borrowed(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Document {
    pub title: String,
    pub text: String,
}

pub enum Padding {
    Longest,
    MaxLength,
}

pub enum Truncation {
    MaxLength(usize),
}

#[derive(Default)]
pub struct ApplyChatTemplateArgs<'a, I, R = Role, T = String>
where
    I: IntoIterator<Item = Chat<'a, R, T>>,
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    // pub conversations: &'a [Conversation<R, T>],
    pub conversations: I,
    /// Tool JSON schemas exposed to the template as the `tools` variable.
    /// A tools-aware chat template (e.g. Hermes-style Qwen) renders these into
    /// the system prompt under `{% if tools %}`. `None` leaves `tools`
    /// undefined, matching a no-tools render.
    pub tools: Option<&'a serde_json::Value>,
    pub documents: Option<&'a [Document]>,
    pub model_id: &'a str,
    pub chat_template_id: Option<&'a str>,
    pub add_generation_prompt: Option<bool>,
    pub continue_final_message: Option<bool>,
}

pub fn load_model_chat_template_from_str(content: &str) -> std::io::Result<Option<String>> {
    serde_json::from_str::<serde_json::Value>(content)
        .map(|value| {
            value
                .get("chat_template")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
        })
        .map_err(Into::into)
}

pub fn load_model_chat_template_from_file(
    file: impl AsRef<Path>,
) -> std::io::Result<Option<String>> {
    let content = read_to_string(file)?;
    load_model_chat_template_from_str(&content)
}


pub fn apply_chat_template<'a, I, R, T>(
    env: &mut Environment<'static>,
    model_template: String,
    args: ApplyChatTemplateArgs<'a, I, R, T>,
) -> Result<Vec<String>, Error>
where
    I: IntoIterator<Item = Chat<'a, R, T>>,
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    let ApplyChatTemplateArgs {
        conversations,
        tools,
        documents,
        model_id,
        chat_template_id,
        add_generation_prompt,
        continue_final_message,
    } = args;

    let add_generation_prompt = add_generation_prompt.unwrap_or(false);
    let continue_final_message = continue_final_message.unwrap_or(false);

    let template = match chat_template_id {
        Some(chat_template_id) => env.get_template(chat_template_id)?,
        None => match env.get_template(model_id) {
            Ok(template) => template,
            Err(_) => {
                env.add_template_owned(model_id.to_owned(), model_template)?;
                env.get_template(model_id)
                    .expect("Newly added template must be present")
            }
        },
    };

    // TODO: allow return_generation_indices

    render_jinja_template(
        template,
        conversations,
        tools,
        documents,
        Some(add_generation_prompt),
        Some(continue_final_message),
    )
}

// TODO: render with assistant indices
fn render_jinja_template<'a, R, T>(
    template: Template,
    conversations: impl IntoIterator<Item = Chat<'a, R, T>>,
    tools: Option<&'a serde_json::Value>,
    documents: Option<&'a [Document]>,
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
) -> Result<Vec<String>, Error>
where
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    let add_generation_prompt = add_generation_prompt.unwrap_or(false);
    let continue_final_message = continue_final_message.unwrap_or(false);

    // TODO: what does checking for "messages" key do in the python code?
    let mut rendered = Vec::new();
    for chat in conversations {
        let mut rendered_chat = template.render(context! {
            messages => chat,
            tools => tools,
            documents => documents,
            add_generation_prompt => add_generation_prompt,
        })?;

        if continue_final_message {
            let Some(final_message) = chat.last().map(|chat| &chat.content) else {
                continue;
            };

            let final_message_str = final_message.to_string();

            if !rendered_chat.contains(final_message_str.trim()) {
                return Err(Error::FinalMsgNotInChat);
            }

            let final_msg_loc = rendered_chat.rfind(final_message_str.trim()).unwrap();
            // `final_msg_len` includes any trailing whitespace of the original
            // message. If the template trimmed that whitespace, the range can
            // run past the end of `rendered_chat` (or split a UTF-8 boundary),
            // so probe with `get` instead of indexing — `None` falls through
            // to the trimmed-length branch.
            let final_msg_len = final_message_str.trim_start().len();
            let untrimmed_match = rendered_chat
                .get(final_msg_loc..final_msg_loc + final_msg_len)
                .is_some_and(|s| s == final_message_str);
            rendered_chat = if untrimmed_match {
                // The template preserves spacing or the message doesn't have trailing spacing, so things are simple
                rendered_chat[..final_msg_loc + final_msg_len].to_string()
            } else {
                // The message has trailing spacing that was trimmed, so we must be more cautious
                rendered_chat[..final_msg_loc + final_message_str.trim().len()].to_string()
            };
        }
        rendered.push(rendered_chat);
    }

    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use minijinja::Environment;
    use std::path::PathBuf;

    use crate::tokenizer::{
        apply_chat_template, load_model_chat_template_from_file, ApplyChatTemplateArgs,
        Conversation, Role,
    };

    /// Returns the path to test fixtures. Uses TEST_MODEL_DIR env var if set,
    /// otherwise falls back to the fixtures bundled in the repo.
    fn fixtures_dir() -> PathBuf {
        std::env::var("TEST_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3")
            })
    }

    #[test]
    fn test_load_chat_template_from_file() {
        let file = fixtures_dir().join("tokenizer_config.json");
        let chat_template = load_model_chat_template_from_file(file).unwrap().unwrap();
        assert!(!chat_template.is_empty());
    }

    #[test]
    fn test_apply_chat_template() {
        let file = fixtures_dir().join("tokenizer_config.json");
        let model_chat_template = load_model_chat_template_from_file(file).unwrap().unwrap();
        assert!(!model_chat_template.is_empty());

        let model_id = "mlx-community/Qwen3-4B-bf16".to_string();
        let conversations = vec![Conversation {
            role: Role::User,
            content: "hello",
        }];
        let args = ApplyChatTemplateArgs {
            conversations: [conversations.into()],
            tools: None,
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };

        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        let rendered_chat = apply_chat_template(&mut env, model_chat_template, args).unwrap();
        println!("{:?}", rendered_chat);
    }

    // F16: tool schemas passed via `tools` must reach the template's `tools`
    // variable (previously the param was a stubbed TODO and silently dropped).
    #[test]
    fn test_apply_chat_template_passes_tools() {
        // Inline tools-aware template — no model fixture needed.
        let template = "{% if tools %}{% for t in tools %}TOOL:{{ t.function.name }}\n\
            {% endfor %}{% endif %}{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"
            .to_string();
        let model_id = "test/inline".to_string();
        let conversations = vec![Conversation {
            role: Role::User,
            content: "hi",
        }];
        let tools = serde_json::json!([
            { "type": "function", "function": { "name": "get_weather", "parameters": {} } }
        ]);
        let args = ApplyChatTemplateArgs {
            conversations: [conversations.into()],
            tools: Some(&tools),
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        let out = apply_chat_template(&mut env, template, args).unwrap();
        assert!(
            out[0].contains("TOOL:get_weather"),
            "tools must be injected into the template, got: {:?}",
            out[0]
        );
    }

    #[test]
    #[ignore = "requires local model files (tokenizer.json is 11MB)"]
    fn test_tokenizer_apply_chat_template() {
        let tokenizer_file = fixtures_dir().join("tokenizer.json");
        let tokenizer_config_file = fixtures_dir().join("tokenizer_config.json");

        let model_id = "mlx-community/Qwen3-4B-bf16".to_string();

        let conversations = vec![Conversation {
            role: Role::User,
            content: "hello",
        }];

        let mut tokenizer = super::Tokenizer::from_file(tokenizer_file).unwrap();

        let model_chat_template = load_model_chat_template_from_file(tokenizer_config_file)
            .unwrap()
            .unwrap();
        assert!(!model_chat_template.is_empty());

        let args = ApplyChatTemplateArgs {
            conversations: [conversations.into()],
            tools: None,
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };

        let rendered_chat = tokenizer
            .apply_chat_template(model_chat_template, args)
            .unwrap();
        println!("{:?}", rendered_chat);
    }

    #[test]
    #[ignore = "requires local model files (tokenizer.json is 11MB)"]
    fn test_tokenizer_apply_chat_template_and_encode() {
        let tokenizer_file = fixtures_dir().join("tokenizer.json");
        let tokenizer_config_file = fixtures_dir().join("tokenizer_config.json");

        let model_id = "mlx-community/Qwen3-4B-bf16".to_string();

        let conversations = vec![Conversation {
            role: Role::User,
            content: "hello",
        }];
        let mut tokenizer = super::Tokenizer::from_file(tokenizer_file).unwrap();

        let model_chat_template = load_model_chat_template_from_file(tokenizer_config_file)
            .unwrap()
            .unwrap();
        assert!(!model_chat_template.is_empty());

        let args = ApplyChatTemplateArgs {
            conversations: [conversations.into()],
            tools: None,
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };

        let encodings = tokenizer
            .apply_chat_template_and_encode(model_chat_template, args)
            .unwrap();
        println!("{:?}", encodings.iter().map(|e| e.get_ids()).flatten());
    }
}
