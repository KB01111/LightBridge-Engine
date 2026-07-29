//! Exact GGUF-backed tokenizer and chat semantics for the selected Hy3 model.

mod assistant;
mod chat;
mod error;
mod gguf;

use bridge_gguf::GgufFile;
use tokenizers::Tokenizer;

pub use assistant::{parse_assistant_output, AssistantOutput, AssistantParseError};
pub use chat::{
    format_chat, ChatMessage, ChatTemplateOptions, ReasoningEffort, ToolCall, ARGUMENT_KEY_BEGIN_TOKEN,
    ARGUMENT_KEY_END_TOKEN, ARGUMENT_VALUE_BEGIN_TOKEN, ARGUMENT_VALUE_END_TOKEN, ASSISTANT_TOKEN, BOS_TOKEN,
    EOS_TOKEN, REASONING_MODE_TOKEN, THINK_BEGIN_TOKEN, THINK_END_TOKEN, TOOL_CALLS_BEGIN_TOKEN,
    TOOL_CALLS_END_TOKEN, TOOL_CALL_BEGIN_TOKEN, TOOL_CALL_END_TOKEN, TOOL_RESPONSES_BEGIN_TOKEN,
    TOOL_RESPONSES_END_TOKEN, TOOL_RESPONSE_BEGIN_TOKEN, TOOL_RESPONSE_END_TOKEN, TOOL_SEPARATOR_TOKEN,
    USER_TOKEN,
};
pub use error::{ChatError, TokenizerError};
pub use gguf::{GgmlTokenType, SpecialTokenIds};

#[derive(Debug)]
pub struct Hy3Tokenizer {
    inner: Tokenizer,
    tokens: Vec<String>,
    token_types: Vec<GgmlTokenType>,
    special_ids: SpecialTokenIds,
    chat_template: String,
}

impl Hy3Tokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, TokenizerError> {
        let parsed = gguf::parse(file)?;
        Ok(Self {
            inner: parsed.tokenizer,
            tokens: parsed.tokens,
            token_types: parsed.token_types,
            special_ids: parsed.special_ids,
            chat_template: parsed.chat_template,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError::Encode(error.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    pub fn format_and_encode(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<Vec<u32>, TokenizerError> {
        let prompt = format_chat(messages, options)?;
        self.encode(&prompt)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_type(&self, id: u32) -> Option<GgmlTokenType> {
        self.token_types.get(id as usize).copied()
    }

    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }

    pub const fn special_ids(&self) -> SpecialTokenIds {
        self.special_ids
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Hy3Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        tokenizers::step_decode_stream(
            &self.tokenizer.inner,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| TokenizerError::Decode(error.to_string()))
    }
}
