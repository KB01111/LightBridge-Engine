use std::collections::BTreeMap;
use std::fmt::Write;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ChatError;

pub const BOS_TOKEN: &str = "<｜hy_begin_of_sentence:opensource｜>";
pub const EOS_TOKEN: &str = "<｜hy_eos:opensource｜>";
pub const USER_TOKEN: &str = "<｜hy_User:opensource｜>";
pub const ASSISTANT_TOKEN: &str = "<｜hy_Assistant:opensource｜>";
pub const THINK_BEGIN_TOKEN: &str = "<think:opensource>";
pub const THINK_END_TOKEN: &str = "</think:opensource>";
pub const TOOL_CALLS_BEGIN_TOKEN: &str = "<tool_calls:opensource>";
pub const TOOL_CALLS_END_TOKEN: &str = "</tool_calls:opensource>";
pub const TOOL_CALL_BEGIN_TOKEN: &str = "<tool_call:opensource>";
pub const TOOL_CALL_END_TOKEN: &str = "</tool_call:opensource>";
pub const TOOL_SEPARATOR_TOKEN: &str = "<tool_sep:opensource>";
pub const ARGUMENT_KEY_BEGIN_TOKEN: &str = "<arg_key:opensource>";
pub const ARGUMENT_KEY_END_TOKEN: &str = "</arg_key:opensource>";
pub const ARGUMENT_VALUE_BEGIN_TOKEN: &str = "<arg_value:opensource>";
pub const ARGUMENT_VALUE_END_TOKEN: &str = "</arg_value:opensource>";
pub const TOOL_RESPONSES_BEGIN_TOKEN: &str = "<tool_responses:opensource>";
pub const TOOL_RESPONSES_END_TOKEN: &str = "</tool_responses:opensource>";
pub const TOOL_RESPONSE_BEGIN_TOKEN: &str = "<tool_response:opensource>";
pub const TOOL_RESPONSE_END_TOKEN: &str = "</tool_response:opensource>";
pub const REASONING_MODE_TOKEN: &str = "<｜reasoning_mode:opensource｜>";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    High,
    Low,
    #[default]
    NoThink,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Low => "low",
            Self::NoThink => "no_think",
        }
    }
}

impl TryFrom<&str> for ReasoningEffort {
    type Error = ChatError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "high" => Ok(Self::High),
            "low" => Ok(Self::Low),
            "no_think" => Ok(Self::NoThink),
            _ => Err(ChatError::InvalidReasoningEffort),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "role")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        #[serde(default, alias = "reasoning_content")]
        reasoning: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        content: String,
    },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self::Tool {
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub reasoning_effort: ReasoningEffort,
    pub preserved_thinking: Option<bool>,
    pub is_training: bool,
    pub raw_last_assistant: bool,
    pub tools: Vec<Value>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            reasoning_effort: ReasoningEffort::NoThink,
            preserved_thinking: None,
            is_training: false,
            raw_last_assistant: false,
            tools: Vec::new(),
        }
    }
}

pub fn format_chat(messages: &[ChatMessage], options: &ChatTemplateOptions) -> Result<String, ChatError> {
    for (tool_index, tool) in options.tools.iter().enumerate() {
        if !tool.is_object() {
            return Err(ChatError::InvalidToolDefinition { tool_index });
        }
    }

    let last_user_index = messages
        .iter()
        .rposition(|message| matches!(message, ChatMessage::User { .. }));
    let mut system_prompt = messages
        .iter()
        .filter_map(|message| match message {
            ChatMessage::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<String>();

    if options.tools.is_empty() {
        write!(
            system_prompt,
            "{REASONING_MODE_TOKEN}reasoning_effort:{}",
            options.reasoning_effort.as_str()
        )
        .expect("writing to String cannot fail");
    }

    let mut output = String::with_capacity(system_prompt.len() + 256);
    output.push_str(BOS_TOKEN);
    output.push_str(&system_prompt);
    if !options.tools.is_empty() {
        append_tool_instructions(&mut output, &system_prompt, options)?;
    }

    let preserve_thinking = options.preserved_thinking.unwrap_or(!options.tools.is_empty());
    let mut previous_was_tool = false;
    let mut is_first_tool_response = true;
    let mut last_is_assistant = false;

    for (index, message) in messages.iter().enumerate() {
        match message {
            ChatMessage::System { .. } => {}
            ChatMessage::User { content } => {
                close_tool_responses_if_needed(&mut output, &mut previous_was_tool);
                output.push_str(USER_TOKEN);
                output.push_str(content);
                last_is_assistant = false;
            }
            ChatMessage::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                close_tool_responses_if_needed(&mut output, &mut previous_was_tool);
                output.push_str(ASSISTANT_TOKEN);

                let include_reasoning = options.is_training
                    || preserve_thinking
                    || last_user_index.is_some_and(|last_user| index > last_user);
                let mut assistant_content = String::new();
                assistant_content.push_str(THINK_BEGIN_TOKEN);
                if include_reasoning {
                    if let Some(reasoning) = reasoning {
                        assistant_content.push_str(reasoning);
                    }
                }
                assistant_content.push_str(THINK_END_TOKEN);
                assistant_content.push_str(content);

                if tool_calls.is_empty() {
                    if index + 1 == messages.len() && options.raw_last_assistant {
                        output.push_str(content);
                    } else {
                        output.push_str(&assistant_content);
                        if index + 1 != messages.len() || options.is_training {
                            output.push_str(EOS_TOKEN);
                        }
                    }
                } else {
                    output.push_str(&assistant_content);
                    append_tool_calls(&mut output, tool_calls)?;
                    output.push_str(EOS_TOKEN);
                }

                previous_was_tool = false;
                last_is_assistant = index + 1 == messages.len();
            }
            ChatMessage::Tool { content } => {
                previous_was_tool = true;
                if is_first_tool_response {
                    output.push_str(TOOL_RESPONSES_BEGIN_TOKEN);
                    output.push('\n');
                    is_first_tool_response = false;
                }
                output.push_str(TOOL_RESPONSE_BEGIN_TOKEN);
                output.push('\n');
                output.push_str(content);
                output.push('\n');
                output.push_str(TOOL_RESPONSE_END_TOKEN);
                output.push('\n');
                last_is_assistant = false;
            }
        }
    }

    close_tool_responses_if_needed(&mut output, &mut previous_was_tool);
    if options.add_generation_prompt && !last_is_assistant {
        output.push_str(ASSISTANT_TOKEN);
        match options.reasoning_effort {
            ReasoningEffort::High | ReasoningEffort::Low => output.push_str(THINK_BEGIN_TOKEN),
            ReasoningEffort::NoThink => {
                output.push_str(THINK_BEGIN_TOKEN);
                output.push_str(THINK_END_TOKEN);
            }
        }
    }

    Ok(output)
}

fn append_tool_instructions(
    output: &mut String,
    system_prompt: &str,
    options: &ChatTemplateOptions,
) -> Result<(), ChatError> {
    if system_prompt.is_empty() {
        output.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.");
    } else {
        output.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.");
    }
    output.push_str("\n\nYou are provided with function signatures within <tools></tools> XML tags:");
    output.push_str("\n<tools>\n");
    for (tool_index, tool) in options.tools.iter().enumerate() {
        if tool_index > 0 {
            output.push('\n');
        }
        let encoded =
            serde_json::to_string(tool).map_err(|error| ChatError::ToolDefinitionSerialization {
                tool_index,
                message: error.to_string(),
            })?;
        output.push_str(&encoded);
    }
    output.push_str("\n</tools>\n\n");
    write!(
        output,
        "For function call returns, you should first print {TOOL_CALLS_BEGIN_TOKEN}\n\
         For each function call, you should return object like:\n\
         {TOOL_CALL_BEGIN_TOKEN}{{function-name}}{TOOL_SEPARATOR_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}{{arg-key-1}}{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}{{arg-value-1}}{ARGUMENT_VALUE_END_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}{{arg-key-2}}{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}{{arg-value-2}}{ARGUMENT_VALUE_END_TOKEN}\n\
         ...\n\
         {TOOL_CALL_END_TOKEN}\n\
         At the end of function call returns, you should print \
         {TOOL_CALLS_END_TOKEN}{REASONING_MODE_TOKEN}reasoning_effort:{}",
        options.reasoning_effort.as_str()
    )
    .expect("writing to String cannot fail");
    Ok(())
}

fn append_tool_calls(output: &mut String, tool_calls: &[ToolCall]) -> Result<(), ChatError> {
    output.push_str(TOOL_CALLS_BEGIN_TOKEN);
    output.push('\n');
    for (call_index, call) in tool_calls.iter().enumerate() {
        if call.name.is_empty() {
            return Err(ChatError::EmptyToolName { call_index });
        }
        output.push_str(TOOL_CALL_BEGIN_TOKEN);
        output.push_str(&call.name);
        output.push_str(TOOL_SEPARATOR_TOKEN);
        output.push('\n');
        for (key, value) in &call.arguments {
            output.push_str(ARGUMENT_KEY_BEGIN_TOKEN);
            output.push_str(key);
            output.push_str(ARGUMENT_KEY_END_TOKEN);
            output.push('\n');
            output.push_str(ARGUMENT_VALUE_BEGIN_TOKEN);
            match value {
                Value::String(value) => output.push_str(value),
                _ => {
                    let encoded = serde_json::to_string(value).map_err(|error| {
                        ChatError::ToolArgumentSerialization {
                            key: key.clone(),
                            message: error.to_string(),
                        }
                    })?;
                    output.push_str(&encoded);
                }
            }
            output.push_str(ARGUMENT_VALUE_END_TOKEN);
            output.push('\n');
        }
        output.push_str(TOOL_CALL_END_TOKEN);
        output.push('\n');
    }
    output.push_str(TOOL_CALLS_END_TOKEN);
    Ok(())
}

fn close_tool_responses_if_needed(output: &mut String, previous_was_tool: &mut bool) {
    if *previous_was_tool {
        output.push_str(TOOL_RESPONSES_END_TOKEN);
        *previous_was_tool = false;
    }
}
