use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{
    ReasoningEffort, ToolCall, ARGUMENT_KEY_BEGIN_TOKEN, ARGUMENT_KEY_END_TOKEN, ARGUMENT_VALUE_BEGIN_TOKEN,
    ARGUMENT_VALUE_END_TOKEN, THINK_BEGIN_TOKEN, THINK_END_TOKEN, TOOL_CALLS_BEGIN_TOKEN,
    TOOL_CALLS_END_TOKEN, TOOL_CALL_BEGIN_TOKEN, TOOL_CALL_END_TOKEN, TOOL_SEPARATOR_TOKEN,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantOutput {
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl AssistantOutput {
    pub fn plain(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }
}

pub fn parse_assistant_output(
    raw: &str,
    reasoning_effort: ReasoningEffort,
) -> Result<AssistantOutput, AssistantParseError> {
    let (reasoning, body) = split_reasoning(raw, reasoning_effort);
    let Some((content, calls_and_tail)) = body.split_once(TOOL_CALLS_BEGIN_TOKEN) else {
        return Ok(AssistantOutput {
            content: body.to_owned(),
            reasoning,
            tool_calls: Vec::new(),
        });
    };
    let Some((calls, tail)) = calls_and_tail.split_once(TOOL_CALLS_END_TOKEN) else {
        return Err(AssistantParseError::MissingMarker {
            marker: TOOL_CALLS_END_TOKEN,
        });
    };
    if !tail.trim_matches(['\r', '\n']).is_empty() {
        return Err(AssistantParseError::TrailingToolText);
    }

    Ok(AssistantOutput {
        content: content.to_owned(),
        reasoning,
        tool_calls: parse_tool_calls(calls)?,
    })
}

fn split_reasoning(raw: &str, effort: ReasoningEffort) -> (Option<String>, &str) {
    if effort != ReasoningEffort::NoThink {
        let raw = raw.strip_prefix(THINK_BEGIN_TOKEN).unwrap_or(raw);
        return match raw.split_once(THINK_END_TOKEN) {
            Some((reasoning, body)) => (Some(reasoning.to_owned()), body),
            None => (Some(raw.to_owned()), ""),
        };
    }
    if let Some(raw) = raw.strip_prefix(THINK_BEGIN_TOKEN) {
        if let Some((reasoning, body)) = raw.split_once(THINK_END_TOKEN) {
            return (Some(reasoning.to_owned()), body);
        }
    }
    (None, raw)
}

fn parse_tool_calls(mut input: &str) -> Result<Vec<ToolCall>, AssistantParseError> {
    let mut calls = Vec::new();
    input = trim_newlines(input);
    while !input.is_empty() {
        input = input
            .strip_prefix(TOOL_CALL_BEGIN_TOKEN)
            .ok_or(AssistantParseError::MissingMarker {
                marker: TOOL_CALL_BEGIN_TOKEN,
            })?;
        let (name, remainder) =
            input
                .split_once(TOOL_SEPARATOR_TOKEN)
                .ok_or(AssistantParseError::MissingMarker {
                    marker: TOOL_SEPARATOR_TOKEN,
                })?;
        if name.is_empty() {
            return Err(AssistantParseError::EmptyToolName {
                call_index: calls.len(),
            });
        }
        let (arguments, remainder) =
            remainder
                .split_once(TOOL_CALL_END_TOKEN)
                .ok_or(AssistantParseError::MissingMarker {
                    marker: TOOL_CALL_END_TOKEN,
                })?;
        calls.push(ToolCall {
            name: name.to_owned(),
            arguments: parse_arguments(arguments, calls.len())?,
        });
        input = trim_newlines(remainder);
    }
    Ok(calls)
}

fn parse_arguments(
    mut input: &str,
    call_index: usize,
) -> Result<BTreeMap<String, Value>, AssistantParseError> {
    let mut arguments = BTreeMap::new();
    input = trim_newlines(input);
    while !input.is_empty() {
        input = input
            .strip_prefix(ARGUMENT_KEY_BEGIN_TOKEN)
            .ok_or(AssistantParseError::MissingMarker {
                marker: ARGUMENT_KEY_BEGIN_TOKEN,
            })?;
        let (key, remainder) =
            input
                .split_once(ARGUMENT_KEY_END_TOKEN)
                .ok_or(AssistantParseError::MissingMarker {
                    marker: ARGUMENT_KEY_END_TOKEN,
                })?;
        if key.is_empty() {
            return Err(AssistantParseError::EmptyArgumentKey { call_index });
        }
        input = trim_newlines(remainder);
        input = input
            .strip_prefix(ARGUMENT_VALUE_BEGIN_TOKEN)
            .ok_or(AssistantParseError::MissingMarker {
                marker: ARGUMENT_VALUE_BEGIN_TOKEN,
            })?;
        let (value, remainder) =
            input
                .split_once(ARGUMENT_VALUE_END_TOKEN)
                .ok_or(AssistantParseError::MissingMarker {
                    marker: ARGUMENT_VALUE_END_TOKEN,
                })?;
        let value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()));
        if arguments.insert(key.to_owned(), value).is_some() {
            return Err(AssistantParseError::DuplicateArgumentKey {
                call_index,
                key: key.to_owned(),
            });
        }
        input = trim_newlines(remainder);
    }
    Ok(arguments)
}

fn trim_newlines(input: &str) -> &str {
    input.trim_start_matches(['\r', '\n'])
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AssistantParseError {
    #[error("generated assistant output is missing protocol marker {marker:?}")]
    MissingMarker { marker: &'static str },
    #[error("generated tool call {call_index} has an empty function name")]
    EmptyToolName { call_index: usize },
    #[error("generated tool call {call_index} has an empty argument key")]
    EmptyArgumentKey { call_index: usize },
    #[error("generated tool call {call_index} repeats argument key {key:?}")]
    DuplicateArgumentKey { call_index: usize, key: String },
    #[error("generated assistant output has text after the tool-call envelope")]
    TrailingToolText,
}
