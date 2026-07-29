use std::collections::BTreeMap;

use bridge_tokenizer::{
    format_chat, ChatMessage, ChatTemplateOptions, ReasoningEffort, ToolCall, ARGUMENT_KEY_BEGIN_TOKEN,
    ARGUMENT_KEY_END_TOKEN, ARGUMENT_VALUE_BEGIN_TOKEN, ARGUMENT_VALUE_END_TOKEN, ASSISTANT_TOKEN, BOS_TOKEN,
    EOS_TOKEN, REASONING_MODE_TOKEN, THINK_BEGIN_TOKEN, THINK_END_TOKEN, TOOL_CALLS_BEGIN_TOKEN,
    TOOL_CALLS_END_TOKEN, TOOL_CALL_BEGIN_TOKEN, TOOL_CALL_END_TOKEN, TOOL_RESPONSES_BEGIN_TOKEN,
    TOOL_RESPONSES_END_TOKEN, TOOL_RESPONSE_BEGIN_TOKEN, TOOL_RESPONSE_END_TOKEN, TOOL_SEPARATOR_TOKEN,
    USER_TOKEN,
};
use serde_json::json;

#[test]
fn formats_no_think_generation_prompt_exactly() {
    let messages = [ChatMessage::system("Be concise."), ChatMessage::user("Hello")];
    let actual = format_chat(&messages, &ChatTemplateOptions::default()).unwrap();
    let expected = format!(
        "{BOS_TOKEN}Be concise.{REASONING_MODE_TOKEN}reasoning_effort:no_think\
         {USER_TOKEN}Hello{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}{THINK_END_TOKEN}"
    );
    assert_eq!(actual, expected);
}

#[test]
fn high_effort_generation_prompt_opens_thinking() {
    let options = ChatTemplateOptions {
        reasoning_effort: ReasoningEffort::High,
        ..ChatTemplateOptions::default()
    };
    let actual = format_chat(&[ChatMessage::user("Solve it")], &options).unwrap();
    assert!(actual.ends_with(&format!("{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}")));
    assert!(!actual.ends_with(THINK_END_TOKEN));
}

#[test]
fn old_reasoning_is_removed_without_tools() {
    let messages = [
        ChatMessage::user("first"),
        ChatMessage::Assistant {
            content: "answer".into(),
            reasoning: Some("private".into()),
            tool_calls: Vec::new(),
        },
        ChatMessage::user("next"),
    ];
    let actual = format_chat(&messages, &ChatTemplateOptions::default()).unwrap();
    assert!(!actual.contains("private"));
    assert!(actual.contains(&format!(
        "{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}{THINK_END_TOKEN}answer{EOS_TOKEN}"
    )));
}

#[test]
fn preserved_reasoning_and_tool_protocol_match_the_official_template() {
    let mut arguments = BTreeMap::new();
    arguments.insert("count".into(), json!(3));
    arguments.insert("query".into(), json!("rust"));
    let messages = [
        ChatMessage::system("Use evidence."),
        ChatMessage::user("Search"),
        ChatMessage::Assistant {
            content: "Calling.".into(),
            reasoning: Some("Need a source.".into()),
            tool_calls: vec![ToolCall {
                name: "search".into(),
                arguments,
            }],
        },
        ChatMessage::tool(r#"{"result":"ok"}"#),
        ChatMessage::user("Summarize"),
    ];
    let options = ChatTemplateOptions {
        reasoning_effort: ReasoningEffort::Low,
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object"}
            }
        })],
        ..ChatTemplateOptions::default()
    };

    let actual = format_chat(&messages, &options).unwrap();
    assert!(actual.starts_with(&format!("{BOS_TOKEN}Use evidence.\n\n# Tools")));
    assert!(actual.contains(&format!(
        "{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}Need a source.{THINK_END_TOKEN}Calling.\
         {TOOL_CALLS_BEGIN_TOKEN}\n\
         {TOOL_CALL_BEGIN_TOKEN}search{TOOL_SEPARATOR_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}count{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}3{ARGUMENT_VALUE_END_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}query{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}rust{ARGUMENT_VALUE_END_TOKEN}\n\
         {TOOL_CALL_END_TOKEN}\n{TOOL_CALLS_END_TOKEN}{EOS_TOKEN}"
    )));
    assert!(actual.contains(&format!(
        "{TOOL_RESPONSES_BEGIN_TOKEN}\n{TOOL_RESPONSE_BEGIN_TOKEN}\n\
         {{\"result\":\"ok\"}}\n{TOOL_RESPONSE_END_TOKEN}\n\
         {TOOL_RESPONSES_END_TOKEN}{USER_TOKEN}Summarize"
    )));
    assert!(actual.ends_with(&format!("{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}")));
}

#[test]
fn last_assistant_is_left_open_for_continuation_without_eos() {
    let messages = [ChatMessage::user("Question"), ChatMessage::assistant("Partial")];
    let actual = format_chat(&messages, &ChatTemplateOptions::default()).unwrap();
    assert!(actual.ends_with(&format!(
        "{ASSISTANT_TOKEN}{THINK_BEGIN_TOKEN}{THINK_END_TOKEN}Partial"
    )));
    assert!(!actual.ends_with(EOS_TOKEN));
}
