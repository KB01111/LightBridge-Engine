use bridge_tokenizer::{
    parse_assistant_output, AssistantParseError, ReasoningEffort, ARGUMENT_KEY_BEGIN_TOKEN,
    ARGUMENT_KEY_END_TOKEN, ARGUMENT_VALUE_BEGIN_TOKEN, ARGUMENT_VALUE_END_TOKEN, THINK_END_TOKEN,
    TOOL_CALLS_BEGIN_TOKEN, TOOL_CALLS_END_TOKEN, TOOL_CALL_BEGIN_TOKEN, TOOL_CALL_END_TOKEN,
    TOOL_SEPARATOR_TOKEN,
};
use serde_json::json;

#[test]
fn parses_reasoning_content_and_multiple_typed_tool_arguments() {
    let raw = format!(
        "check source{THINK_END_TOKEN}Calling tools.{TOOL_CALLS_BEGIN_TOKEN}\n\
         {TOOL_CALL_BEGIN_TOKEN}search{TOOL_SEPARATOR_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}count{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}3{ARGUMENT_VALUE_END_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}query{ARGUMENT_KEY_END_TOKEN}\n\
         {ARGUMENT_VALUE_BEGIN_TOKEN}rust{ARGUMENT_VALUE_END_TOKEN}\n\
         {TOOL_CALL_END_TOKEN}\n\
         {TOOL_CALL_BEGIN_TOKEN}notify{TOOL_SEPARATOR_TOKEN}\n\
         {TOOL_CALL_END_TOKEN}\n\
         {TOOL_CALLS_END_TOKEN}"
    );
    let parsed = parse_assistant_output(&raw, ReasoningEffort::High).unwrap();
    assert_eq!(parsed.reasoning.as_deref(), Some("check source"));
    assert_eq!(parsed.content, "Calling tools.");
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].name, "search");
    assert_eq!(parsed.tool_calls[0].arguments["count"], json!(3));
    assert_eq!(parsed.tool_calls[0].arguments["query"], json!("rust"));
    assert_eq!(parsed.tool_calls[1].name, "notify");
    assert!(parsed.tool_calls[1].arguments.is_empty());
}

#[test]
fn preserves_plain_and_incomplete_reasoning_outputs_without_panicking() {
    let plain = parse_assistant_output("plain answer", ReasoningEffort::NoThink).unwrap();
    assert_eq!(plain.content, "plain answer");
    assert_eq!(plain.reasoning, None);

    let reasoning = parse_assistant_output("still thinking", ReasoningEffort::Low).unwrap();
    assert_eq!(reasoning.content, "");
    assert_eq!(reasoning.reasoning.as_deref(), Some("still thinking"));
}

#[test]
fn rejects_incomplete_and_duplicate_tool_protocol() {
    let incomplete = format!("content{TOOL_CALLS_BEGIN_TOKEN}{TOOL_CALL_BEGIN_TOKEN}search");
    assert!(matches!(
        parse_assistant_output(&incomplete, ReasoningEffort::NoThink),
        Err(AssistantParseError::MissingMarker { .. })
    ));

    let duplicate = format!(
        "{TOOL_CALLS_BEGIN_TOKEN}{TOOL_CALL_BEGIN_TOKEN}search{TOOL_SEPARATOR_TOKEN}\n\
         {ARGUMENT_KEY_BEGIN_TOKEN}q{ARGUMENT_KEY_END_TOKEN}\
         {ARGUMENT_VALUE_BEGIN_TOKEN}a{ARGUMENT_VALUE_END_TOKEN}\
         {ARGUMENT_KEY_BEGIN_TOKEN}q{ARGUMENT_KEY_END_TOKEN}\
         {ARGUMENT_VALUE_BEGIN_TOKEN}b{ARGUMENT_VALUE_END_TOKEN}\
         {TOOL_CALL_END_TOKEN}{TOOL_CALLS_END_TOKEN}"
    );
    assert!(matches!(
        parse_assistant_output(&duplicate, ReasoningEffort::NoThink),
        Err(AssistantParseError::DuplicateArgumentKey { .. })
    ));
}
