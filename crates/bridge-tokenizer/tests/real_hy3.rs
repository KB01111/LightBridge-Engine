use std::path::Path;

use bridge_gguf::open;
use bridge_tokenizer::{
    ChatMessage, ChatTemplateOptions, Hy3Tokenizer, ReasoningEffort, ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN,
    USER_TOKEN,
};

#[test]
fn selected_hy3_header_matches_official_tokenizer_vectors() {
    let Some(path) = std::env::var_os("BRIDGE_HY3_HEADER") else {
        eprintln!("skipped: BRIDGE_HY3_HEADER is not set");
        return;
    };
    let parsed = open(Path::new(&path)).unwrap();
    let tokenizer = Hy3Tokenizer::from_gguf(&parsed).unwrap();

    let vectors: &[(&str, &[u32])] = &[
        ("Hello, world!", &[16_883, 11, 2_385, 0]),
        ("1234567", &[7_827, 18_695, 22]),
        ("你好，世界", &[17_687, 270, 3_042]),
        ("日本語テスト", &[99_706, 23_480, 39_119]),
        (" leading  spaces\nnext", &[6_255, 206, 10_004, 185, 7_532]),
        (
            "emoji: 🤖🚀",
            &[20_734, 12_130, 25, 28_959, 97, 230, 14_341, 234, 208],
        ),
        ("Café naïve", &[34, 3_339, 1_318, 106_098]),
    ];
    for (text, expected) in vectors {
        assert_eq!(tokenizer.encode(text).unwrap(), *expected, "{text:?}");
        assert_eq!(tokenizer.decode(expected, false).unwrap(), *text, "{text:?}");
    }

    assert_eq!(tokenizer.vocabulary_size(), 120_832);
    assert_eq!(tokenizer.special_ids().bos, 120_000);
    assert_eq!(tokenizer.special_ids().eos, 120_025);
    assert_eq!(tokenizer.special_ids().pad, 120_002);
    assert_eq!(tokenizer.special_ids().separator, 120_007);
    assert_eq!(tokenizer.token_to_id(BOS_TOKEN), Some(120_000));
    assert_eq!(tokenizer.token_to_id(EOS_TOKEN), Some(120_025));
    assert_eq!(tokenizer.token_to_id(USER_TOKEN), Some(120_006));
    assert_eq!(tokenizer.token_to_id(ASSISTANT_TOKEN), Some(120_007));
}

#[test]
fn selected_hy3_chat_prompt_tokenizes_special_markers_atomically() {
    let Some(path) = std::env::var_os("BRIDGE_HY3_HEADER") else {
        eprintln!("skipped: BRIDGE_HY3_HEADER is not set");
        return;
    };
    let parsed = open(Path::new(&path)).unwrap();
    let tokenizer = Hy3Tokenizer::from_gguf(&parsed).unwrap();
    let messages = [ChatMessage::system("Be concise."), ChatMessage::user("Hello")];
    let options = ChatTemplateOptions {
        reasoning_effort: ReasoningEffort::NoThink,
        ..ChatTemplateOptions::default()
    };
    let ids = tokenizer.format_and_encode(&messages, &options).unwrap();
    assert_eq!(
        ids,
        [
            120_000, 5_231, 36_461, 13, 120_044, 70_830, 277, 9_002, 647, 497, 25, 5_130, 25_152, 1_326,
            120_006, 16_883, 120_007, 120_029, 120_030,
        ]
    );
}
