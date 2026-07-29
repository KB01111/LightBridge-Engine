use bridge_gguf::{Endianness, GgufArray, GgufFile, GgufValue, GgufValueType};
use bridge_tokenizer::{GgmlTokenType, Hy3Tokenizer, BOS_TOKEN, EOS_TOKEN, USER_TOKEN};

fn string_array(values: &[&str]) -> GgufValue {
    GgufValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: values
            .iter()
            .map(|value| GgufValue::String((*value).into()))
            .collect(),
    })
}

fn token_type_array(values: &[i32]) -> GgufValue {
    GgufValue::Array(GgufArray {
        element_type: GgufValueType::I32,
        values: values.iter().copied().map(GgufValue::I32).collect(),
    })
}

fn fixture() -> GgufFile {
    let tokens = [
        "a",
        BOS_TOKEN,
        EOS_TOKEN,
        "<｜hy_pad:opensource｜>",
        USER_TOKEN,
        "<custom>",
    ];
    GgufFile {
        version: 3,
        endianness: Endianness::Little,
        metadata: vec![
            ("tokenizer.ggml.model".into(), GgufValue::String("gpt2".into())),
            (
                "tokenizer.ggml.pre".into(),
                GgufValue::String("hunyuan-dense".into()),
            ),
            ("tokenizer.ggml.tokens".into(), string_array(&tokens)),
            (
                "tokenizer.ggml.token_type".into(),
                token_type_array(&[1, 3, 3, 3, 3, 4]),
            ),
            ("tokenizer.ggml.merges".into(), string_array(&[])),
            ("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(1)),
            ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(2)),
            ("tokenizer.ggml.padding_token_id".into(), GgufValue::U32(3)),
            ("tokenizer.ggml.separator_token_id".into(), GgufValue::U32(4)),
            (
                "tokenizer.chat_template".into(),
                GgufValue::String("fixture".into()),
            ),
        ],
        tensors: Vec::new(),
        alignment: 32,
        data_offset: 0,
        file_len: 0,
    }
}

#[test]
fn builds_from_checked_gguf_metadata() {
    let tokenizer = Hy3Tokenizer::from_gguf(&fixture()).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 6);
    assert_eq!(tokenizer.special_ids().bos, 1);
    assert_eq!(tokenizer.special_ids().eos, 2);
    assert_eq!(tokenizer.special_ids().pad, 3);
    assert_eq!(tokenizer.special_ids().separator, 4);
    assert_eq!(tokenizer.id_to_token(5), Some("<custom>"));
    assert_eq!(tokenizer.token_type(5), Some(GgmlTokenType::UserDefined));
    assert_eq!(tokenizer.chat_template(), "fixture");
}

#[test]
fn encodes_control_and_user_defined_tokens_atomically() {
    let tokenizer = Hy3Tokenizer::from_gguf(&fixture()).unwrap();
    assert_eq!(tokenizer.encode("a").unwrap(), [0]);
    assert_eq!(tokenizer.encode(BOS_TOKEN).unwrap(), [1]);
    assert_eq!(tokenizer.encode("<custom>").unwrap(), [5]);
    assert_eq!(tokenizer.decode(&[0], false).unwrap(), "a");
    assert_eq!(tokenizer.decode(&[1], true).unwrap(), "");
}

#[test]
fn incremental_decoder_emits_stable_chunks() {
    let tokenizer = Hy3Tokenizer::from_gguf(&fixture()).unwrap();
    let mut decoder = tokenizer.incremental_decoder(true);
    assert_eq!(decoder.push(1).unwrap(), None);
    assert_eq!(decoder.push(0).unwrap(), Some("a".into()));
}

#[test]
fn rejects_duplicate_tokens_before_building_bpe() {
    let mut file = fixture();
    file.metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.tokens")
        .unwrap()
        .1 = string_array(&["a", "a", EOS_TOKEN, "<pad>", USER_TOKEN, "<custom>"]);
    let error = Hy3Tokenizer::from_gguf(&file).unwrap_err();
    assert!(error.to_string().contains("duplicate tokenizer token"));
}

#[test]
fn rejects_out_of_range_special_ids() {
    let mut file = fixture();
    file.metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.bos_token_id")
        .unwrap()
        .1 = GgufValue::U32(6);
    let error = Hy3Tokenizer::from_gguf(&file).unwrap_err();
    assert!(error.to_string().contains("outside the vocabulary"));
}

#[test]
fn accepts_the_legacy_separator_key_spelling_used_by_the_selected_checkpoint() {
    let mut file = fixture();
    let entry = file
        .metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.separator_token_id")
        .unwrap();
    entry.0 = "tokenizer.ggml.seperator_token_id".into();

    let tokenizer = Hy3Tokenizer::from_gguf(&file).unwrap();
    assert_eq!(tokenizer.special_ids().separator, 4);
}
