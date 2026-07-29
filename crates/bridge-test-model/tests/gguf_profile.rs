use bridge_model_hy3::{validate_selected_model, Hy3TensorRole};
use bridge_test_model::{ReducedHy3Model, BLOCK_COUNT};
use sha2::{Digest, Sha256};

#[test]
fn native_gguf_parser_and_explicit_profile_authorize_the_exact_fixture() {
    let model = ReducedHy3Model::new().unwrap();
    let validated = model.parse_and_validate_gguf().unwrap();

    assert_eq!(validated.config(), model.config());
    assert_eq!(validated.tensors().len(), 30);
    assert!(!validated.has_mtp());
    assert!(validated
        .tensor_for_role(Hy3TensorRole::AttentionQ {
            layer: (BLOCK_COUNT - 1) as u32,
        })
        .is_some());
    assert!(validated
        .tensor_for_role(Hy3TensorRole::RoutedDown { layer: 1 })
        .is_some());
}

#[test]
fn selected_checkpoint_wrapper_cannot_be_weakened_by_the_reduced_profile() {
    let model = ReducedHy3Model::new().unwrap();
    let bytes = model.gguf_bytes().unwrap();
    let parsed = bridge_gguf::GgufReader::new(std::io::Cursor::new(bytes))
        .read()
        .unwrap();
    let set = bridge_gguf_split::testing::from_file(parsed).unwrap();

    assert!(validate_selected_model(&set).is_err());
}

#[test]
fn serialized_reduced_gguf_is_byte_deterministic() {
    let first = ReducedHy3Model::new().unwrap().gguf_bytes().unwrap();
    let second = ReducedHy3Model::new().unwrap().gguf_bytes().unwrap();
    assert_eq!(first, second);
    assert_eq!(
        format!("{:x}", Sha256::digest(&first)),
        "48ca2ae274edd72496b836055d021d0bf312884d58850e168591c863a99dfe05"
    );
}
