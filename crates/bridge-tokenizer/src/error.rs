use bridge_gguf::{GgufValueType, MetadataError};

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("unsupported tokenizer model {0:?}; expected \"gpt2\"")]
    UnsupportedModel(String),
    #[error("unsupported tokenizer pre-tokenizer {0:?}; expected \"hunyuan-dense\"")]
    UnsupportedPretokenizer(String),
    #[error("tokenizer metadata array {key:?} has element type {actual:?}, expected {expected:?}")]
    WrongArrayElementType {
        key: &'static str,
        expected: GgufValueType,
        actual: GgufValueType,
    },
    #[error("tokenizer metadata array {key:?} contains a value of the wrong type at index {index}")]
    WrongArrayValueType { key: &'static str, index: usize },
    #[error("tokenizer vocabulary is empty")]
    EmptyVocabulary,
    #[error("tokenizer vocabulary is too large to address with u32 token IDs")]
    VocabularyTooLarge,
    #[error("duplicate tokenizer token {token:?} at IDs {first} and {second}")]
    DuplicateToken { token: String, first: u32, second: u32 },
    #[error("token-type count {types} does not match vocabulary count {tokens}")]
    TokenTypeCountMismatch { tokens: usize, types: usize },
    #[error("invalid GGML token type {value} at token ID {id}")]
    InvalidTokenType { id: u32, value: i32 },
    #[error("invalid tokenizer merge at index {index}: {merge:?}")]
    InvalidMerge { index: usize, merge: String },
    #[error("token ID metadata {key:?} is outside the vocabulary: {id} >= {vocabulary_size}")]
    TokenIdOutOfRange {
        key: &'static str,
        id: u32,
        vocabulary_size: usize,
    },
    #[error("tokenizer construction failed: {0}")]
    Construction(String),
    #[error("tokenization failed: {0}")]
    Encode(String),
    #[error("token decoding failed: {0}")]
    Decode(String),
    #[error("chat formatting failed: {0}")]
    Chat(#[from] ChatError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChatError {
    #[error("reasoning effort must be one of no_think, low, or high")]
    InvalidReasoningEffort,
    #[error("tool call {call_index} has an empty function name")]
    EmptyToolName { call_index: usize },
    #[error("tool definition {tool_index} is not a JSON object")]
    InvalidToolDefinition { tool_index: usize },
    #[error("tool-call argument {key:?} could not be serialized: {message}")]
    ToolArgumentSerialization { key: String, message: String },
    #[error("tool definition {tool_index} could not be serialized: {message}")]
    ToolDefinitionSerialization { tool_index: usize, message: String },
}
