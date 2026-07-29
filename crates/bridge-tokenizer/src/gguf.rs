use std::collections::HashMap;

use bridge_gguf::{GgufArray, GgufFile, GgufValue, GgufValueType, MetadataError};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeBuilder, Vocab};
use tokenizers::pre_tokenizers::{
    byte_level::ByteLevel,
    sequence::Sequence,
    split::{Split, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::error::TokenizerError;

const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKEN_TYPES_KEY: &str = "tokenizer.ggml.token_type";
const MERGES_KEY: &str = "tokenizer.ggml.merges";
const MODEL_KEY: &str = "tokenizer.ggml.model";
const PRETOKENIZER_KEY: &str = "tokenizer.ggml.pre";
const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";
const BOS_ID_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_ID_KEY: &str = "tokenizer.ggml.eos_token_id";
const PAD_ID_KEY: &str = "tokenizer.ggml.padding_token_id";
const SEPARATOR_ID_KEY: &str = "tokenizer.ggml.separator_token_id";
const LEGACY_SEPARATOR_ID_KEY: &str = "tokenizer.ggml.seperator_token_id";

const HUNYUAN_NUMBER_PATTERN: &str = r"\p{N}{1,3}";
const HUNYUAN_CJK_PATTERN: &str = r"[一-龥぀-ゟ゠-ヿ]+";
const HUNYUAN_WORD_PATTERN: &str = concat!(
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"##,
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+",
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlTokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TryFrom<(u32, i32)> for GgmlTokenType {
    type Error = TokenizerError;

    fn try_from((id, value): (u32, i32)) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(TokenizerError::InvalidTokenType { id, value }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub separator: u32,
}

pub(crate) struct ParsedTokenizer {
    pub tokenizer: Tokenizer,
    pub tokens: Vec<String>,
    pub token_types: Vec<GgmlTokenType>,
    pub special_ids: SpecialTokenIds,
    pub chat_template: String,
}

pub(crate) fn parse(file: &GgufFile) -> Result<ParsedTokenizer, TokenizerError> {
    let model = file.get_string(MODEL_KEY)?;
    if model != "gpt2" {
        return Err(TokenizerError::UnsupportedModel(model.to_owned()));
    }
    let pretokenizer = file.get_string(PRETOKENIZER_KEY)?;
    if pretokenizer != "hunyuan-dense" {
        return Err(TokenizerError::UnsupportedPretokenizer(pretokenizer.to_owned()));
    }

    let tokens = string_array(file.get_array(TOKENS_KEY)?, TOKENS_KEY)?;
    if tokens.is_empty() {
        return Err(TokenizerError::EmptyVocabulary);
    }
    if tokens.len() > u32::MAX as usize {
        return Err(TokenizerError::VocabularyTooLarge);
    }
    let raw_types = i32_array(file.get_array(TOKEN_TYPES_KEY)?, TOKEN_TYPES_KEY)?;
    if raw_types.len() != tokens.len() {
        return Err(TokenizerError::TokenTypeCountMismatch {
            tokens: tokens.len(),
            types: raw_types.len(),
        });
    }
    let token_types = raw_types
        .into_iter()
        .enumerate()
        .map(|(id, value)| GgmlTokenType::try_from((id as u32, value)))
        .collect::<Result<Vec<_>, _>>()?;
    let raw_merges = string_array(file.get_array(MERGES_KEY)?, MERGES_KEY)?;
    let merges = raw_merges
        .into_iter()
        .enumerate()
        .map(|(index, merge)| parse_merge(index, merge))
        .collect::<Result<Vec<_>, _>>()?;

    let mut seen = HashMap::<&str, u32>::with_capacity(tokens.len());
    let mut vocab = Vocab::with_capacity(tokens.len());
    for (index, token) in tokens.iter().enumerate() {
        let id = index as u32;
        if let Some(first) = seen.insert(token.as_str(), id) {
            return Err(TokenizerError::DuplicateToken {
                token: token.clone(),
                first,
                second: id,
            });
        }
        vocab.insert(token.clone(), id);
    }

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|error| TokenizerError::Construction(error.to_string()))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(hunyuan_dense_pretokenizer()?));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    let control_tokens = tokens
        .iter()
        .zip(&token_types)
        .filter(|(_, token_type)| **token_type == GgmlTokenType::Control)
        .map(|(token, _)| AddedToken::from(token.clone(), true).normalized(false))
        .collect::<Vec<_>>();
    tokenizer
        .add_special_tokens(control_tokens)
        .map_err(|error| TokenizerError::Construction(error.to_string()))?;
    let user_defined_tokens = tokens
        .iter()
        .zip(&token_types)
        .filter(|(_, token_type)| **token_type == GgmlTokenType::UserDefined)
        .map(|(token, _)| AddedToken::from(token.clone(), false).normalized(false))
        .collect::<Vec<_>>();
    tokenizer
        .add_tokens(user_defined_tokens)
        .map_err(|error| TokenizerError::Construction(error.to_string()))?;

    let special_ids = SpecialTokenIds {
        bos: checked_id(file, BOS_ID_KEY, tokens.len())?,
        eos: checked_id(file, EOS_ID_KEY, tokens.len())?,
        pad: checked_id(file, PAD_ID_KEY, tokens.len())?,
        separator: checked_separator_id(file, tokens.len())?,
    };
    let chat_template = file.get_string(CHAT_TEMPLATE_KEY)?.to_owned();

    Ok(ParsedTokenizer {
        tokenizer,
        tokens,
        token_types,
        special_ids,
        chat_template,
    })
}

fn hunyuan_dense_pretokenizer() -> Result<Sequence, TokenizerError> {
    let split = |pattern: &'static str| {
        Split::new(
            SplitPattern::Regex(pattern.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map(PreTokenizerWrapper::from)
        .map_err(|error| TokenizerError::Construction(error.to_string()))
    };
    Ok(Sequence::new(vec![
        split(HUNYUAN_NUMBER_PATTERN)?,
        split(HUNYUAN_CJK_PATTERN)?,
        split(HUNYUAN_WORD_PATTERN)?,
        ByteLevel::new(false, true, false).into(),
    ]))
}

fn string_array(array: &GgufArray, key: &'static str) -> Result<Vec<String>, TokenizerError> {
    if array.element_type != GgufValueType::String {
        return Err(TokenizerError::WrongArrayElementType {
            key,
            expected: GgufValueType::String,
            actual: array.element_type,
        });
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::String(value) => Ok(value.clone()),
            _ => Err(TokenizerError::WrongArrayValueType { key, index }),
        })
        .collect()
}

fn i32_array(array: &GgufArray, key: &'static str) -> Result<Vec<i32>, TokenizerError> {
    if array.element_type != GgufValueType::I32 {
        return Err(TokenizerError::WrongArrayElementType {
            key,
            expected: GgufValueType::I32,
            actual: array.element_type,
        });
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::I32(value) => Ok(*value),
            _ => Err(TokenizerError::WrongArrayValueType { key, index }),
        })
        .collect()
}

fn parse_merge(index: usize, merge: String) -> Result<(String, String), TokenizerError> {
    let Some((left, right)) = merge.split_once(' ') else {
        return Err(TokenizerError::InvalidMerge { index, merge });
    };
    if left.is_empty() || right.is_empty() || right.contains(' ') {
        return Err(TokenizerError::InvalidMerge { index, merge });
    }
    Ok((left.to_owned(), right.to_owned()))
}

fn checked_id(file: &GgufFile, key: &'static str, vocabulary_size: usize) -> Result<u32, TokenizerError> {
    let id = file.get_u32(key)?;
    if id as usize >= vocabulary_size {
        return Err(TokenizerError::TokenIdOutOfRange {
            key,
            id,
            vocabulary_size,
        });
    }
    Ok(id)
}

fn checked_separator_id(file: &GgufFile, vocabulary_size: usize) -> Result<u32, TokenizerError> {
    match checked_id(file, SEPARATOR_ID_KEY, vocabulary_size) {
        Ok(id) => Ok(id),
        Err(TokenizerError::Metadata(MetadataError::Missing { .. })) => {
            checked_id(file, LEGACY_SEPARATOR_ID_KEY, vocabulary_size)
        }
        Err(error) => Err(error),
    }
}
