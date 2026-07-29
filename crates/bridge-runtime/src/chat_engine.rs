use std::ops::ControlFlow;
use std::path::Path;

use bridge_gguf_split::open_set;
use bridge_tokenizer::{
    parse_assistant_output, AssistantOutput, ChatMessage, ChatTemplateOptions, Hy3Tokenizer,
};
use sha2::{Digest, Sha256};

use crate::{
    CancellationToken, CausalModel, GenerationError, GenerationOutcome, GenerationSession, Generator,
    Hy3ScalarError, Hy3ScalarModel, Hy3ScalarOptions, Hy3ScalarSession, SamplingConfig,
};

const CHAT_SESSION_MAGIC: &[u8; 8] = b"LBGS0001";
const CHAT_SESSION_VERSION: u32 = 1;
const CHAT_SESSION_DIGEST_BYTES: usize = 32;

#[derive(Debug)]
pub struct Hy3ChatEngine {
    generator: Generator<Hy3ScalarModel>,
    tokenizer: Hy3Tokenizer,
}

#[derive(Debug)]
pub struct Hy3ChatSession {
    generation: GenerationSession<Hy3ScalarSession>,
}

impl Hy3ChatSession {
    pub fn history(&self) -> &[u32] {
        self.generation.history()
    }

    pub fn has_logits(&self) -> bool {
        self.generation.has_logits()
    }
}

impl Hy3ChatEngine {
    pub fn open_selected(
        model_path: impl AsRef<Path>,
        options: Hy3ScalarOptions,
    ) -> Result<Self, ChatEngineError> {
        let model_path = model_path.as_ref();
        let set = open_set(model_path)?;
        let metadata = set
            .files()
            .first()
            .ok_or(ChatEngineError::MissingMetadataShard)?
            .parsed();
        let tokenizer = Hy3Tokenizer::from_gguf(metadata)?;
        let model = Hy3ScalarModel::open_selected(model_path, options)?;
        Self::from_parts(model, tokenizer)
    }

    /// Builds a chat engine from an already-authorized model and tokenizer.
    pub fn from_parts(model: Hy3ScalarModel, tokenizer: Hy3Tokenizer) -> Result<Self, ChatEngineError> {
        if tokenizer.vocabulary_size() != model.config().vocabulary_size as usize {
            return Err(ChatEngineError::VocabularyMismatch {
                tokenizer: tokenizer.vocabulary_size(),
                model: model.config().vocabulary_size as usize,
            });
        }
        Ok(Self {
            generator: Generator::new(model)?,
            tokenizer,
        })
    }

    pub fn model(&self) -> &Hy3ScalarModel {
        self.generator.model()
    }

    pub const fn tokenizer(&self) -> &Hy3Tokenizer {
        &self.tokenizer
    }

    pub fn new_session(&self) -> Result<Hy3ChatSession, ChatEngineError> {
        Ok(Hy3ChatSession {
            generation: self.generator.new_session()?,
        })
    }

    pub fn session_position(&self, session: &Hy3ChatSession) -> usize {
        self.generator.position(&session.generation)
    }

    pub fn complete<F>(
        &self,
        messages: &[ChatMessage],
        template: &ChatTemplateOptions,
        sampling: SamplingConfig,
        cancellation: &CancellationToken,
        emit: F,
    ) -> Result<ChatCompletion, ChatEngineError>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.complete_with_stops(messages, template, sampling, &[], cancellation, emit)
    }

    pub fn complete_with_stops<F>(
        &self,
        messages: &[ChatMessage],
        template: &ChatTemplateOptions,
        sampling: SamplingConfig,
        stop_sequences: &[String],
        cancellation: &CancellationToken,
        emit: F,
    ) -> Result<ChatCompletion, ChatEngineError>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        let mut session = self.new_session()?;
        self.complete_with_stops_in_session(
            &mut session,
            messages,
            template,
            sampling,
            stop_sequences,
            cancellation,
            emit,
        )
    }

    pub fn complete_in_session<F>(
        &self,
        session: &mut Hy3ChatSession,
        messages: &[ChatMessage],
        template: &ChatTemplateOptions,
        sampling: SamplingConfig,
        cancellation: &CancellationToken,
        emit: F,
    ) -> Result<ChatCompletion, ChatEngineError>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.complete_with_stops_in_session(session, messages, template, sampling, &[], cancellation, emit)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn complete_with_stops_in_session<F>(
        &self,
        session: &mut Hy3ChatSession,
        messages: &[ChatMessage],
        template: &ChatTemplateOptions,
        mut sampling: SamplingConfig,
        stop_sequences: &[String],
        cancellation: &CancellationToken,
        mut emit: F,
    ) -> Result<ChatCompletion, ChatEngineError>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        if let Some(index) = stop_sequences.iter().position(String::is_empty) {
            return Err(ChatEngineError::EmptyStopSequence { index });
        }
        let prompt_tokens = self.tokenizer.format_and_encode(messages, template)?;
        if !prompt_tokens.starts_with(session.generation.history()) {
            self.generator.reset(&mut session.generation);
        }
        let cached_prompt_tokens = session.generation.history().len();
        let prompt_suffix = &prompt_tokens[cached_prompt_tokens..];
        sampling.stop_tokens.insert(self.tokenizer.special_ids().eos);
        let mut decoder = self.tokenizer.incremental_decoder(true);
        let mut stop_filter = TextStopFilter::new(stop_sequences);
        let mut incremental_text = String::new();
        let mut visible_text = String::new();
        let mut decode_error = None;
        let mut callback_stopped = false;
        let mut matched_stop = false;
        let outcome = self.generator.generate_stream(
            &mut session.generation,
            prompt_suffix,
            sampling,
            cancellation,
            |token| {
                let chunk = match decoder.push(token.token_id) {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        decode_error = Some(error);
                        return ControlFlow::Break(());
                    }
                };
                if let Some(chunk) = chunk {
                    incremental_text.push_str(&chunk);
                    let update = stop_filter.push(&chunk);
                    visible_text.push_str(&update.text);
                    if !update.text.is_empty() && emit(&update.text).is_break() {
                        callback_stopped = true;
                        return ControlFlow::Break(());
                    }
                    if update.stopped {
                        matched_stop = true;
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            },
        )?;
        if let Some(error) = decode_error {
            return Err(ChatEngineError::Tokenizer(error));
        }

        let decoded_text = self.tokenizer.decode(&outcome.token_ids, true)?;
        let mut outcome = outcome;
        outcome.stats.prompt_tokens = prompt_tokens.len();
        if !callback_stopped && !matched_stop {
            let suffix = decoded_text
                .strip_prefix(&incremental_text)
                .ok_or(ChatEngineError::IncrementalDecodeDiverged)?;
            let update = stop_filter.push(suffix);
            visible_text.push_str(&update.text);
            if !update.text.is_empty() && emit(&update.text).is_break() {
                callback_stopped = true;
                outcome.stop_reason = crate::StopReason::Callback;
            }
            if update.stopped {
                matched_stop = true;
            } else if !callback_stopped {
                let suffix = stop_filter.finish();
                visible_text.push_str(&suffix);
                if !suffix.is_empty() && emit(&suffix).is_break() {
                    outcome.stop_reason = crate::StopReason::Callback;
                }
            }
        }
        if matched_stop {
            outcome.stop_reason = crate::StopReason::StopSequence;
        }
        let raw_decoded = self.tokenizer.decode(&outcome.token_ids, false)?;
        let raw_text = truncate_at_first_stop(&raw_decoded, stop_sequences).to_owned();
        let (assistant, structured_output_error) =
            match parse_assistant_output(&raw_text, template.reasoning_effort) {
                Ok(assistant) => (assistant, None),
                Err(error) => (AssistantOutput::plain(visible_text), Some(error.to_string())),
            };
        Ok(ChatCompletion {
            prompt_token_ids: prompt_tokens,
            cached_prompt_tokens,
            text: assistant.content.clone(),
            raw_text,
            assistant,
            structured_output_error,
            generation: outcome,
        })
    }

    pub fn export_session(
        &self,
        session: &Hy3ChatSession,
        maximum_bytes: usize,
    ) -> Result<Vec<u8>, ChatEngineError> {
        if maximum_bytes == 0 {
            return Err(ChatEngineError::ZeroSessionLimit);
        }
        if !session.generation.is_healthy() {
            return Err(ChatEngineError::UnhealthySessionSnapshot);
        }
        let history = session.generation.history();
        let logits = session.generation.logits();
        let position = self.generator.position(&session.generation);
        if history.len() != position {
            return Err(ChatEngineError::SessionPositionMismatch {
                history: history.len(),
                position,
            });
        }
        if logits.len() != self.model().config().vocabulary_size as usize {
            return Err(ChatEngineError::SessionLogitLength {
                expected: self.model().config().vocabulary_size as usize,
                actual: logits.len(),
            });
        }
        for (index, &value) in logits.iter().enumerate() {
            if !value.is_finite() {
                return Err(ChatEngineError::NonFiniteSessionLogit {
                    index,
                    bits: value.to_bits(),
                });
            }
        }

        let kv = self
            .model()
            .export_kv_snapshot(session.generation.model_state(), maximum_bytes)?;
        let fixed_bytes = CHAT_SESSION_MAGIC
            .len()
            .checked_add(4)
            .and_then(|value| value.checked_add(32))
            .and_then(|value| value.checked_add(8 * 3))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(CHAT_SESSION_DIGEST_BYTES))
            .ok_or(ChatEngineError::SessionArithmeticOverflow)?;
        let total_bytes = fixed_bytes
            .checked_add(
                history
                    .len()
                    .checked_mul(4)
                    .ok_or(ChatEngineError::SessionArithmeticOverflow)?,
            )
            .and_then(|value| value.checked_add(logits.len().checked_mul(4)?))
            .and_then(|value| value.checked_add(kv.len()))
            .ok_or(ChatEngineError::SessionArithmeticOverflow)?;
        if total_bytes > maximum_bytes {
            return Err(ChatEngineError::SessionTooLarge {
                actual: total_bytes,
                maximum: maximum_bytes,
            });
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(total_bytes)
            .map_err(|_| ChatEngineError::SessionAllocationFailed {
                requested: total_bytes,
            })?;
        output.extend_from_slice(CHAT_SESSION_MAGIC);
        output.extend_from_slice(&CHAT_SESSION_VERSION.to_le_bytes());
        output.extend_from_slice(&self.model().model_fingerprint());
        output.extend_from_slice(
            &u64::try_from(history.len())
                .map_err(|_| ChatEngineError::SessionArithmeticOverflow)?
                .to_le_bytes(),
        );
        output.extend_from_slice(
            &u64::try_from(logits.len())
                .map_err(|_| ChatEngineError::SessionArithmeticOverflow)?
                .to_le_bytes(),
        );
        output.extend_from_slice(
            &u64::try_from(kv.len())
                .map_err(|_| ChatEngineError::SessionArithmeticOverflow)?
                .to_le_bytes(),
        );
        output.push(u8::from(session.generation.has_logits()));
        output.extend_from_slice(&[0; 3]);
        for &token in history {
            output.extend_from_slice(&token.to_le_bytes());
        }
        for &logit in logits {
            output.extend_from_slice(&logit.to_bits().to_le_bytes());
        }
        output.extend_from_slice(&kv);
        let digest = Sha256::digest(&output);
        output.extend_from_slice(&digest);
        debug_assert_eq!(output.len(), total_bytes);
        Ok(output)
    }

    pub fn restore_session(
        &self,
        bytes: &[u8],
        maximum_bytes: usize,
    ) -> Result<Hy3ChatSession, ChatEngineError> {
        if maximum_bytes == 0 {
            return Err(ChatEngineError::ZeroSessionLimit);
        }
        if bytes.len() > maximum_bytes {
            return Err(ChatEngineError::SessionTooLarge {
                actual: bytes.len(),
                maximum: maximum_bytes,
            });
        }
        if bytes.len() < CHAT_SESSION_DIGEST_BYTES {
            return Err(ChatEngineError::TruncatedSessionSnapshot);
        }
        let payload_end = bytes.len() - CHAT_SESSION_DIGEST_BYTES;
        if Sha256::digest(&bytes[..payload_end]).as_slice() != &bytes[payload_end..] {
            return Err(ChatEngineError::SessionChecksum);
        }
        let mut cursor = SessionCursor::new(&bytes[..payload_end]);
        if cursor.take(CHAT_SESSION_MAGIC.len())? != CHAT_SESSION_MAGIC {
            return Err(ChatEngineError::SessionFormat);
        }
        let version = cursor.u32()?;
        if version != CHAT_SESSION_VERSION {
            return Err(ChatEngineError::SessionVersion {
                expected: CHAT_SESSION_VERSION,
                actual: version,
            });
        }
        if cursor.take(32)? != self.model().model_fingerprint() {
            return Err(ChatEngineError::SessionModelBinding);
        }
        let history_len = cursor.usize()?;
        let logits_len = cursor.usize()?;
        let kv_len = cursor.usize()?;
        let has_logits = match cursor.u8()? {
            0 => false,
            1 => true,
            actual => return Err(ChatEngineError::SessionFlags { actual }),
        };
        if cursor.take(3)? != [0; 3] {
            return Err(ChatEngineError::SessionReservedBytes);
        }
        if history_len > self.model().context_length() {
            return Err(ChatEngineError::SessionHistoryLength {
                actual: history_len,
                maximum: self.model().context_length(),
            });
        }
        let expected_logits = self.model().config().vocabulary_size as usize;
        if logits_len != expected_logits {
            return Err(ChatEngineError::SessionLogitLength {
                expected: expected_logits,
                actual: logits_len,
            });
        }
        if has_logits != (history_len > 0) {
            return Err(ChatEngineError::SessionLogitState {
                history: history_len,
                has_logits,
            });
        }
        let mut history = Vec::new();
        history
            .try_reserve_exact(history_len)
            .map_err(|_| ChatEngineError::SessionAllocationFailed {
                requested: history_len,
            })?;
        for _ in 0..history_len {
            let token = cursor.u32()?;
            if token as usize >= expected_logits {
                return Err(ChatEngineError::SessionTokenOutOfRange {
                    token,
                    vocabulary_size: expected_logits,
                });
            }
            history.push(token);
        }
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(logits_len)
            .map_err(|_| ChatEngineError::SessionAllocationFailed {
                requested: logits_len,
            })?;
        for index in 0..logits_len {
            let value = f32::from_bits(cursor.u32()?);
            if !value.is_finite() {
                return Err(ChatEngineError::NonFiniteSessionLogit {
                    index,
                    bits: value.to_bits(),
                });
            }
            logits.push(value);
        }
        let kv = cursor.take(kv_len)?;
        if cursor.remaining() != 0 {
            return Err(ChatEngineError::SessionTrailingBytes {
                actual: cursor.remaining(),
            });
        }

        let mut session = self.new_session()?;
        self.model()
            .restore_kv_snapshot(session.generation.model_state_mut(), kv, maximum_bytes)?;
        let position = self.generator.position(&session.generation);
        if position != history_len {
            return Err(ChatEngineError::SessionPositionMismatch {
                history: history_len,
                position,
            });
        }
        session.generation.restore_metadata(history, logits, has_logits);
        Ok(session)
    }
}

struct SessionCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SessionCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ChatEngineError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ChatEngineError::SessionArithmeticOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ChatEngineError::TruncatedSessionSnapshot)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ChatEngineError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ChatEngineError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| ChatEngineError::TruncatedSessionSnapshot)?,
        ))
    }

    fn usize(&mut self) -> Result<usize, ChatEngineError> {
        usize::try_from(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ChatEngineError::TruncatedSessionSnapshot)?,
        ))
        .map_err(|_| ChatEngineError::SessionArithmeticOverflow)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn truncate_at_first_stop<'a>(text: &'a str, stop_sequences: &[String]) -> &'a str {
    let stop_at = stop_sequences
        .iter()
        .filter_map(|stop| text.find(stop))
        .min()
        .unwrap_or(text.len());
    &text[..stop_at]
}

#[derive(Debug)]
struct TextStopFilter<'a> {
    stop_sequences: &'a [String],
    pending: String,
}

impl<'a> TextStopFilter<'a> {
    fn new(stop_sequences: &'a [String]) -> Self {
        Self {
            stop_sequences,
            pending: String::new(),
        }
    }

    fn push(&mut self, chunk: &str) -> TextStopUpdate {
        self.pending.push_str(chunk);
        let stop_at = self
            .stop_sequences
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min();
        if let Some(stop_at) = stop_at {
            let text = self.pending[..stop_at].to_owned();
            self.pending.clear();
            return TextStopUpdate { text, stopped: true };
        }

        let keep = self
            .stop_sequences
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(index, _)| index)
                    .chain(std::iter::once(stop.len()))
                    .map(|prefix_len| &stop[..prefix_len])
            })
            .filter(|prefix| self.pending.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let emit_len = self.pending.len() - keep;
        let text = self.pending[..emit_len].to_owned();
        self.pending.drain(..emit_len);
        TextStopUpdate { text, stopped: false }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct TextStopUpdate {
    text: String,
    stopped: bool,
}

#[derive(Debug)]
pub struct ChatCompletion {
    pub prompt_token_ids: Vec<u32>,
    pub cached_prompt_tokens: usize,
    pub text: String,
    pub raw_text: String,
    pub assistant: AssistantOutput,
    pub structured_output_error: Option<String>,
    pub generation: GenerationOutcome,
}

#[derive(Debug, thiserror::Error)]
pub enum ChatEngineError {
    #[error(transparent)]
    Split(#[from] bridge_gguf_split::SplitError),
    #[error(transparent)]
    Tokenizer(#[from] bridge_tokenizer::TokenizerError),
    #[error(transparent)]
    Model(#[from] Hy3ScalarError),
    #[error(transparent)]
    Sampling(#[from] crate::SamplingError),
    #[error(transparent)]
    Generation(#[from] GenerationError<Hy3ScalarError>),
    #[error("GGUF set contains no metadata shard")]
    MissingMetadataShard,
    #[error("tokenizer vocabulary has {tokenizer} entries, model expects {model}")]
    VocabularyMismatch { tokenizer: usize, model: usize },
    #[error("incremental tokenizer output diverged from final decoding")]
    IncrementalDecodeDiverged,
    #[error("stop sequence {index} must not be empty")]
    EmptyStopSequence { index: usize },
    #[error("session snapshot byte limit must be greater than zero")]
    ZeroSessionLimit,
    #[error("a poisoned generation session cannot be persisted")]
    UnhealthySessionSnapshot,
    #[error("session history has {history} tokens but model KV position is {position}")]
    SessionPositionMismatch { history: usize, position: usize },
    #[error("session logits have length {actual}, expected {expected}")]
    SessionLogitLength { expected: usize, actual: usize },
    #[error("session logit {index} is non-finite (F32 bits {bits:#010x})")]
    NonFiniteSessionLogit { index: usize, bits: u32 },
    #[error("checked arithmetic overflow while processing a session snapshot")]
    SessionArithmeticOverflow,
    #[error("session snapshot is {actual} bytes, maximum is {maximum}")]
    SessionTooLarge { actual: usize, maximum: usize },
    #[error("allocation failed while reserving {requested} session snapshot entries")]
    SessionAllocationFailed { requested: usize },
    #[error("session snapshot is truncated")]
    TruncatedSessionSnapshot,
    #[error("session snapshot checksum does not match")]
    SessionChecksum,
    #[error("session snapshot has an invalid format marker")]
    SessionFormat,
    #[error("session snapshot version {actual} is unsupported; expected {expected}")]
    SessionVersion { expected: u32, actual: u32 },
    #[error("session snapshot belongs to a different model")]
    SessionModelBinding,
    #[error("session snapshot has invalid flags byte {actual:#04x}")]
    SessionFlags { actual: u8 },
    #[error("session snapshot reserved bytes are non-zero")]
    SessionReservedBytes,
    #[error("session history has {actual} tokens, maximum is {maximum}")]
    SessionHistoryLength { actual: usize, maximum: usize },
    #[error("session history length {history} is incompatible with has_logits={has_logits}")]
    SessionLogitState { history: usize, has_logits: bool },
    #[error("session token {token} is outside vocabulary size {vocabulary_size}")]
    SessionTokenOutOfRange { token: u32, vocabulary_size: usize },
    #[error("session snapshot has {actual} trailing payload bytes")]
    SessionTrailingBytes { actual: usize },
}

#[cfg(test)]
mod tests {
    use super::{TextStopFilter, TextStopUpdate};

    #[test]
    fn stop_filter_holds_prefixes_and_never_emits_a_matched_sequence() {
        let stops = vec!["</done>".to_owned(), "halt".to_owned()];
        let mut filter = TextStopFilter::new(&stops);
        assert_eq!(
            filter.push("answer</do"),
            TextStopUpdate {
                text: "answer".into(),
                stopped: false,
            }
        );
        assert_eq!(
            filter.push("ne>ignored"),
            TextStopUpdate {
                text: String::new(),
                stopped: true,
            }
        );
        assert!(filter.finish().is_empty());
    }

    #[test]
    fn stop_filter_releases_a_partial_prefix_when_it_diverges() {
        let stops = vec!["stop".to_owned()];
        let mut filter = TextStopFilter::new(&stops);
        assert_eq!(filter.push("a st").text, "a ");
        assert_eq!(filter.push("ep").text, "step");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn stop_filter_matches_unicode_across_chunks() {
        let stops = vec!["slut🛑".to_owned()];
        let mut filter = TextStopFilter::new(&stops);
        assert_eq!(filter.push("klart sl").text, "klart ");
        assert!(!filter.push("ut").stopped);
        assert!(filter.push("🛑tail").stopped);
    }
}
