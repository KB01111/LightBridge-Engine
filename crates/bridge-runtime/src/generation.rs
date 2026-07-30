use std::error::Error;
use std::ops::ControlFlow;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{Sampler, SamplingConfig, SamplingError};

pub trait CausalModel: Send + Sync {
    type Session: Send;
    type Error: Error + Send + Sync + 'static;

    fn vocabulary_size(&self) -> usize;
    fn context_length(&self) -> usize;
    fn new_session(&self) -> Result<Self::Session, Self::Error>;
    fn reset_session(&self, session: &mut Self::Session);
    fn position(&self, session: &Self::Session) -> usize;
    /// Returns the preferred number of prompt tokens to evaluate in each prefill chunk.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(model.preferred_prefill_chunk(), 1);
    /// ```
    fn preferred_prefill_chunk(&self) -> usize {
        1
    }
    /// Specifies the speculative n-gram width supported by the model.
    ///
    /// The default implementation indicates that speculative n-gram execution is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(None::<usize>, None);
    /// ```
    fn speculative_ngram_t(&self) -> Option<usize> {
        None
    }
    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error>;

    /// Advances the model state for one token and optionally projects logits.
    ///
    /// The default implementation evaluates the token regardless of `project_logits`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_token_with_projection(&mut session, token_id, &mut logits, true)?;
    /// # Ok::<(), ModelError>(())
    /// ```
    ///
    /// `project_logits` indicates whether the model should compute output logits for
    /// the updated state. Models that support separate state advancement and logits
    /// projection can override this method.
    ///
    /// # Parameters
    ///
    /// * `project_logits` — Whether to project logits for the updated state.
    ///
    /// # Errors
    ///
    /// Returns the model-specific error produced while evaluating the token.
    fn evaluate_token_with_projection(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Self::Error> {
        let _ = project_logits;
        self.evaluate_token(session, token_id, logits)
    }

    /// Processes tokens in order and optionally computes logits for the final token.
    ///
    /// By default, each token is evaluated sequentially. When `project_logits` is
    /// `true`, logits are projected only for the final token in `token_ids`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_tokens_with_projection(
    ///     &mut session,
    ///     &[first_token, second_token],
    ///     &mut logits,
    ///     true,
    /// )?;
    /// ```
    fn evaluate_tokens_with_projection(
        &self,
        session: &mut Self::Session,
        token_ids: &[u32],
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Self::Error> {
        for (index, &token_id) in token_ids.iter().enumerate() {
            self.evaluate_token_with_projection(
                session,
                token_id,
                logits,
                project_logits && index + 1 == token_ids.len(),
            )?;
        }
        Ok(())
    }

    /// Evaluates a speculative group of tokens and writes logits for each resulting position in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// fn evaluate_group<M: CausalModel>(
    ///     model: &M,
    ///     session: &mut M::Session,
    ///     token_ids: &[u32],
    ///     logits: &mut [f32],
    /// ) {
    ///     let _ = model.evaluate_speculative_tokens(session, token_ids, logits);
    /// }
    /// ```
    ///
    /// Returns `Some(Ok(()))` when evaluation succeeds, `Some(Err(_))` when model
    /// evaluation fails, or `None` when speculative evaluation is unsupported.
    fn evaluate_speculative_tokens(
        &self,
        _session: &mut Self::Session,
        _token_ids: &[u32],
        _logits: &mut [f32],
    ) -> Option<Result<(), Self::Error>> {
        None
    }

    /// Provides an optional hook for restoring model state to a committed position.
    ///
    /// The default implementation reports that speculative rewinding is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// # let result: Option<Result<(), ()>> = None;
    /// assert!(result.is_none());
    /// ```
    fn rewind_speculative(
        &self,
        _session: &mut Self::Session,
        _position: usize,
    ) -> Option<Result<(), Self::Error>> {
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct GenerationSession<S> {
    model: S,
    history: Vec<u32>,
    logits: Vec<f32>,
    speculative_logits: Vec<f32>,
    has_logits: bool,
    healthy: bool,
}

impl<S> GenerationSession<S> {
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn has_logits(&self) -> bool {
        self.has_logits
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    pub(crate) fn model_state(&self) -> &S {
        &self.model
    }

    pub(crate) fn model_state_mut(&mut self) -> &mut S {
        &mut self.model
    }

    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Restores the session's history and logits metadata, clears speculative logits, and marks the session healthy.
    ///
    /// # Arguments
    ///
    /// * `history` - Committed token history to restore.
    /// * `logits` - Logits associated with the restored history.
    /// * `has_logits` - Whether the restored logits are valid for the history.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn example<S>(session: &mut GenerationSession<S>) {
    /// session.restore_metadata(vec![1, 2], vec![0.0; 10], true);
    /// assert_eq!(session.history(), &[1, 2]);
    /// assert!(session.has_logits());
    /// assert!(session.is_healthy());
    /// # }
    /// ```
    pub(crate) fn restore_metadata(&mut self, history: Vec<u32>, logits: Vec<f32>, has_logits: bool) {
        self.history = history;
        self.logits = logits;
        self.speculative_logits.fill(0.0);
        self.has_logits = has_logits;
        self.healthy = true;
    }
}

#[derive(Debug)]
pub struct Generator<M> {
    model: M,
}

impl<M: CausalModel> Generator<M> {
    pub fn new(model: M) -> Result<Self, SamplingError> {
        SamplingConfig::default().validate(model.vocabulary_size())?;
        Ok(Self { model })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// Creates a fresh generation session with model state and allocated logit storage.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn example<M: CausalModel>(
    /// #     generator: &Generator<M>,
    /// # ) -> Result<(), GenerationError<M::Error>> {
    /// let session = generator.new_session()?;
    /// assert!(!session.has_logits());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The session starts with an empty history and is marked healthy. Its speculative
    /// logit storage is sized according to the model's speculative decoding support.
    pub fn new_session() -> Result<GenerationSession<M::Session>, GenerationError<M::Error>>
    pub fn new_session(&self) -> Result<GenerationSession<M::Session>, GenerationError<M::Error>> {
        let model = self.model.new_session().map_err(GenerationError::Model)?;
        let vocabulary_size = self.model.vocabulary_size();
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(vocabulary_size)
            .map_err(|_| GenerationError::AllocationFailed)?;
        logits.resize(vocabulary_size, 0.0);
        let speculative_values = match self.model.speculative_ngram_t() {
            Some(2) => vocabulary_size
                .checked_mul(2)
                .ok_or(GenerationError::ArithmeticOverflow)?,
            Some(other) => return Err(GenerationError::InvalidSpeculativeWidth(other)),
            None => 0,
        };
        let mut speculative_logits = Vec::new();
        speculative_logits
            .try_reserve_exact(speculative_values)
            .map_err(|_| GenerationError::AllocationFailed)?;
        speculative_logits.resize(speculative_values, 0.0);
        Ok(GenerationSession {
            model,
            history: Vec::new(),
            logits,
            speculative_logits,
            has_logits: false,
            healthy: true,
        })
    }

    /// Resets the model and clears all generation state in the session.
    
    ///
    
    /// # Examples
    
    ///
    
    /// ```no_run
    
    /// # let generator = todo!();
    
    /// # let mut session = todo!();
    
    /// generator.reset(&mut session);
    
    /// ```
    pub fn reset(&self, session: &mut GenerationSession<M::Session>) {
        self.model.reset_session(&mut session.model);
        session.history.clear();
        session.logits.fill(0.0);
        session.speculative_logits.fill(0.0);
        session.has_logits = false;
        session.healthy = true;
    }

    pub fn position(&self, session: &GenerationSession<M::Session>) -> usize {
        self.model.position(&session.model)
    }

    pub fn generate(
        &self,
        session: &mut GenerationSession<M::Session>,
        prompt_tokens: &[u32],
        config: SamplingConfig,
        cancellation: &CancellationToken,
    ) -> Result<GenerationOutcome, GenerationError<M::Error>> {
        self.generate_stream(session, prompt_tokens, config, cancellation, |_| {
            ControlFlow::Continue(())
        })
    }

    /// Generates tokens from a prompt and emits each generated token to a callback.
    ///
    /// Prompt tokens are evaluated before generation. Generation stops when the configured
    /// limit, context capacity, stop token, cancellation token, or callback control flow
    /// requests it. The session is updated with evaluated and generated tokens.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let cancellation = CancellationToken::new();
    /// let outcome = generator.generate_stream(
    ///     &mut session,
    ///     &prompt_tokens,
    ///     SamplingConfig::default(),
    ///     &cancellation,
    ///     |token| {
    ///         println!("{}", token.token_id);
    ///         ControlFlow::Continue(())
    ///     },
    /// )?;
    /// # Ok::<(), GenerationError<_>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when configuration, prompt tokens, context capacity, session
    /// state, allocation, sampling, or model execution is invalid.
    pub fn generate_stream<F>(
    pub fn generate_stream<F>(
        &self,
        session: &mut GenerationSession<M::Session>,
        prompt_tokens: &[u32],
        config: SamplingConfig,
        cancellation: &CancellationToken,
        mut emit: F,
    ) -> Result<GenerationOutcome, GenerationError<M::Error>>
    where
        F: FnMut(GeneratedToken) -> ControlFlow<()>,
    {
        let started = Instant::now();
        let vocabulary_size = self.model.vocabulary_size();
        config
            .validate(vocabulary_size)
            .map_err(GenerationError::Sampling)?;
        if !session.healthy {
            return Err(GenerationError::SessionPoisoned);
        }
        for &token_id in prompt_tokens {
            if token_id as usize >= vocabulary_size {
                return Err(GenerationError::TokenOutOfRange {
                    token_id,
                    vocabulary_size,
                });
            }
        }

        let position = self.model.position(&session.model);
        let context_length = self.model.context_length();
        if position > context_length || prompt_tokens.len() > context_length - position {
            return Err(GenerationError::PromptExceedsContext {
                position,
                prompt_tokens: prompt_tokens.len(),
                context_length,
            });
        }
        if prompt_tokens.is_empty() && !session.has_logits {
            return Err(GenerationError::EmptyPrompt);
        }

        let remaining_after_prompt = context_length - position - prompt_tokens.len();
        let generation_limit = config.max_new_tokens.min(remaining_after_prompt);
        let requested_history = prompt_tokens
            .len()
            .checked_add(generation_limit)
            .ok_or(GenerationError::ArithmeticOverflow)?;
        session
            .history
            .try_reserve(requested_history)
            .map_err(|_| GenerationError::AllocationFailed)?;
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(generation_limit)
            .map_err(|_| GenerationError::AllocationFailed)?;

        if cancellation.is_cancelled() {
            return Ok(outcome(
                generated,
                StopReason::Cancelled,
                prompt_tokens.len(),
                Duration::ZERO,
                Duration::ZERO,
                started.elapsed(),
            ));
        }

        let prefill_chunk = self.model.preferred_prefill_chunk();
        if !matches!(prefill_chunk, 1 | 2 | 4 | 8) {
            return Err(GenerationError::InvalidPrefillChunk(prefill_chunk));
        }
        let prefill_started = Instant::now();
        for (chunk_index, tokens) in prompt_tokens.chunks(prefill_chunk).enumerate() {
            if cancellation.is_cancelled() {
                return Ok(outcome(
                    generated,
                    StopReason::Cancelled,
                    session.history.len(),
                    prefill_started.elapsed(),
                    Duration::ZERO,
                    started.elapsed(),
                ));
            }
            let consumed = chunk_index
                .checked_mul(prefill_chunk)
                .and_then(|value| value.checked_add(tokens.len()))
                .ok_or(GenerationError::ArithmeticOverflow)?;
            let project_logits = consumed == prompt_tokens.len();
            self.evaluate_many(session, tokens, project_logits)?;
        }
        let prefill_duration = prefill_started.elapsed();
        let mut sampler = Sampler::new(config.clone(), vocabulary_size).map_err(GenerationError::Sampling)?;
        let decode_started = Instant::now();

        let mut decoded_tokens = 0_usize;
        while decoded_tokens < generation_limit {
            if cancellation.is_cancelled() {
                return Ok(outcome(
                    generated,
                    StopReason::Cancelled,
                    prompt_tokens.len(),
                    prefill_duration,
                    decode_started.elapsed(),
                    started.elapsed(),
                ));
            }
            let token_id = sampler
                .sample(&session.logits, &session.history)
                .map_err(GenerationError::Sampling)?;

            let speculative_pair = if generation_limit - decoded_tokens >= 2
                && config.temperature == 0.0
                && config.stop_tokens.is_empty()
                && self.model.speculative_ngram_t() == Some(2)
            {
                ngram_draft_t2(&session.history).filter(|draft| draft[0] == token_id)
            } else {
                None
            };
            if let Some(draft) = speculative_pair {
                let base_position = self.model.position(&session.model);
                let base_history = session.history.len();
                match self.model.evaluate_speculative_tokens(
                    &mut session.model,
                    &draft,
                    &mut session.speculative_logits,
                ) {
                    Some(Ok(())) => {}
                    Some(Err(error)) => {
                        session.healthy = false;
                        session.has_logits = false;
                        return Err(GenerationError::Model(error));
                    }
                    None => {
                        session.healthy = false;
                        session.has_logits = false;
                        return Err(GenerationError::SpeculativeExecutionUnavailable);
                    }
                }

                session.history.push(draft[0]);
                let second_token =
                    match sampler.sample(&session.speculative_logits[..vocabulary_size], &session.history) {
                        Ok(token) => token,
                        Err(error) => {
                            session.history.truncate(base_history);
                            self.rewind_speculative(session, base_position)?;
                            return Err(GenerationError::Sampling(error));
                        }
                    };
                let accepted = [draft[0], second_token];
                if second_token == draft[1] {
                    session.history.push(draft[1]);
                    session
                        .logits
                        .copy_from_slice(&session.speculative_logits[vocabulary_size..2 * vocabulary_size]);
                    session.has_logits = true;
                } else {
                    self.rewind_speculative(session, base_position + 1)?;
                    session
                        .logits
                        .copy_from_slice(&session.speculative_logits[..vocabulary_size]);
                    session.has_logits = true;
                    self.evaluate(session, second_token, true)?;
                }

                for (offset, accepted_token) in accepted.into_iter().enumerate() {
                    let index = decoded_tokens;
                    decoded_tokens += 1;
                    generated.push(accepted_token);
                    if emit(GeneratedToken {
                        token_id: accepted_token,
                        index,
                    })
                    .is_break()
                    {
                        if offset == 0 {
                            session.history.truncate(base_history + 1);
                            self.rewind_speculative(session, base_position + 1)?;
                            session
                                .logits
                                .copy_from_slice(&session.speculative_logits[..vocabulary_size]);
                            session.has_logits = true;
                        }
                        return Ok(outcome(
                            generated,
                            StopReason::Callback,
                            prompt_tokens.len(),
                            prefill_duration,
                            decode_started.elapsed(),
                            started.elapsed(),
                        ));
                    }
                }
                continue;
            }

            self.evaluate(session, token_id, true)?;

            let is_stop = config.stop_tokens.contains(&token_id);
            if !is_stop || config.emit_stop_token {
                let token = GeneratedToken {
                    token_id,
                    index: decoded_tokens,
                };
                generated.push(token_id);
                if emit(token).is_break() {
                    return Ok(outcome(
                        generated,
                        StopReason::Callback,
                        prompt_tokens.len(),
                        prefill_duration,
                        decode_started.elapsed(),
                        started.elapsed(),
                    ));
                }
            }
            decoded_tokens += 1;
            if is_stop {
                return Ok(outcome(
                    generated,
                    StopReason::StopToken(token_id),
                    prompt_tokens.len(),
                    prefill_duration,
                    decode_started.elapsed(),
                    started.elapsed(),
                ));
            }
        }

        let stop_reason = if generation_limit < config.max_new_tokens {
            StopReason::ContextLength
        } else {
            StopReason::MaxTokens
        };
        Ok(outcome(
            generated,
            stop_reason,
            prompt_tokens.len(),
            prefill_duration,
            decode_started.elapsed(),
            started.elapsed(),
        ))
    }

    /// Evaluates a token and updates the generation session state.
    ///
    /// On success, appends the token to the session history and records whether
    /// logits are available. A model error marks the session as unhealthy.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// generator.evaluate(&mut session, token_id, true)?;
    /// assert!(session.has_logits());
    /// # Ok::<(), GenerationError<M::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError::Model`] if token evaluation fails.
    fn evaluate(
        &self,
        session: &mut GenerationSession<M::Session>,
        token_id: u32,
        project_logits: bool,
    ) -> Result<(), GenerationError<M::Error>> {
        if let Err(error) = self.model.evaluate_token_with_projection(
            &mut session.model,
            token_id,
            &mut session.logits,
            project_logits,
        ) {
            session.healthy = false;
            session.has_logits = false;
            return Err(GenerationError::Model(error));
        }
        session.history.push(token_id);
        session.has_logits = project_logits;
        Ok(())
    }

    /// Evaluates multiple tokens and records them in the session history.
    ///
    /// On success, updates logits availability according to `project_logits`. A model
    /// evaluation error marks the session as unhealthy and returns it as a generation
    /// error.
    ///
    /// # Arguments
    ///
    /// * `session` - The generation session to update.
    /// * `token_ids` - The tokens to evaluate in order.
    /// * `project_logits` - Whether the resulting logits correspond to the final token.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// generator.evaluate_many(&mut session, &[1, 2, 3], true)?;
    /// # Ok::<(), GenerationError<M::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GenerationError::Model` if model evaluation fails.
    fn evaluate_many(
        &self,
        session: &mut GenerationSession<M::Session>,
        token_ids: &[u32],
        project_logits: bool,
    ) -> Result<(), GenerationError<M::Error>> {
        if let Err(error) = self.model.evaluate_tokens_with_projection(
            &mut session.model,
            token_ids,
            &mut session.logits,
            project_logits,
        ) {
            session.healthy = false;
            session.has_logits = false;
            return Err(GenerationError::Model(error));
        }
        session.history.extend_from_slice(token_ids);
        session.has_logits = project_logits;
        Ok(())
    }

    /// Rewinds the model session to a position after speculative execution.
    ///
    /// The session is marked unhealthy when the model reports an error or does not
    /// support speculative rewinding.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError::Model`] when the model reports an error, or
    /// [`GenerationError::SpeculativeRewindUnavailable`] when rewinding is unsupported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let result = generator.rewind_speculative(&mut session, position);
    /// result?;
    /// # Ok::<(), GenerationError<M::Error>>(())
    /// ```
    fn rewind_speculative(
        &self,
        session: &mut GenerationSession<M::Session>,
        position: usize,
    ) -> Result<(), GenerationError<M::Error>> {
        match self.model.rewind_speculative(&mut session.model, position) {
            Some(Ok(())) => Ok(()),
            Some(Err(error)) => {
                session.healthy = false;
                session.has_logits = false;
                Err(GenerationError::Model(error))
            }
            None => {
                session.healthy = false;
                session.has_logits = false;
                Err(GenerationError::SpeculativeRewindUnavailable)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    StopToken(u32),
    StopSequence,
    MaxTokens,
    ContextLength,
    Cancelled,
    Callback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
    pub total_duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutcome {
    pub token_ids: Vec<u32>,
    pub stop_reason: StopReason,
    pub stats: GenerationStats,
}

#[derive(Debug, thiserror::Error)]
pub enum GenerationError<E: Error + 'static> {
    #[error("model execution failed: {0}")]
    Model(#[source] E),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error("generation session is poisoned by a prior model failure; reset it before reuse")]
    SessionPoisoned,
    #[error("token ID {token_id} is outside vocabulary size {vocabulary_size}")]
    TokenOutOfRange { token_id: u32, vocabulary_size: usize },
    #[error(
        "prompt of {prompt_tokens} tokens at position {position} exceeds context length {context_length}"
    )]
    PromptExceedsContext {
        position: usize,
        prompt_tokens: usize,
        context_length: usize,
    },
    #[error("an empty prompt requires an existing session logit state")]
    EmptyPrompt,
    #[error("checked arithmetic overflow while sizing generation")]
    ArithmeticOverflow,
    #[error("model requested invalid prefill chunk {0}; expected 1, 2, 4, or 8")]
    InvalidPrefillChunk(usize),
    #[error("model requested speculative width {0}; only T=2 is supported")]
    InvalidSpeculativeWidth(usize),
    #[error("model advertised speculation but did not implement grouped verification")]
    SpeculativeExecutionUnavailable,
    #[error("model advertised speculation but did not implement lossless rewind")]
    SpeculativeRewindUnavailable,
    #[error("allocation failed while reserving bounded generation state")]
    AllocationFailed,
}

/// Finds a two-token continuation by matching a recent history suffix against an earlier sequence.
///
/// # Examples
///
/// ```
/// let history = [1, 2, 3, 1, 2, 3];
/// assert_eq!(ngram_draft_t2(&history), Some([1, 2]));
/// ```
///
/// Returns `Some` with the two tokens following the matching sequence, or `None` if no match is found.
fn ngram_draft_t2(history: &[u32]) -> Option<[u32; 2]> {
    let maximum_suffix = history.len().saturating_sub(2).min(4);
    for suffix_length in (1..=maximum_suffix).rev() {
        let suffix_start = history.len() - suffix_length;
        let latest_candidate = history.len() - suffix_length - 2;
        for candidate_start in (0..=latest_candidate).rev() {
            let candidate_end = candidate_start + suffix_length;
            if history[candidate_start..candidate_end] == history[suffix_start..] {
                return Some([history[candidate_end], history[candidate_end + 1]]);
            }
        }
    }
    None
}

/// Constructs a generation outcome with token data, stop reason, and timing statistics.
///
/// # Examples
///
/// ```
/// let outcome = outcome(
///     vec![1, 2],
///     StopReason::MaxTokens,
///     3,
///     Duration::from_millis(10),
///     Duration::from_millis(20),
///     Duration::from_millis(30),
/// );
///
/// assert_eq!(outcome.token_ids, vec![1, 2]);
/// assert_eq!(outcome.stats.generated_tokens, 2);
/// ```
fn outcome(
    token_ids: Vec<u32>,
    stop_reason: StopReason,
    prompt_tokens: usize,
    prefill_duration: Duration,
    decode_duration: Duration,
    total_duration: Duration,
) -> GenerationOutcome {
    GenerationOutcome {
        stats: GenerationStats {
            prompt_tokens,
            generated_tokens: token_ids.len(),
            prefill_duration,
            decode_duration,
            total_duration,
        },
        token_ids,
        stop_reason,
    }
}
