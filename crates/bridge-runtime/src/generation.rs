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
    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error>;
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

    pub(crate) fn restore_metadata(&mut self, history: Vec<u32>, logits: Vec<f32>, has_logits: bool) {
        self.history = history;
        self.logits = logits;
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

    pub fn new_session(&self) -> Result<GenerationSession<M::Session>, GenerationError<M::Error>> {
        let model = self.model.new_session().map_err(GenerationError::Model)?;
        let vocabulary_size = self.model.vocabulary_size();
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(vocabulary_size)
            .map_err(|_| GenerationError::AllocationFailed)?;
        logits.resize(vocabulary_size, 0.0);
        Ok(GenerationSession {
            model,
            history: Vec::new(),
            logits,
            has_logits: false,
            healthy: true,
        })
    }

    pub fn reset(&self, session: &mut GenerationSession<M::Session>) {
        self.model.reset_session(&mut session.model);
        session.history.clear();
        session.logits.fill(0.0);
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

        let prefill_started = Instant::now();
        for &token_id in prompt_tokens {
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
            self.evaluate(session, token_id)?;
        }
        let prefill_duration = prefill_started.elapsed();
        let mut sampler = Sampler::new(config.clone(), vocabulary_size).map_err(GenerationError::Sampling)?;
        let decode_started = Instant::now();

        for index in 0..generation_limit {
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
            self.evaluate(session, token_id)?;

            let is_stop = config.stop_tokens.contains(&token_id);
            if !is_stop || config.emit_stop_token {
                let token = GeneratedToken { token_id, index };
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

    fn evaluate(
        &self,
        session: &mut GenerationSession<M::Session>,
        token_id: u32,
    ) -> Result<(), GenerationError<M::Error>> {
        if let Err(error) = self
            .model
            .evaluate_token(&mut session.model, token_id, &mut session.logits)
        {
            session.healthy = false;
            session.has_logits = false;
            return Err(GenerationError::Model(error));
        }
        session.history.push(token_id);
        session.has_logits = true;
        Ok(())
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
    #[error("allocation failed while reserving bounded generation state")]
    AllocationFailed,
}

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
