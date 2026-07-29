use std::collections::BTreeSet;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub stop_tokens: BTreeSet<u32>,
    pub emit_stop_token: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.9,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            stop_tokens: BTreeSet::new(),
            emit_stop_token: false,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self, vocabulary_size: usize) -> Result<(), SamplingError> {
        if vocabulary_size == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if !(self.top_p.is_finite() && 0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(SamplingError::InvalidTopP(self.top_p));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(SamplingError::InvalidRepetitionPenalty(self.repetition_penalty));
        }
        if let Some(&token_id) = self
            .stop_tokens
            .iter()
            .find(|&&token_id| token_id as usize >= vocabulary_size)
        {
            return Err(SamplingError::StopTokenOutOfRange {
                token_id,
                vocabulary_size,
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SamplingError {
    #[error("cannot sample an empty vocabulary")]
    EmptyVocabulary,
    #[error("temperature must be finite and non-negative, got {0}")]
    InvalidTemperature(f32),
    #[error("top_p must be finite and in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("repetition penalty must be finite and positive, got {0}")]
    InvalidRepetitionPenalty(f32),
    #[error("stop token {token_id} is outside vocabulary size {vocabulary_size}")]
    StopTokenOutOfRange { token_id: u32, vocabulary_size: usize },
    #[error("logit count {actual} does not match vocabulary size {expected}")]
    LogitCountMismatch { expected: usize, actual: usize },
    #[error("model produced NaN at token ID {token_id}")]
    NanLogit { token_id: u32 },
    #[error("all candidate logits are negative infinity")]
    NoCandidate,
    #[error("allocation failed while reserving sampler storage")]
    AllocationFailed,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    token_id: u32,
    logit: f32,
    weight: f64,
}

#[derive(Debug)]
pub struct Sampler {
    config: SamplingConfig,
    vocabulary_size: usize,
    rng: ChaCha8Rng,
    candidates: Vec<Candidate>,
    repeated: Vec<bool>,
}

impl Sampler {
    pub fn new(config: SamplingConfig, vocabulary_size: usize) -> Result<Self, SamplingError> {
        config.validate(vocabulary_size)?;
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(vocabulary_size)
            .map_err(|_| SamplingError::AllocationFailed)?;
        let mut repeated = Vec::new();
        repeated
            .try_reserve_exact(vocabulary_size)
            .map_err(|_| SamplingError::AllocationFailed)?;
        repeated.resize(vocabulary_size, false);
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
            vocabulary_size,
            candidates,
            repeated,
        })
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<u32, SamplingError> {
        if logits.len() != self.vocabulary_size {
            return Err(SamplingError::LogitCountMismatch {
                expected: self.vocabulary_size,
                actual: logits.len(),
            });
        }
        for (token_id, &logit) in logits.iter().enumerate() {
            if logit.is_nan() {
                return Err(SamplingError::NanLogit {
                    token_id: token_id as u32,
                });
            }
        }

        self.repeated.fill(false);
        if self.config.repetition_penalty != 1.0 && self.config.repeat_last_n > 0 {
            let start = history.len().saturating_sub(self.config.repeat_last_n);
            for &token_id in &history[start..] {
                if let Some(repeated) = self.repeated.get_mut(token_id as usize) {
                    *repeated = true;
                }
            }
        }

        self.candidates.clear();
        for (token_id, &raw_logit) in logits.iter().enumerate() {
            let logit = if self.repeated[token_id] {
                apply_repetition_penalty(raw_logit, self.config.repetition_penalty)
            } else {
                raw_logit
            };
            self.candidates.push(Candidate {
                token_id: token_id as u32,
                logit,
                weight: 0.0,
            });
        }
        self.candidates.sort_unstable_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });

        if self.config.temperature == 0.0 {
            return self
                .candidates
                .first()
                .filter(|candidate| candidate.logit != f32::NEG_INFINITY)
                .map(|candidate| candidate.token_id)
                .ok_or(SamplingError::NoCandidate);
        }

        let keep = if self.config.top_k == 0 {
            self.candidates.len()
        } else {
            self.config.top_k.min(self.candidates.len())
        };
        self.candidates.truncate(keep);
        let positive_infinity_count = self
            .candidates
            .iter()
            .take_while(|candidate| candidate.logit == f32::INFINITY)
            .count();
        if positive_infinity_count > 0 {
            self.candidates.truncate(positive_infinity_count);
            for candidate in &mut self.candidates {
                candidate.weight = 1.0;
            }
        } else {
            let Some(maximum) = self.candidates.first().map(|candidate| candidate.logit) else {
                return Err(SamplingError::NoCandidate);
            };
            if maximum == f32::NEG_INFINITY {
                return Err(SamplingError::NoCandidate);
            }
            let temperature = f64::from(self.config.temperature);
            for candidate in &mut self.candidates {
                candidate.weight = ((f64::from(candidate.logit) - f64::from(maximum)) / temperature).exp();
            }
        }

        apply_top_p(&mut self.candidates, self.config.top_p);
        let total = self
            .candidates
            .iter()
            .map(|candidate| candidate.weight)
            .sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplingError::NoCandidate);
        }
        let target = self.rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        for candidate in &self.candidates {
            cumulative += candidate.weight;
            if target < cumulative {
                return Ok(candidate.token_id);
            }
        }
        self.candidates
            .last()
            .map(|candidate| candidate.token_id)
            .ok_or(SamplingError::NoCandidate)
    }
}

fn apply_repetition_penalty(logit: f32, penalty: f32) -> f32 {
    if logit < 0.0 {
        logit * penalty
    } else {
        logit / penalty
    }
}

fn apply_top_p(candidates: &mut Vec<Candidate>, top_p: f32) {
    if top_p >= 1.0 || candidates.len() <= 1 {
        return;
    }
    let total = candidates.iter().map(|candidate| candidate.weight).sum::<f64>();
    let threshold = total * f64::from(top_p);
    let mut cumulative = 0.0;
    let mut keep = 0;
    for candidate in candidates.iter() {
        cumulative += candidate.weight;
        keep += 1;
        if cumulative >= threshold {
            break;
        }
    }
    candidates.truncate(keep.max(1));
}
