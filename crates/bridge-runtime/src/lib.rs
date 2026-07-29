//! Bounded autoregressive generation, sampling, cancellation, and session state.

mod chat_engine;
mod generation;
mod hy3_scalar;
mod sampling;

pub use chat_engine::{ChatCompletion, ChatEngineError, Hy3ChatEngine, Hy3ChatSession};
pub use generation::{
    CancellationToken, CausalModel, GeneratedToken, GenerationError, GenerationOutcome, GenerationSession,
    GenerationStats, Generator, StopReason,
};
pub use hy3_scalar::{
    validate_selected_payload, ExpertReadError, ExpertSourceOptions, Hy3MemoryBudget, Hy3ScalarError,
    Hy3ScalarModel, Hy3ScalarOptions, Hy3ScalarSession, SelectedPayloadFile, SelectedPayloadReport,
    DEFAULT_MEMORY_HEADROOM_BYTES, SELECTED_HY3_IQ2_M_BYTES, SELECTED_HY3_IQ2_M_SHA256,
};
pub use sampling::{Sampler, SamplingConfig, SamplingError};
