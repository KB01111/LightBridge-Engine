use std::convert::Infallible;
use std::ops::ControlFlow;

use bridge_runtime::{
    CancellationToken, CausalModel, GenerationError, Generator, SamplingConfig, StopReason,
};

#[derive(Debug)]
struct MockModel {
    context: usize,
}

#[derive(Debug)]
struct MockSession {
    position: usize,
}

impl CausalModel for MockModel {
    type Session = MockSession;
    type Error = Infallible;

    fn vocabulary_size(&self) -> usize {
        4
    }

    fn context_length(&self) -> usize {
        self.context
    }

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(MockSession { position: 0 })
    }

    fn reset_session(&self, session: &mut Self::Session) {
        session.position = 0;
    }

    fn position(&self, session: &Self::Session) -> usize {
        session.position
    }

    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error> {
        session.position += 1;
        logits.fill(-10.0);
        logits[((token_id + 1) % 4) as usize] = 10.0;
        Ok(())
    }
}

fn greedy(max_new_tokens: usize) -> SamplingConfig {
    SamplingConfig {
        max_new_tokens,
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

#[test]
fn prefills_then_generates_and_keeps_session_ready_for_continuation() {
    let generator = Generator::new(MockModel { context: 16 }).unwrap();
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate(&mut session, &[0, 1], greedy(3), &CancellationToken::new())
        .unwrap();
    assert_eq!(outcome.token_ids, [2, 3, 0]);
    assert_eq!(outcome.stop_reason, StopReason::MaxTokens);
    assert_eq!(generator.position(&session), 5);
    assert_eq!(session.history(), [0, 1, 2, 3, 0]);

    let continuation = generator
        .generate(&mut session, &[], greedy(2), &CancellationToken::new())
        .unwrap();
    assert_eq!(continuation.token_ids, [1, 2]);
    assert_eq!(generator.position(&session), 7);
}

#[test]
fn stop_token_is_cached_but_not_emitted_by_default() {
    let generator = Generator::new(MockModel { context: 16 }).unwrap();
    let mut session = generator.new_session().unwrap();
    let mut config = greedy(5);
    config.stop_tokens.insert(2);
    let outcome = generator
        .generate(&mut session, &[1], config, &CancellationToken::new())
        .unwrap();
    assert!(outcome.token_ids.is_empty());
    assert_eq!(outcome.stop_reason, StopReason::StopToken(2));
    assert_eq!(session.history(), [1, 2]);
}

#[test]
fn context_limit_truncates_generation_without_overrun() {
    let generator = Generator::new(MockModel { context: 3 }).unwrap();
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate(&mut session, &[0, 1], greedy(5), &CancellationToken::new())
        .unwrap();
    assert_eq!(outcome.token_ids, [2]);
    assert_eq!(outcome.stop_reason, StopReason::ContextLength);
    assert_eq!(generator.position(&session), 3);
}

#[test]
fn invalid_prompt_is_atomic() {
    let generator = Generator::new(MockModel { context: 4 }).unwrap();
    let mut session = generator.new_session().unwrap();
    let error = generator
        .generate(&mut session, &[0, 4], greedy(1), &CancellationToken::new())
        .unwrap_err();
    assert!(matches!(
        error,
        GenerationError::TokenOutOfRange {
            token_id: 4,
            vocabulary_size: 4
        }
    ));
    assert_eq!(generator.position(&session), 0);
    assert!(session.history().is_empty());
}

#[test]
fn cancellation_and_callback_stop_cleanly() {
    let generator = Generator::new(MockModel { context: 16 }).unwrap();
    let mut session = generator.new_session().unwrap();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let cancelled = generator
        .generate(&mut session, &[0], greedy(2), &cancellation)
        .unwrap();
    assert_eq!(cancelled.stop_reason, StopReason::Cancelled);
    assert_eq!(generator.position(&session), 0);

    let mut seen = Vec::new();
    let callback = generator
        .generate_stream(
            &mut session,
            &[0],
            greedy(4),
            &CancellationToken::new(),
            |token| {
                seen.push(token.token_id);
                ControlFlow::Break(())
            },
        )
        .unwrap();
    assert_eq!(seen, [1]);
    assert_eq!(callback.stop_reason, StopReason::Callback);
    assert_eq!(generator.position(&session), 2);
}

#[test]
fn reset_restores_a_fresh_session() {
    let generator = Generator::new(MockModel { context: 8 }).unwrap();
    let mut session = generator.new_session().unwrap();
    generator
        .generate(&mut session, &[0], greedy(1), &CancellationToken::new())
        .unwrap();
    generator.reset(&mut session);
    assert_eq!(generator.position(&session), 0);
    assert!(session.history().is_empty());
    assert!(!session.has_logits());
    assert!(session.is_healthy());
}
