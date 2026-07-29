use bridge_kernels_reference::ReferenceExecutionMode;
use bridge_runtime::{CancellationToken, CausalModel, Generator, SamplingConfig, StopReason};
use bridge_test_model::{
    ReducedHy3Model, ReducedHy3Session, TestModelError, CONTEXT_LENGTH, VOCABULARY_SIZE,
};

struct ReducedAdapter {
    model: ReducedHy3Model,
}

impl CausalModel for ReducedAdapter {
    type Session = ReducedHy3Session;
    type Error = TestModelError;

    fn vocabulary_size(&self) -> usize {
        VOCABULARY_SIZE
    }

    fn context_length(&self) -> usize {
        CONTEXT_LENGTH
    }

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        self.model.new_session()
    }

    fn reset_session(&self, session: &mut Self::Session) {
        session.reset();
    }

    fn position(&self, session: &Self::Session) -> usize {
        session.position()
    }

    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error> {
        let output = self
            .model
            .evaluate_token(session, ReferenceExecutionMode::LlamaQ8K, token_id)?;
        logits.copy_from_slice(output.logits);
        Ok(())
    }
}

#[test]
fn complete_two_block_hy3_model_generates_multiple_tokens() {
    let generator = Generator::new(ReducedAdapter {
        model: ReducedHy3Model::new().unwrap(),
    })
    .unwrap();
    let mut session = generator.new_session().unwrap();
    let config = SamplingConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let outcome = generator
        .generate(&mut session, &[3, 11], config, &CancellationToken::new())
        .unwrap();
    assert_eq!(outcome.token_ids.len(), 4);
    assert_eq!(outcome.stop_reason, StopReason::MaxTokens);
    assert_eq!(generator.position(&session), 6);
    assert!(outcome
        .token_ids
        .iter()
        .all(|&token_id| token_id < VOCABULARY_SIZE as u32));
}
