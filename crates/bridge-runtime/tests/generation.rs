use std::convert::Infallible;
use std::ops::ControlFlow;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

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

#[test]
fn prefill_projects_only_the_final_prompt_position() {
    struct CountingModel {
        projections: Arc<AtomicUsize>,
        batches: Arc<AtomicUsize>,
    }

    impl CausalModel for CountingModel {
        type Session = usize;
        type Error = Infallible;

        /// Reports the fixed number of tokens supported by the model.
        ///
        /// # Examples
        ///
        /// ```
        /// let model = MockModel { context: 16 };
        /// assert_eq!(model.vocabulary_size(), 4);
        /// ```
        fn vocabulary_size(&self) -> usize {
            4
        }

        /// Reports the model's maximum supported context length.
        ///
        /// # Examples
        ///
        /// ```ignore
        /// assert_eq!(model.context_length(), 16);
        /// ```
        fn context_length(&self) -> usize {
            16
        }

        /// Creates a session initialized at position zero.
        ///
        /// # Examples
        ///
        /// ```
        /// let session = model.new_session().unwrap();
        /// assert_eq!(session, 0);
        /// ```
        fn new_session(&self) -> Result<Self::Session, Self::Error> {
            Ok(0)
        }

        fn reset_session(&self, session: &mut Self::Session) {
            *session = 0;
        }

        /// Reports the current position in the session.
        ///
        /// # Examples
        ///
        /// ```
        /// let session = 3;
        /// assert_eq!(*&session, 3);
        /// ```
        fn position(&self, session: &Self::Session) -> usize {
            *session
        }

        /// Specifies that prefill should process tokens in chunks of two.
        ///
        /// # Examples
        ///
        /// ```
        /// let chunk_size = 2;
        /// assert_eq!(chunk_size, 2);
        /// ```
        fn preferred_prefill_chunk(&self) -> usize {
            2
        }

        /// Evaluates a token for the session and updates the logits with the predicted next-token scores.
        ///
        /// # Examples
        ///
        /// ```ignore
        /// let model = MockModel { context: 16 };
        /// let mut session = model.new_session();
        /// let mut logits = vec![0.0; model.vocabulary_size()];
        ///
        /// model.evaluate_token(&mut session, 0, &mut logits).unwrap();
        /// assert_eq!(model.position(&session), 1);
        /// ```
        ///
        /// # Returns
        ///
        /// `Ok(())` when evaluation succeeds.
        fn evaluate_token(
            &self,
            session: &mut Self::Session,
            token_id: u32,
            logits: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.evaluate_token_with_projection(session, token_id, logits, true)
        }

        /// Evaluates one token and optionally projects its logits into the provided buffer.
        ///
        /// When projection is enabled, the logits identify `(token_id + 1) % 4` as the
        /// preferred next token and all other vocabulary entries receive a lower score.
        ///
        /// # Examples
        ///
        /// ```rust,ignore
        /// let mut logits = vec![0.0; 4];
        /// model.evaluate_token_with_projection(&mut session, 0, &mut logits, true)?;
        /// assert_eq!(logits[1], 10.0);
        /// # Ok::<(), _>(())
        /// ```
        ///
        /// # Arguments
        ///
        /// * `token_id` - Token whose deterministic successor is represented in `logits`.
        /// * `logits` - Buffer to populate when projection is enabled.
        /// * `project_logits` - Whether to populate the logits buffer.
        fn evaluate_token_with_projection(
            &self,
            session: &mut Self::Session,
            token_id: u32,
            logits: &mut [f32],
            project_logits: bool,
        ) -> Result<(), Self::Error> {
            *session += 1;
            if project_logits {
                self.projections.fetch_add(1, Ordering::Relaxed);
                logits.fill(-10.0);
                logits[((token_id + 1) % 4) as usize] = 10.0;
            }
            Ok(())
        }

        /// Evaluates a batch of tokens and optionally projects logits for the final token.
        ///
        /// # Parameters
        ///
        /// * `token_ids` - Tokens to evaluate in order.
        /// * `project_logits` - Whether to project logits for the final token.
        ///
        /// # Examples
        ///
        /// ```ignore
        /// model.evaluate_tokens_with_projection(&mut session, &tokens, &mut logits, true)?;
        /// # Ok::<(), ModelError>(())
        /// ```
        ///
        /// # Errors
        ///
        /// Returns the first error encountered while evaluating a token.
        fn evaluate_tokens_with_projection(
            &self,
            session: &mut Self::Session,
            token_ids: &[u32],
            logits: &mut [f32],
            project_logits: bool,
        ) -> Result<(), Self::Error> {
            self.batches.fetch_add(1, Ordering::Relaxed);
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
    }

    let projections = Arc::new(AtomicUsize::new(0));
    let batches = Arc::new(AtomicUsize::new(0));
    let generator = Generator::new(CountingModel {
        projections: Arc::clone(&projections),
        batches: Arc::clone(&batches),
    })
    .unwrap();
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate(&mut session, &[0, 1, 2, 3], greedy(0), &CancellationToken::new())
        .unwrap();
    assert!(outcome.token_ids.is_empty());
    assert_eq!(generator.position(&session), 4);
    assert!(session.has_logits());
    assert_eq!(projections.load(Ordering::Relaxed), 1);
    assert_eq!(batches.load(Ordering::Relaxed), 2);
}

#[derive(Debug)]
struct SpeculativeModel {
    reject_second: bool,
    grouped_calls: Arc<AtomicUsize>,
    rewinds: Arc<AtomicUsize>,
}

impl SpeculativeModel {
    /// Writes deterministic logits for the next token in the speculative decoding sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = SpeculativeModel {
    ///     reject_second: false,
    ///     grouped_calls: Arc::new(AtomicUsize::new(0)),
    ///     rewinds: Arc::new(AtomicUsize::new(0)),
    /// };
    /// let mut logits = [0.0; 4];
    ///
    /// model.write_logits(0, &mut logits);
    ///
    /// assert_eq!(logits, [-10.0, 10.0, -10.0, -10.0]);
    /// ```
    fn write_logits(&self, token_id: u32, logits: &mut [f32]) {
        let next = match token_id {
            0 => 1,
            1 if self.reject_second => 3,
            1 => 2,
            2 => 0,
            _ => 0,
        };
        logits.fill(-10.0);
        logits[next] = 10.0;
    }
}

impl CausalModel for SpeculativeModel {
    type Session = usize;
    type Error = Infallible;

    /// Reports the model's fixed vocabulary size.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = MockModel { context: 16 };
    /// assert_eq!(model.vocabulary_size(), 4);
    /// ```
    fn vocabulary_size(&self) -> usize {
        4
    }

    /// Reports the maximum number of tokens supported by the model context.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(32, 32);
    /// ```
    fn context_length(&self) -> usize {
        32
    }

    /// Creates a new session positioned at the start of evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// let session = model.new_session()?;
    /// assert_eq!(session, 0);
    /// # Ok::<(), _>(())
    /// ```
    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(0)
    }

    /// Resets the session position to the beginning.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut session = 5;
    /// *session = 0;
    /// assert_eq!(session, 0);
    /// ```
    fn reset_session(&self, session: &mut Self::Session) {
        *session = 0;
    }

    /// Reports the current position within a session.
    ///
    /// # Examples
    ///
    /// ```
    /// let model = MockModel { context: 16 };
    /// let session = model.new_session();
    ///
    /// assert_eq!(model.position(&session), 0);
    /// ```
    fn position(&self, session: &Self::Session) -> usize {
        *session
    }

    /// Specifies that prompt prefill should process two tokens per chunk.
    ///
    /// # Examples
    ///
    /// ```
    /// let chunk_size = 2;
    /// assert_eq!(chunk_size, 2);
    /// ```
    fn preferred_prefill_chunk(&self) -> usize {
        2
    }

    /// Enables speculative n-gram generation with a width of two.
    ///
    /// # Examples
    ///
    /// ```
    /// let width: Option<usize> = Some(2);
    /// assert_eq!(width, Some(2));
    /// ```
    fn speculative_ngram_t(&self) -> Option<usize> {
        Some(2)
    }

    /// Evaluates a token, advances the session position, and writes the resulting logits.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut session = 0;
    /// let mut logits = vec![0.0; 4];
    /// model.evaluate_token(&mut session, 0, &mut logits)?;
    /// assert_eq!(session, 1);
    /// # Ok::<(), _>(())
    /// ```
    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error> {
        *session += 1;
        self.write_logits(token_id, logits);
        Ok(())
    }

    /// Evaluates a batch of speculative tokens and writes one logits row for each token.
    ///
    /// The session position advances once for each token paired with a four-element logits
    /// row. This implementation always reports successful evaluation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut session = model.new_session();
    /// let token_ids = [0, 1];
    /// let mut logits = vec![0.0; 8];
    ///
    /// model.evaluate_speculative_tokens(&mut session, &token_ids, &mut logits);
    /// assert_eq!(model.position(&session), 2);
    /// ```
    fn evaluate_speculative_tokens(
        &self,
        session: &mut Self::Session,
        token_ids: &[u32],
        logits: &mut [f32],
    ) -> Option<Result<(), Self::Error>> {
        self.grouped_calls.fetch_add(1, Ordering::Relaxed);
        for (&token_id, row) in token_ids.iter().zip(logits.chunks_exact_mut(4)) {
            *session += 1;
            self.write_logits(token_id, row);
        }
        Some(Ok(()))
    }

    /// Rewinds the speculative session to the specified position.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut session = model.new_session();
    /// let result = model.rewind_speculative(&mut session, 2);
    ///
    /// assert!(result.is_some());
    /// assert_eq!(model.position(&session), 2);
    /// ```
    fn rewind_speculative(
        &self,
        session: &mut Self::Session,
        position: usize,
    ) -> Option<Result<(), Self::Error>> {
        self.rewinds.fetch_add(1, Ordering::Relaxed);
        *session = position;
        Some(Ok(()))
    }
}

/// Creates a speculative-decoding generator and counters for grouped evaluations and rewinds.
///
/// # Parameters
///
/// `reject_second` determines whether the speculative model rejects its second candidate token.
///
/// # Returns
///
/// A generator, a counter for grouped speculative evaluations, and a counter for speculative rewinds.
///
/// # Examples
///
/// ```
/// let (_generator, grouped_calls, rewinds) = speculative_generator(false);
/// assert_eq!(grouped_calls.load(Ordering::Relaxed), 0);
/// assert_eq!(rewinds.load(Ordering::Relaxed), 0);
/// ```
fn speculative_generator(
    reject_second: bool,
) -> (Generator<SpeculativeModel>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let grouped_calls = Arc::new(AtomicUsize::new(0));
    let rewinds = Arc::new(AtomicUsize::new(0));
    let generator = Generator::new(SpeculativeModel {
        reject_second,
        grouped_calls: Arc::clone(&grouped_calls),
        rewinds: Arc::clone(&rewinds),
    })
    .unwrap();
    (generator, grouped_calls, rewinds)
}

#[test]
fn t2_ngram_speculation_accepts_a_matching_pair() {
    let (generator, grouped_calls, rewinds) = speculative_generator(false);
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate(&mut session, &[0, 1, 2, 0], greedy(2), &CancellationToken::new())
        .unwrap();

    assert_eq!(outcome.token_ids, [1, 2]);
    assert_eq!(session.history(), [0, 1, 2, 0, 1, 2]);
    assert_eq!(generator.position(&session), 6);
    assert_eq!(grouped_calls.load(Ordering::Relaxed), 1);
    assert_eq!(rewinds.load(Ordering::Relaxed), 0);
}

#[test]
fn t2_ngram_speculation_rewinds_and_replays_a_rejected_second_token() {
    let (generator, grouped_calls, rewinds) = speculative_generator(true);
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate(&mut session, &[0, 1, 2, 0], greedy(2), &CancellationToken::new())
        .unwrap();

    assert_eq!(outcome.token_ids, [1, 3]);
    assert_eq!(session.history(), [0, 1, 2, 0, 1, 3]);
    assert_eq!(generator.position(&session), 6);
    assert_eq!(grouped_calls.load(Ordering::Relaxed), 1);
    assert_eq!(rewinds.load(Ordering::Relaxed), 1);
}

#[test]
fn callback_after_first_speculative_token_rewinds_the_unobserved_token() {
    let (generator, grouped_calls, rewinds) = speculative_generator(false);
    let mut session = generator.new_session().unwrap();
    let outcome = generator
        .generate_stream(
            &mut session,
            &[0, 1, 2, 0],
            greedy(2),
            &CancellationToken::new(),
            |_| ControlFlow::Break(()),
        )
        .unwrap();

    assert_eq!(outcome.token_ids, [1]);
    assert_eq!(outcome.stop_reason, StopReason::Callback);
    assert_eq!(session.history(), [0, 1, 2, 0, 1]);
    assert_eq!(generator.position(&session), 5);
    assert_eq!(grouped_calls.load(Ordering::Relaxed), 1);
    assert_eq!(rewinds.load(Ordering::Relaxed), 1);
}
