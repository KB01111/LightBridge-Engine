use std::collections::BTreeSet;

use bridge_runtime::{Sampler, SamplingConfig, SamplingError};

#[test]
fn zero_temperature_is_greedy_with_stable_ties() {
    let config = SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let mut sampler = Sampler::new(config, 4).unwrap();
    assert_eq!(sampler.sample(&[1.0, 3.0, 3.0, 2.0], &[]).unwrap(), 1);
}

#[test]
fn seeded_sampling_is_reproducible() {
    let config = SamplingConfig {
        seed: 42,
        temperature: 0.8,
        top_k: 3,
        top_p: 0.9,
        ..SamplingConfig::default()
    };
    let mut left = Sampler::new(config.clone(), 4).unwrap();
    let mut right = Sampler::new(config, 4).unwrap();
    let left_ids = (0..16)
        .map(|_| left.sample(&[0.1, 0.2, 0.3, 0.4], &[]).unwrap())
        .collect::<Vec<_>>();
    let right_ids = (0..16)
        .map(|_| right.sample(&[0.1, 0.2, 0.3, 0.4], &[]).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(left_ids, right_ids);
}

#[test]
fn top_k_and_top_p_bound_candidates() {
    let config = SamplingConfig {
        seed: 7,
        temperature: 1.0,
        top_k: 2,
        top_p: 0.01,
        ..SamplingConfig::default()
    };
    let mut sampler = Sampler::new(config, 4).unwrap();
    for _ in 0..32 {
        assert_eq!(sampler.sample(&[4.0, 3.0, 2.0, 1.0], &[]).unwrap(), 0);
    }
}

#[test]
fn repetition_penalty_changes_greedy_selection() {
    let config = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 2.0,
        repeat_last_n: 4,
        ..SamplingConfig::default()
    };
    let mut sampler = Sampler::new(config, 3).unwrap();
    assert_eq!(sampler.sample(&[4.0, 3.0, 2.0], &[0]).unwrap(), 1);
}

#[test]
fn invalid_configuration_fails_before_sampling() {
    let config = SamplingConfig {
        top_p: 0.0,
        stop_tokens: BTreeSet::from([9]),
        ..SamplingConfig::default()
    };
    assert_eq!(
        Sampler::new(config, 4).unwrap_err(),
        SamplingError::InvalidTopP(0.0)
    );
}

#[test]
fn nan_and_all_negative_infinity_are_typed_errors() {
    let mut sampler = Sampler::new(SamplingConfig::default(), 2).unwrap();
    assert_eq!(
        sampler.sample(&[0.0, f32::NAN], &[]).unwrap_err(),
        SamplingError::NanLogit { token_id: 1 }
    );
    assert_eq!(
        sampler
            .sample(&[f32::NEG_INFINITY, f32::NEG_INFINITY], &[])
            .unwrap_err(),
        SamplingError::NoCandidate
    );
}
