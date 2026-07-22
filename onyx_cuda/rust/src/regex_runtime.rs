//! Independent anchored byte-DFA runtime for Windows regex constraints.

use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::{Anchored, MatchKind};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegexStateHandle {
    owner_id: u64,
    epoch: u64,
    value: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum RegexRuntimeError {
    Compilation(String),
    InvalidInput(String),
    InvalidTokenId { token_id: usize, vocab_size: usize },
    State(String),
}

impl fmt::Display for RegexRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compilation(message) | Self::InvalidInput(message) | Self::State(message) => {
                formatter.write_str(message)
            }
            Self::InvalidTokenId {
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "grammar token ID {token_id} is outside vocabulary range [0, {vocab_size})"
            ),
        }
    }
}

pub(crate) struct RegexConstraintCore {
    vocabulary: Vec<Vec<u8>>,
    dfa: dense::DFA<Vec<u32>>,
    initial_state: StateID,
    owner_id: u64,
    epoch: u64,
    next_state_id: u64,
    states: HashMap<u64, StateID>,
}

impl RegexConstraintCore {
    pub(crate) fn compile(
        vocabulary: Vec<Vec<u8>>,
        pattern: &str,
    ) -> Result<Self, RegexRuntimeError> {
        if vocabulary.is_empty() {
            return Err(RegexRuntimeError::InvalidInput(
                "grammar vocabulary cannot be empty".to_string(),
            ));
        }
        if pattern.is_empty() {
            return Err(RegexRuntimeError::InvalidInput(
                "regex pattern cannot be empty".to_string(),
            ));
        }

        let dfa_config = dense::Config::new()
            .start_kind(StartKind::Anchored)
            .match_kind(MatchKind::LeftmostFirst);
        let dfa = dense::Builder::new()
            .configure(dfa_config)
            .build(pattern)
            .map_err(|error| {
                RegexRuntimeError::Compilation(format!("failed to compile regex pattern: {error}"))
            })?;
        let start = StartConfig::new().anchored(Anchored::Yes);
        let initial_state = dfa.start_state(&start).map_err(|error| {
            RegexRuntimeError::Compilation(format!(
                "failed to derive anchored regex start state: {error}"
            ))
        })?;
        let owner_id = NEXT_OWNER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                RegexRuntimeError::State("regex constraint owner counter overflowed".to_string())
            })?;

        Ok(Self {
            vocabulary,
            dfa,
            initial_state,
            owner_id,
            epoch: 1,
            next_state_id: 1,
            states: HashMap::new(),
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub(crate) fn init_state(&mut self) -> Result<RegexStateHandle, RegexRuntimeError> {
        self.insert_state(self.initial_state)
    }

    pub(crate) fn advance_state(
        &mut self,
        handle: RegexStateHandle,
        token_id: usize,
    ) -> Result<RegexStateHandle, RegexRuntimeError> {
        if token_id >= self.vocabulary.len() {
            return Err(RegexRuntimeError::InvalidTokenId {
                token_id,
                vocab_size: self.vocabulary.len(),
            });
        }
        let parent = self.active_state(handle)?;
        let child = self.transition(parent, &self.vocabulary[token_id]);
        self.insert_state(child)
    }

    pub(crate) fn get_valid_token_ids(
        &self,
        handle: RegexStateHandle,
    ) -> Result<Vec<usize>, RegexRuntimeError> {
        let state = self.active_state(handle)?;
        let mut valid = Vec::new();
        for (token_id, token_bytes) in self.vocabulary.iter().enumerate() {
            if token_bytes.is_empty() {
                continue;
            }
            let candidate = self.transition(state, token_bytes);
            if !self.dfa.is_dead_state(candidate) {
                valid.push(token_id);
            }
        }
        Ok(valid)
    }

    pub(crate) fn is_match_state(
        &self,
        handle: RegexStateHandle,
    ) -> Result<bool, RegexRuntimeError> {
        let state = self.active_state(handle)?;
        if self.dfa.is_dead_state(state) {
            return Ok(false);
        }
        Ok(self.dfa.is_match_state(self.dfa.next_eoi_state(state)))
    }

    pub(crate) fn is_dead_state(
        &self,
        handle: RegexStateHandle,
    ) -> Result<bool, RegexRuntimeError> {
        Ok(self.dfa.is_dead_state(self.active_state(handle)?))
    }

    pub(crate) fn release_state(
        &mut self,
        handle: RegexStateHandle,
    ) -> Result<(), RegexRuntimeError> {
        self.validate_known_handle(handle)?;
        self.states.remove(&handle.value);
        Ok(())
    }

    pub(crate) fn release_states(
        &mut self,
        handles: &[RegexStateHandle],
    ) -> Result<(), RegexRuntimeError> {
        for &handle in handles {
            self.validate_known_handle(handle)?;
        }
        for handle in handles {
            self.states.remove(&handle.value);
        }
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<(), RegexRuntimeError> {
        let next_epoch = self.epoch.checked_add(1).ok_or_else(|| {
            RegexRuntimeError::State("regex state epoch counter overflowed".to_string())
        })?;
        self.epoch = next_epoch;
        self.next_state_id = 1;
        self.states.clear();
        Ok(())
    }

    fn transition(&self, state: StateID, token_bytes: &[u8]) -> StateID {
        token_bytes
            .iter()
            .fold(state, |current, &byte| self.dfa.next_state(current, byte))
    }

    fn insert_state(&mut self, state: StateID) -> Result<RegexStateHandle, RegexRuntimeError> {
        let value = self.next_state_id;
        let next_value = value.checked_add(1).ok_or_else(|| {
            RegexRuntimeError::State("regex state handle counter overflowed".to_string())
        })?;
        self.next_state_id = next_value;
        self.states.insert(value, state);
        Ok(RegexStateHandle {
            owner_id: self.owner_id,
            epoch: self.epoch,
            value,
        })
    }

    fn active_state(&self, handle: RegexStateHandle) -> Result<StateID, RegexRuntimeError> {
        self.validate_handle_identity(handle)?;
        self.states.get(&handle.value).copied().ok_or_else(|| {
            RegexRuntimeError::State(
                "regex state handle is unknown or has been released".to_string(),
            )
        })
    }

    fn validate_known_handle(&self, handle: RegexStateHandle) -> Result<(), RegexRuntimeError> {
        self.validate_handle_identity(handle)?;
        if handle.value == 0 || handle.value >= self.next_state_id {
            return Err(RegexRuntimeError::State(
                "regex state handle is unknown".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_handle_identity(&self, handle: RegexStateHandle) -> Result<(), RegexRuntimeError> {
        if handle.owner_id != self.owner_id {
            return Err(RegexRuntimeError::State(
                "regex state belongs to another constraint".to_string(),
            ));
        }
        if handle.epoch != self.epoch {
            return Err(RegexRuntimeError::State(
                "regex state belongs to an earlier reset epoch".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn active_state_count(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        constraint: &RegexConstraintCore,
        state: RegexStateHandle,
    ) -> (Vec<usize>, bool, bool) {
        (
            constraint.get_valid_token_ids(state).unwrap(),
            constraint.is_match_state(state).unwrap(),
            constraint.is_dead_state(state).unwrap(),
        )
    }

    #[test]
    fn literal_parity_and_empty_token_behavior() {
        let vocabulary = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"ab".to_vec(),
            b"x".to_vec(),
            Vec::new(),
            b"abc".to_vec(),
            b"c".to_vec(),
        ];
        let mut constraint = RegexConstraintCore::compile(vocabulary, "ab").unwrap();
        let initial = constraint.init_state().unwrap();
        assert_eq!(
            snapshot(&constraint, initial),
            (vec![0, 2, 5], false, false)
        );

        let after_a = constraint.advance_state(initial, 0).unwrap();
        let matched = constraint.advance_state(initial, 2).unwrap();
        let dead = constraint.advance_state(initial, 3).unwrap();
        let empty = constraint.advance_state(initial, 4).unwrap();
        let exhausted = constraint.advance_state(initial, 5).unwrap();

        assert_eq!(snapshot(&constraint, after_a), (vec![1], false, false));
        assert_eq!(
            snapshot(&constraint, matched),
            (vec![0, 1, 3, 6], true, false)
        );
        assert_eq!(snapshot(&constraint, dead), (vec![], false, true));
        assert_eq!(snapshot(&constraint, empty), snapshot(&constraint, initial));
        assert_ne!(empty, initial);
        assert_eq!(snapshot(&constraint, exhausted), (vec![], false, false));
        assert_eq!(
            snapshot(&constraint, initial),
            (vec![0, 2, 5], false, false)
        );
    }

    #[test]
    fn alternation_repetition_and_character_class_match_reference() {
        let mut alternation = RegexConstraintCore::compile(
            vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"c".to_vec(),
                b"bc".to_vec(),
                b"x".to_vec(),
                Vec::new(),
            ],
            "(?:a|bc)",
        )
        .unwrap();
        let initial = alternation.init_state().unwrap();
        assert_eq!(
            snapshot(&alternation, initial),
            (vec![0, 1, 3], false, false)
        );
        let after_b = alternation.advance_state(initial, 1).unwrap();
        assert_eq!(snapshot(&alternation, after_b), (vec![2], false, false));
        let matched = alternation.advance_state(initial, 3).unwrap();
        assert_eq!(
            snapshot(&alternation, matched),
            (vec![0, 1, 2, 4], true, false)
        );

        let mut repetition = RegexConstraintCore::compile(
            vec![
                b"1".to_vec(),
                b"2".to_vec(),
                b"12".to_vec(),
                b"123".to_vec(),
                b"1234".to_vec(),
                b"a".to_vec(),
                Vec::new(),
            ],
            "[0-9]{2,3}",
        )
        .unwrap();
        let initial = repetition.init_state().unwrap();
        let two_digits = repetition.advance_state(initial, 2).unwrap();
        let three_digits = repetition.advance_state(initial, 3).unwrap();
        let four_digits = repetition.advance_state(initial, 4).unwrap();
        assert_eq!(
            snapshot(&repetition, two_digits),
            (vec![0, 1, 2, 5], true, false)
        );
        assert_eq!(
            snapshot(&repetition, three_digits),
            (vec![0, 1, 5], true, false)
        );
        assert_eq!(snapshot(&repetition, four_digits), (vec![], false, false));

        let mut class = RegexConstraintCore::compile(
            vec![
                b"A".to_vec(),
                b"B".to_vec(),
                b"C".to_vec(),
                b"D".to_vec(),
                b"AB".to_vec(),
                Vec::new(),
                b"-".to_vec(),
            ],
            "[A-C]+",
        )
        .unwrap();
        let initial = class.init_state().unwrap();
        assert_eq!(snapshot(&class, initial), (vec![0, 1, 2, 4], false, false));
        let matched = class.advance_state(initial, 4).unwrap();
        assert_eq!(
            snapshot(&class, matched),
            (vec![0, 1, 2, 3, 4, 6], true, false)
        );
    }

    #[test]
    fn utf8_bytes_can_be_split_or_consumed_in_one_token() {
        let mut constraint = RegexConstraintCore::compile(
            vec![
                vec![0xC3],
                vec![0xA9],
                vec![0xC3, 0xA9],
                b"!".to_vec(),
                Vec::new(),
                b"x".to_vec(),
            ],
            r"\x{00E9}!",
        )
        .unwrap();
        let initial = constraint.init_state().unwrap();
        assert_eq!(snapshot(&constraint, initial), (vec![0, 2], false, false));

        let split_prefix = constraint.advance_state(initial, 0).unwrap();
        assert_eq!(snapshot(&constraint, split_prefix), (vec![1], false, false));
        let split_character = constraint.advance_state(split_prefix, 1).unwrap();
        let whole_character = constraint.advance_state(initial, 2).unwrap();
        assert_eq!(
            snapshot(&constraint, split_character),
            (vec![3], false, false)
        );
        assert_eq!(
            snapshot(&constraint, whole_character),
            (vec![3], false, false)
        );
        let matched = constraint.advance_state(whole_character, 3).unwrap();
        assert_eq!(
            snapshot(&constraint, matched),
            (vec![0, 1, 3, 5], true, false)
        );
    }

    #[test]
    fn branches_preserve_their_parent_and_dead_states_propagate() {
        let mut constraint = RegexConstraintCore::compile(
            vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"c".to_vec(),
                b"ab".to_vec(),
                b"ac".to_vec(),
                b"x".to_vec(),
            ],
            "(?:ab|ac)",
        )
        .unwrap();
        let initial = constraint.init_state().unwrap();
        let after_a = constraint.advance_state(initial, 0).unwrap();
        let after_ab = constraint.advance_state(after_a, 1).unwrap();
        let after_ac = constraint.advance_state(after_a, 2).unwrap();
        let dead = constraint.advance_state(initial, 5).unwrap();
        let dead_child = constraint.advance_state(dead, 0).unwrap();

        assert_eq!(
            snapshot(&constraint, initial),
            (vec![0, 3, 4], false, false)
        );
        assert_eq!(snapshot(&constraint, after_a), (vec![1, 2], false, false));
        assert!(constraint.is_match_state(after_ab).unwrap());
        assert!(constraint.is_match_state(after_ac).unwrap());
        assert_eq!(snapshot(&constraint, dead), (vec![], false, true));
        assert_eq!(snapshot(&constraint, dead_child), (vec![], false, true));
    }

    #[test]
    fn release_reset_and_ownership_are_deterministic() {
        let mut first = RegexConstraintCore::compile(vec![b"a".to_vec()], "a").unwrap();
        let mut second = RegexConstraintCore::compile(vec![b"a".to_vec()], "a").unwrap();
        let first_state = first.init_state().unwrap();
        let second_state = second.init_state().unwrap();

        let error = first.get_valid_token_ids(second_state).unwrap_err();
        assert!(matches!(error, RegexRuntimeError::State(_)));

        first.release_state(first_state).unwrap();
        first.release_state(first_state).unwrap();
        assert!(first.get_valid_token_ids(first_state).is_err());

        let active = first.init_state().unwrap();
        let foreign = second.init_state().unwrap();
        assert!(first.release_states(&[active, foreign]).is_err());
        assert!(first.get_valid_token_ids(active).is_ok());

        first.release_states(&[active, active]).unwrap();
        first.release_states(&[active]).unwrap();
        assert!(first.get_valid_token_ids(active).is_err());

        let stale = first.init_state().unwrap();
        first.reset().unwrap();
        assert!(first.get_valid_token_ids(stale).is_err());
        assert!(first.release_state(stale).is_err());
        let fresh = first.init_state().unwrap();
        assert_eq!(fresh.value, 1);
        assert_ne!(fresh.epoch, stale.epoch);
    }

    #[test]
    fn invalid_patterns_tokens_and_counter_overflow_are_errors() {
        assert!(matches!(
            RegexConstraintCore::compile(vec![b"a".to_vec()], "("),
            Err(RegexRuntimeError::Compilation(_))
        ));
        assert!(matches!(
            RegexConstraintCore::compile(Vec::new(), "a"),
            Err(RegexRuntimeError::InvalidInput(_))
        ));

        let mut constraint = RegexConstraintCore::compile(vec![b"a".to_vec()], "a").unwrap();
        let state = constraint.init_state().unwrap();
        assert!(matches!(
            constraint.advance_state(state, 1),
            Err(RegexRuntimeError::InvalidTokenId { .. })
        ));

        constraint.next_state_id = u64::MAX;
        assert!(matches!(
            constraint.init_state(),
            Err(RegexRuntimeError::State(_))
        ));
        constraint.next_state_id = 2;
        constraint.epoch = u64::MAX;
        let current_epoch_state = RegexStateHandle {
            owner_id: constraint.owner_id,
            epoch: u64::MAX,
            value: state.value,
        };
        assert!(constraint.get_valid_token_ids(current_epoch_state).is_ok());
        assert!(matches!(
            constraint.reset(),
            Err(RegexRuntimeError::State(_))
        ));
        assert!(constraint.get_valid_token_ids(current_epoch_state).is_ok());
    }

    #[test]
    fn repeated_lifecycle_returns_to_zero_active_states() {
        let mut constraint =
            RegexConstraintCore::compile(vec![b"a".to_vec(), b"b".to_vec()], "ab").unwrap();
        for iteration in 0..1_000 {
            let initial = constraint.init_state().unwrap();
            let child = constraint.advance_state(initial, 0).unwrap();
            constraint.release_states(&[initial, child]).unwrap();
            assert_eq!(constraint.active_state_count(), 0, "iteration {iteration}");
            constraint.reset().unwrap();
            assert_eq!(constraint.active_state_count(), 0, "iteration {iteration}");
        }
    }
}
