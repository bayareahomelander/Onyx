//! DFA-based regex constraint engine.

use std::sync::Arc;

use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::Anchored;

use crate::constraint::{ConstraintEngine, ConstraintError};

pub struct CompiledDfa {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
}

pub fn compile_pattern_dfa(pattern: &str) -> Result<CompiledDfa, String> {
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .start_kind(regex_automata::dfa::StartKind::Anchored)
                .match_kind(regex_automata::MatchKind::LeftmostFirst),
        )
        .build(pattern)
        .map_err(|error| format!("Failed to compile regex: {error}"))?;

    let start_config = StartConfig::new().anchored(Anchored::Yes);
    let initial_state = dfa
        .start_state(&start_config)
        .map_err(|error| format!("Failed to get start state: {error}"))?;

    Ok(CompiledDfa { dfa, initial_state })
}

pub struct RegexEngine {
    vocabulary: Arc<Vec<Vec<u8>>>,
    dfa: Arc<dense::DFA<Vec<u32>>>,
    current_state: StateID,
    initial_state: StateID,
}

impl RegexEngine {
    pub fn new(vocabulary: Vec<Vec<u8>>, pattern: &str) -> Result<Self, ConstraintError> {
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::LeftmostFirst),
            )
            .build(pattern)
            .map_err(|error| {
                ConstraintError::CompilationError(format!("Failed to compile regex: {error}"))
            })?;

        let start_config = StartConfig::new().anchored(Anchored::Yes);
        let initial_state = dfa.start_state(&start_config).map_err(|error| {
            ConstraintError::CompilationError(format!("Failed to get start state: {error}"))
        })?;

        Ok(Self {
            vocabulary: Arc::new(vocabulary),
            dfa: Arc::new(dfa),
            current_state: initial_state,
            initial_state,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    fn advance_state_by_token(&self, state: StateID, token_id: usize) -> StateID {
        let mut current = state;
        for &byte in &self.vocabulary[token_id] {
            current = self.dfa.next_state(current, byte);
        }
        current
    }
}

impl ConstraintEngine for RegexEngine {
    fn reset(&mut self) {
        self.current_state = self.initial_state;
    }

    fn get_valid_tokens(&self) -> Vec<usize> {
        let mut valid_tokens = Vec::new();

        for (token_id, token_bytes) in self.vocabulary.iter().enumerate() {
            if token_bytes.is_empty() {
                continue;
            }

            let mut state = self.current_state;
            let mut is_valid = true;
            for &byte in token_bytes {
                state = self.dfa.next_state(state, byte);
                if self.dfa.is_dead_state(state) {
                    is_valid = false;
                    break;
                }
            }

            if is_valid {
                valid_tokens.push(token_id);
            }
        }

        valid_tokens
    }

    fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError> {
        if token_id >= self.vocabulary.len() {
            return Err(ConstraintError::InvalidTokenId {
                token_id,
                vocab_size: self.vocabulary.len(),
            });
        }

        self.current_state = self.advance_state_by_token(self.current_state, token_id);
        Ok(())
    }

    fn is_finished(&self) -> bool {
        let eoi_state = self.dfa.next_eoi_state(self.current_state);
        self.dfa.is_match_state(eoi_state)
    }

    fn is_dead(&self) -> bool {
        self.dfa.is_dead_state(self.current_state)
    }

    fn current_state_id(&self) -> u32 {
        self.current_state.as_u32()
    }

    fn clone_box(&self) -> Box<dyn ConstraintEngine> {
        Box::new(Self {
            vocabulary: Arc::clone(&self.vocabulary),
            dfa: Arc::clone(&self.dfa),
            current_state: self.current_state,
            initial_state: self.initial_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vec<Vec<u8>> {
        vec![
            b"The".to_vec(),
            b" year".to_vec(),
            b" is".to_vec(),
            b" ".to_vec(),
            b"2".to_vec(),
            b"0".to_vec(),
            b"1".to_vec(),
            b"9".to_vec(),
            b"hello".to_vec(),
            b"world".to_vec(),
        ]
    }

    #[test]
    fn test_regex_engine_creation() {
        let engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();
        assert_eq!(engine.vocab_size(), 10);
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_regex_engine_advance() {
        let mut engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();

        engine.advance(0).unwrap();
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());

        engine.reset();
        engine.advance(8).unwrap();
        assert!(engine.is_dead());
    }

    #[test]
    fn test_regex_engine_valid_tokens() {
        let engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();
        let valid = engine.get_valid_tokens();

        assert!(valid.contains(&0));
        assert!(!valid.contains(&8));
    }

    #[test]
    fn test_regex_engine_reset() {
        let mut engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();
        let initial_state = engine.current_state_id();

        engine.advance(0).unwrap();
        assert_ne!(engine.current_state_id(), initial_state);
        engine.reset();
        assert_eq!(engine.current_state_id(), initial_state);
    }

    #[test]
    fn test_regex_engine_full_match() {
        let mut engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();

        for token_id in 0..8 {
            engine.advance(token_id).unwrap();
        }

        assert!(engine.is_finished());
        assert!(!engine.is_dead());
    }
}
