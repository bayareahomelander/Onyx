//! regex constraint engine
//!
//! this module implements the DFA-based regex constraint engine.
//! it uses regex-automata to compile regex patterns into deterministic
//! finite automata for efficient vocabulary filtering during generation.

use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::Anchored;

use crate::constraint::{ConstraintEngine, ConstraintError};

/// compiled DFA with initial state for regex pattern matching
pub struct CompiledDfa {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
}

/// compile a regex pattern into a DFA
/// returns the compiled DFA and initial state
pub fn compile_pattern_dfa(pattern: &str) -> Result<CompiledDfa, String> {
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .start_kind(regex_automata::dfa::StartKind::Anchored)
                .match_kind(regex_automata::MatchKind::LeftmostFirst)
        )
        .build(pattern)
        .map_err(|e| format!("Failed to compile regex: {}", e))?;

    let start_config = StartConfig::new().anchored(Anchored::Yes);
    let initial_state = dfa.start_state(&start_config)
        .map_err(|e| format!("Failed to get start state: {}", e))?;

    Ok(CompiledDfa { dfa, initial_state })
}

/// a regex constraint engine using DFA traversal
///
/// this struct holds a compiled DFA and tracks the current state
/// during token generation. it implements the ConstraintEngine trait
/// to provide vocabulary filtering based on regex patterns.
pub struct RegexEngine {
    vocabulary: Vec<Vec<u8>>,
    dfa: dense::DFA<Vec<u32>>,
    current_state: StateID,
    initial_state: StateID,
    pattern: String,
}

impl RegexEngine {
    /// create a new RegexEngine with the given vocabulary and pattern
    ///
    /// compiles the regex pattern into a DFA and initializes the state
    pub fn new(vocabulary: Vec<Vec<u8>>, pattern: &str) -> Result<Self, ConstraintError> {
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::LeftmostFirst)
            )
            .build(pattern)
            .map_err(|e| ConstraintError::CompilationError(format!("Failed to compile regex: {}", e)))?;

        let start_config = StartConfig::new().anchored(Anchored::Yes);
        let initial_state = dfa.start_state(&start_config)
            .map_err(|e| ConstraintError::CompilationError(format!("Failed to get start state: {}", e)))?;

        Ok(RegexEngine {
            vocabulary,
            dfa,
            current_state: initial_state,
            initial_state,
            pattern: pattern.to_string(),
        })
    }

    /// get the pattern string
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// advance the DFA state by consuming a token's bytes (internal method)
    fn advance_state_by_token(&self, state: StateID, token_id: usize) -> StateID {
        let token_bytes = &self.vocabulary[token_id];
        let mut current = state;

        for &byte in token_bytes {
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

            let mut test_state = self.current_state;
            let mut is_valid = true;

            for &byte in token_bytes {
                test_state = self.dfa.next_state(test_state, byte);

                if self.dfa.is_dead_state(test_state) {
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
        let vocab = make_test_vocab();
        let engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        assert_eq!(engine.vocab_size(), 10);
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_regex_engine_advance() {
        let vocab = make_test_vocab();
        let mut engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        
        // advance with "The" (token 0)
        engine.advance(0).unwrap();
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_regex_engine_valid_tokens() {
        let vocab = make_test_vocab();
        let engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        
        let valid = engine.get_valid_tokens();
        
        // "The" (token 0) should be valid at start
        assert!(valid.contains(&0));
        // "hello" (token 8) should NOT be valid
        assert!(!valid.contains(&8));
    }

    #[test]
    fn test_regex_engine_reset() {
        let vocab = make_test_vocab();
        let mut engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        
        let initial_state = engine.current_state_id();
        
        engine.advance(0).unwrap();
        assert_ne!(engine.current_state_id(), initial_state);
        
        engine.reset();
        assert_eq!(engine.current_state_id(), initial_state);
    }

    #[test]
    fn test_regex_engine_full_match() {
        let vocab = make_test_vocab();
        let mut engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        
        // generate "The year is 2019"
        engine.advance(0).unwrap();  // "The"
        engine.advance(1).unwrap();  // " year"
        engine.advance(2).unwrap();  // " is"
        engine.advance(3).unwrap();  // " "
        engine.advance(4).unwrap();  // "2"
        engine.advance(5).unwrap();  // "0"
        engine.advance(6).unwrap();  // "1"
        engine.advance(7).unwrap();  // "9"
        
        assert!(engine.is_finished());
        assert!(!engine.is_dead());
    }
}
