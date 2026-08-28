//! Internal grammar façade and opaque state registry.

use std::collections::HashMap;

use crate::constraint::{ConstraintEngine, ConstraintError};
use crate::json_engine::JsonEngine;
use crate::regex_engine::RegexEngine;

pub struct GrammarConstraint {
    vocabulary: Vec<Vec<u8>>,
    initial_engine: Option<Box<dyn ConstraintEngine>>,
    states: HashMap<u32, Box<dyn ConstraintEngine>>,
    next_state_id: u32,
}

impl GrammarConstraint {
    fn compiled_initial_engine(&self) -> Result<&dyn ConstraintEngine, ConstraintError> {
        self.initial_engine.as_deref().ok_or_else(|| {
            ConstraintError::InvalidState(
                "No constraint compiled. Call compile_regex or compile_json_schema first.".into(),
            )
        })
    }

    fn state_engine(&self, state: u32) -> Result<&dyn ConstraintEngine, ConstraintError> {
        self.states.get(&state).map(Box::as_ref).ok_or_else(|| {
            ConstraintError::InvalidState(format!("Unknown grammar state handle: {state}"))
        })
    }

    fn insert_state(&mut self, engine: Box<dyn ConstraintEngine>) -> Result<u32, ConstraintError> {
        let state_id = self.next_state_id;
        self.next_state_id = self.next_state_id.checked_add(1).ok_or_else(|| {
            ConstraintError::InvalidState("Grammar state handle counter overflowed".into())
        })?;
        self.states.insert(state_id, engine);
        Ok(state_id)
    }

    fn install_engine(&mut self, engine: Box<dyn ConstraintEngine>) {
        self.initial_engine = Some(engine);
        self.states.clear();
        self.next_state_id = 1;
    }

    pub fn new(vocabulary: Vec<Vec<u8>>) -> Result<Self, ConstraintError> {
        if vocabulary.is_empty() {
            return Err(ConstraintError::InvalidState(
                "Vocabulary cannot be empty".into(),
            ));
        }

        Ok(Self {
            vocabulary,
            initial_engine: None,
            states: HashMap::new(),
            next_state_id: 1,
        })
    }

    pub fn compile_regex(&mut self, pattern: &str) -> Result<(), ConstraintError> {
        let engine = RegexEngine::new(self.vocabulary.clone(), pattern)?;
        self.install_engine(Box::new(engine));
        Ok(())
    }

    pub fn compile_json_schema(&mut self, schema: &str) -> Result<(), ConstraintError> {
        let engine = JsonEngine::new(self.vocabulary.clone(), schema)?;
        self.install_engine(Box::new(engine));
        Ok(())
    }

    pub fn init_state(&mut self) -> Result<u32, ConstraintError> {
        let engine = self.compiled_initial_engine()?.clone_box();
        self.insert_state(engine)
    }

    pub fn advance_state(&mut self, state: u32, token_id: usize) -> Result<u32, ConstraintError> {
        if token_id >= self.vocabulary.len() {
            return Err(ConstraintError::InvalidTokenId {
                token_id,
                vocab_size: self.vocabulary.len(),
            });
        }

        let mut engine = self.state_engine(state)?.clone_box();
        engine.advance(token_id)?;
        self.insert_state(engine)
    }

    pub fn get_valid_token_ids(&self, state: u32) -> Result<Vec<usize>, ConstraintError> {
        Ok(self.state_engine(state)?.get_valid_tokens())
    }

    pub fn is_match_state(&self, state: u32) -> Result<bool, ConstraintError> {
        Ok(self.state_engine(state)?.is_finished())
    }

    pub fn is_dead_state(&self, state: u32) -> Result<bool, ConstraintError> {
        Ok(self.state_engine(state)?.is_dead())
    }

    pub fn reset(&mut self) -> Result<(), ConstraintError> {
        let _ = self.compiled_initial_engine()?;
        self.states.clear();
        self.next_state_id = 1;
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn release_state(&mut self, state: u32) -> Result<(), ConstraintError> {
        if self.states.remove(&state).is_none() {
            return Err(ConstraintError::InvalidState(format!(
                "Unknown grammar state handle: {state}"
            )));
        }
        Ok(())
    }

    pub fn release_states(&mut self, states: Vec<u32>) -> Result<(), ConstraintError> {
        if let Some(state) = states.iter().find(|state| !self.states.contains_key(state)) {
            return Err(ConstraintError::InvalidState(format!(
                "Unknown grammar state handle: {state}"
            )));
        }
        for state in states {
            self.states.remove(&state);
        }
        Ok(())
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
    fn test_create_constraint() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        assert_eq!(constraint.vocab_size(), 10);
        assert!(GrammarConstraint::new(Vec::new()).is_err());
        assert!(constraint.init_state().is_err());
    }

    #[test]
    fn test_compile_and_init() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();
        let old_state = constraint.init_state().unwrap();
        assert_eq!(old_state, 1);

        constraint.compile_regex("hello").unwrap();
        assert!(constraint.get_valid_token_ids(old_state).is_err());
        let replacement = constraint.init_state().unwrap();
        assert_eq!(replacement, 1);

        constraint.reset().unwrap();
        assert!(constraint.get_valid_token_ids(replacement).is_err());
        assert_eq!(constraint.init_state().unwrap(), 1);
    }

    #[test]
    fn test_advance_state() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();

        let state0 = constraint.init_state().unwrap();
        let state1 = constraint.advance_state(state0, 0).unwrap();
        assert_ne!(state0, state1);
        assert!(!constraint.is_dead_state(state1).unwrap());

        let initial_valid = constraint.get_valid_token_ids(state0).unwrap();
        let advanced_valid = constraint.get_valid_token_ids(state1).unwrap();
        assert!(initial_valid.contains(&0));
        assert!(!initial_valid.contains(&1));
        assert!(advanced_valid.contains(&1));
        assert!(matches!(
            constraint.advance_state(state0, 10),
            Err(ConstraintError::InvalidTokenId {
                token_id: 10,
                vocab_size: 10
            })
        ));
    }

    #[test]
    fn test_valid_tokens_filtering() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();

        let state = constraint.init_state().unwrap();
        let valid = constraint.get_valid_token_ids(state).unwrap();
        assert!(valid.contains(&0));
        assert!(!valid.contains(&8));
    }

    #[test]
    fn test_regex_state_handles_are_independent() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        constraint.compile_regex("The year").unwrap();

        let initial = constraint.init_state().unwrap();
        let after_the = constraint.advance_state(initial, 0).unwrap();
        let initial_valid = constraint.get_valid_token_ids(initial).unwrap();
        let after_the_valid = constraint.get_valid_token_ids(after_the).unwrap();

        assert!(initial_valid.contains(&0));
        assert!(!initial_valid.contains(&1));
        assert!(after_the_valid.contains(&1));
        assert!(!after_the_valid.contains(&0));

        constraint.release_state(after_the).unwrap();
        assert!(constraint.get_valid_token_ids(after_the).is_err());
        constraint.release_states(vec![initial]).unwrap();
        assert!(constraint.get_valid_token_ids(initial).is_err());
    }

    #[test]
    fn test_json_state_handles_are_independent() {
        let vocab = vec![
            b"{".to_vec(),
            b"\"a\"".to_vec(),
            b"\"b\"".to_vec(),
            b":".to_vec(),
            b"\"".to_vec(),
            b"1".to_vec(),
        ];
        let schema =
            r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"number"}}}"#;
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_json_schema(schema).unwrap();

        let initial = constraint.init_state().unwrap();
        let in_object = constraint.advance_state(initial, 0).unwrap();
        let after_a_key = constraint.advance_state(in_object, 1).unwrap();
        let after_a_colon = constraint.advance_state(after_a_key, 3).unwrap();
        let after_b_key = constraint.advance_state(in_object, 2).unwrap();
        let after_b_colon = constraint.advance_state(after_b_key, 3).unwrap();

        let valid_for_a = constraint.get_valid_token_ids(after_a_colon).unwrap();
        let valid_for_b = constraint.get_valid_token_ids(after_b_colon).unwrap();
        assert!(valid_for_a.contains(&4));
        assert!(!valid_for_a.contains(&5));
        assert!(valid_for_b.contains(&5));
        assert!(!valid_for_b.contains(&4));

        constraint
            .release_states(vec![after_a_colon, after_b_colon])
            .unwrap();
        assert!(constraint.get_valid_token_ids(after_a_colon).is_err());
        assert!(constraint.get_valid_token_ids(after_b_colon).is_err());
    }

    #[test]
    fn test_unknown_state_handle_errors() {
        let mut constraint = GrammarConstraint::new(make_test_vocab()).unwrap();
        constraint.compile_regex("The year").unwrap();

        assert!(constraint.get_valid_token_ids(999).is_err());
        assert!(constraint.advance_state(999, 0).is_err());
        assert!(constraint.is_match_state(999).is_err());
        assert!(constraint.is_dead_state(999).is_err());
        assert!(constraint.release_state(999).is_err());
        assert!(constraint.release_states(vec![999]).is_err());
    }
}
