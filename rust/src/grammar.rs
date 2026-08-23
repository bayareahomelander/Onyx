//! grammar constraint engine
//!
//! this module provides the Python bindings for grammar constraints.
//! it uses the trait-based ConstraintEngine abstraction to support
//! multiple constraint types (regex, JSON schema, etc.)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::constraint::ConstraintEngine;
use crate::json_engine::JsonEngine;
use crate::regex_engine::RegexEngine;

/// a grammar constraint engine exposed to Python
///
/// this struct provides the Python API for grammar-constrained generation.
/// internally, it holds a trait object that can be any ConstraintEngine implementation.
#[pyclass]
pub struct GrammarConstraint {
    /// the model vocabulary as byte sequences, indexed by token ID
    vocabulary: Vec<Vec<u8>>,
    /// the compiled initial engine template (None until a pattern is compiled)
    initial_engine: Option<Box<dyn ConstraintEngine>>,
    /// independent state engines keyed by opaque Python-visible handles
    states: HashMap<u32, Box<dyn ConstraintEngine>>,
    /// next opaque state handle to allocate
    next_state_id: u32,
    /// most recently returned handle, retained only for legacy snapshot methods
    current_handle: Option<u32>,
    /// saved handle for legacy speculative decoding checkpointing methods
    snapshot: Option<u32>,
    /// the original pattern/schema string for debugging
    pattern: Option<String>,
    /// the type of engine currently active
    engine_type: Option<String>,
}

impl GrammarConstraint {
    fn compiled_initial_engine(&self) -> PyResult<&Box<dyn ConstraintEngine>> {
        self.initial_engine.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "No constraint compiled. Call compile_regex or compile_json_schema first.",
            )
        })
    }

    fn state_engine(&self, state: u32) -> PyResult<&Box<dyn ConstraintEngine>> {
        self.states.get(&state).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown grammar state handle: {}", state))
        })
    }

    fn insert_state(&mut self, engine: Box<dyn ConstraintEngine>) -> PyResult<u32> {
        let state_id = self.next_state_id;
        self.next_state_id = self
            .next_state_id
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("Grammar state handle counter overflowed"))?;
        self.states.insert(state_id, engine);
        self.current_handle = Some(state_id);
        Ok(state_id)
    }

    fn install_engine(
        &mut self,
        engine: Box<dyn ConstraintEngine>,
        pattern: String,
        engine_type: String,
    ) {
        self.initial_engine = Some(engine);
        self.states.clear();
        self.next_state_id = 1;
        self.current_handle = None;
        self.snapshot = None;
        self.pattern = Some(pattern);
        self.engine_type = Some(engine_type);
    }
}

#[pymethods]
impl GrammarConstraint {
    /// create a new GrammarConstraint with the given vocab
    #[new]
    fn new(vocabulary: Vec<Vec<u8>>) -> PyResult<Self> {
        if vocabulary.is_empty() {
            return Err(PyValueError::new_err("Vocabulary cannot be empty"));
        }

        Ok(GrammarConstraint {
            vocabulary,
            initial_engine: None,
            states: HashMap::new(),
            next_state_id: 1,
            current_handle: None,
            snapshot: None,
            pattern: None,
            engine_type: None,
        })
    }

    /// compile a regex pattern into a constraint engine
    ///
    /// this creates a RegexEngine internally and stores it for use
    /// in subsequent operations
    fn compile_regex(&mut self, pattern: &str) -> PyResult<()> {
        let engine = RegexEngine::new(self.vocabulary.clone(), pattern)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        self.install_engine(Box::new(engine), pattern.to_string(), "regex".to_string());
        Ok(())
    }

    /// compile a JSON schema into a constraint engine
    ///
    /// this creates a JsonEngine internally and stores it for use
    /// in subsequent operations. the schema should be a valid JSON schema string.
    fn compile_json_schema(&mut self, schema: &str) -> PyResult<()> {
        let engine = JsonEngine::new(self.vocabulary.clone(), schema)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        self.install_engine(
            Box::new(engine),
            schema.to_string(),
            "json_schema".to_string(),
        );
        Ok(())
    }

    /// create and return a new opaque handle for an initial grammar state
    ///
    /// the returned value is a handle owned by this GrammarConstraint, not a
    /// DFA state number or stack depth.
    fn init_state(&mut self) -> PyResult<u32> {
        let engine = self.compiled_initial_engine()?.clone_box();
        self.insert_state(engine)
    }

    /// advance the supplied grammar state by consuming a token's bytes
    ///
    /// returns a new opaque state handle; the input state remains unchanged.
    fn advance_state(&mut self, state: u32, token_id: usize) -> PyResult<u32> {
        if token_id >= self.vocabulary.len() {
            return Err(PyValueError::new_err(format!(
                "Token ID {} out of range (vocab size: {})",
                token_id,
                self.vocabulary.len()
            )));
        }

        let mut engine = self.state_engine(state)?.clone_box();
        engine
            .advance(token_id)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        self.insert_state(engine)
    }

    /// get all valid token IDs that can continue from the current state
    ///
    /// the state parameter is an opaque handle returned by init_state or advance_state.
    fn get_valid_token_ids(&self, state: u32) -> PyResult<Vec<usize>> {
        Ok(self.state_engine(state)?.get_valid_tokens())
    }

    /// check if the given state is a match state (pattern fully matched)
    ///
    /// the state parameter is an opaque handle returned by init_state or advance_state.
    fn is_match_state(&self, state: u32) -> PyResult<bool> {
        Ok(self.state_engine(state)?.is_finished())
    }

    /// check if the given state is a dead state (no valid continuations)
    ///
    /// the state parameter is an opaque handle returned by init_state or advance_state.
    fn is_dead_state(&self, state: u32) -> PyResult<bool> {
        Ok(self.state_engine(state)?.is_dead())
    }

    /// clear all allocated state handles
    fn reset(&mut self) -> PyResult<()> {
        let _ = self.compiled_initial_engine()?;
        self.states.clear();
        self.next_state_id = 1;
        self.current_handle = None;
        self.snapshot = None;
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// save the most recently returned state handle for legacy checkpointing
    ///
    /// Prefer carrying explicit state handles instead of using this method.
    fn save_snapshot(&mut self) -> PyResult<()> {
        let handle = self.current_handle.ok_or_else(|| {
            PyValueError::new_err("No current state handle. Call init_state first.")
        })?;
        let _ = self.state_engine(handle)?;
        self.snapshot = Some(handle);
        Ok(())
    }

    /// restore the most recently returned state handle for legacy checkpointing
    ///
    /// Prefer carrying explicit state handles instead of using this method.
    fn restore_snapshot(&mut self) -> PyResult<u32> {
        let snap = self
            .snapshot
            .take()
            .ok_or_else(|| PyValueError::new_err("No snapshot saved. Call save_snapshot first."))?;
        let _ = self.state_engine(snap)?;
        self.current_handle = Some(snap);
        Ok(snap)
    }

    /// release a state handle that will no longer be used
    fn release_state(&mut self, state: u32) -> PyResult<()> {
        self.states.remove(&state);
        if self.current_handle == Some(state) {
            self.current_handle = None;
        }
        if self.snapshot == Some(state) {
            self.snapshot = None;
        }
        Ok(())
    }

    /// release multiple state handles that will no longer be used
    fn release_states(&mut self, states: Vec<u32>) -> PyResult<()> {
        for state in states {
            self.states.remove(&state);
            if self.current_handle == Some(state) {
                self.current_handle = None;
            }
            if self.snapshot == Some(state) {
                self.snapshot = None;
            }
        }
        Ok(())
    }

    fn get_pattern(&self) -> Option<String> {
        self.pattern.clone()
    }

    fn get_engine_type(&self) -> Option<String> {
        self.engine_type.clone()
    }

    fn __repr__(&self) -> String {
        match (&self.pattern, &self.engine_type) {
            (Some(p), Some(t)) => {
                let display = if p.len() > 50 {
                    format!("{}...", &p[..50])
                } else {
                    p.clone()
                };
                format!(
                    "GrammarConstraint(vocab_size={}, type='{}', pattern='{}')",
                    self.vocabulary.len(),
                    t,
                    display
                )
            }
            _ => format!(
                "GrammarConstraint(vocab_size={}, pattern=None)",
                self.vocabulary.len()
            ),
        }
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
        let vocab = make_test_vocab();
        let constraint = GrammarConstraint::new(vocab).unwrap();
        assert_eq!(constraint.vocab_size(), 10);
    }

    #[test]
    fn test_compile_and_init() {
        let vocab = make_test_vocab();
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();

        let state = constraint.init_state().unwrap();
        assert!(state > 0);
    }

    #[test]
    fn test_advance_state() {
        let vocab = make_test_vocab();
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();

        let state0 = constraint.init_state().unwrap();
        let state1 = constraint.advance_state(state0, 0).unwrap();

        assert_ne!(state0, state1);
        assert!(!constraint.is_dead_state(state1).unwrap());
    }

    #[test]
    fn test_valid_tokens_filtering() {
        let vocab = make_test_vocab();
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_regex("The year is [0-9]{4}").unwrap();

        let state = constraint.init_state().unwrap();
        let valid = constraint.get_valid_token_ids(state).unwrap();

        assert!(valid.contains(&0));
        assert!(!valid.contains(&8));
    }

    #[test]
    fn test_regex_state_handles_are_independent() {
        let vocab = make_test_vocab();
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_regex("The year").unwrap();

        let initial = constraint.init_state().unwrap();
        let after_the = constraint.advance_state(initial, 0).unwrap();

        let initial_valid = constraint.get_valid_token_ids(initial).unwrap();
        let after_the_valid = constraint.get_valid_token_ids(after_the).unwrap();

        assert!(initial_valid.contains(&0));
        assert!(!initial_valid.contains(&1));
        assert!(after_the_valid.contains(&1));
        assert!(!after_the_valid.contains(&0));
    }

    #[test]
    fn test_json_state_handles_are_independent() {
        let vocab = vec![
            b"{".to_vec(),     // 0
            b"\"a\"".to_vec(), // 1
            b"\"b\"".to_vec(), // 2
            b":".to_vec(),     // 3
            b"\"".to_vec(),    // 4
            b"1".to_vec(),     // 5
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
    }

    #[test]
    fn test_unknown_state_handle_errors() {
        let vocab = make_test_vocab();
        let mut constraint = GrammarConstraint::new(vocab).unwrap();
        constraint.compile_regex("The year").unwrap();

        assert!(constraint.get_valid_token_ids(999).is_err());
        assert!(constraint.advance_state(999, 0).is_err());
        assert!(constraint.is_match_state(999).is_err());
        assert!(constraint.is_dead_state(999).is_err());
    }
}
