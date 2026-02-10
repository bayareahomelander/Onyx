//! grammar constraint engine
//!
//! this module provides the Python bindings for grammar constraints.
//! it uses the trait-based ConstraintEngine abstraction to support
//! multiple constraint types (regex, JSON schema, etc.)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::constraint::ConstraintEngine;
use crate::regex_engine::RegexEngine;
use crate::json_engine::JsonEngine;

/// a grammar constraint engine exposed to Python
///
/// this struct provides the Python API for grammar-constrained generation.
/// internally, it holds a trait object that can be any ConstraintEngine implementation.
#[pyclass]
pub struct GrammarConstraint {
    /// the model vocabulary as byte sequences, indexed by token ID
    vocabulary: Vec<Vec<u8>>,
    /// the active constraint engine (None until a pattern is compiled)
    engine: Option<Box<dyn ConstraintEngine>>,
    /// saved engine snapshot for speculative decoding checkpointing
    snapshot: Option<Box<dyn ConstraintEngine>>,
    /// the original pattern/schema string for debugging
    pattern: Option<String>,
    /// the type of engine currently active
    engine_type: Option<String>,
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
            engine: None,
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

        self.engine = Some(Box::new(engine));
        self.pattern = Some(pattern.to_string());
        self.engine_type = Some("regex".to_string());
        Ok(())
    }

    /// compile a JSON schema into a constraint engine
    ///
    /// this creates a JsonEngine internally and stores it for use
    /// in subsequent operations. the schema should be a valid JSON schema string.
    fn compile_json_schema(&mut self, schema: &str) -> PyResult<()> {
        let engine = JsonEngine::new(self.vocabulary.clone(), schema)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        self.engine = Some(Box::new(engine));
        self.pattern = Some(schema.to_string());
        self.engine_type = Some("json_schema".to_string());
        Ok(())
    }

    /// get the starting state ID for DFA traversal
    ///
    /// returns the initial state ID as a u32
    fn init_state(&self) -> PyResult<u32> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        Ok(engine.current_state_id())
    }

    /// advance the DFA state by consuming a token's bytes
    ///
    /// note: this method uses a state parameter for backward compatibility,
    /// but the engine tracks state internally. the state parameter is ignored.
    fn advance_state(&mut self, _state: u32, token_id: usize) -> PyResult<u32> {
        let engine = self.engine.as_mut()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        if token_id >= self.vocabulary.len() {
            return Err(PyValueError::new_err(format!(
                "Token ID {} out of range (vocab size: {})",
                token_id,
                self.vocabulary.len()
            )));
        }

        engine.advance(token_id)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(engine.current_state_id())
    }

    /// get all valid token IDs that can continue from the current state
    ///
    /// note: the state parameter is for backward compatibility and is ignored.
    /// the engine uses its internal state.
    fn get_valid_token_ids(&self, _state: u32) -> PyResult<Vec<usize>> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        Ok(engine.get_valid_tokens())
    }

    /// check if the given state is a match state (pattern fully matched)
    ///
    /// note: the state parameter is for backward compatibility and is ignored.
    fn is_match_state(&self, _state: u32) -> PyResult<bool> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        Ok(engine.is_finished())
    }

    /// check if the given state is a dead state (no valid continuations)
    ///
    /// note: the state parameter is for backward compatibility and is ignored.
    fn is_dead_state(&self, _state: u32) -> PyResult<bool> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        Ok(engine.is_dead())
    }

    /// reset the constraint engine to its initial state
    fn reset(&mut self) -> PyResult<()> {
        let engine = self.engine.as_mut()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled. Call compile_regex or compile_json_schema first."))?;

        engine.reset();
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// save a snapshot of the current engine state for checkpointing
    ///
    /// this clones the engine so that `restore_snapshot` can revert to this point.
    /// critical for speculative decoding: call before draft speculation.
    fn save_snapshot(&mut self) -> PyResult<()> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| PyValueError::new_err("No constraint compiled."))?;

        self.snapshot = Some(engine.clone_box());
        Ok(())
    }

    /// restore the engine to the previously saved snapshot
    ///
    /// reverts the engine state to the point when `save_snapshot` was called.
    /// critical for speculative decoding: call before verification begins.
    fn restore_snapshot(&mut self) -> PyResult<()> {
        let snap = self.snapshot.take()
            .ok_or_else(|| PyValueError::new_err("No snapshot saved. Call save_snapshot first."))?;

        self.engine = Some(snap);
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
                let display = if p.len() > 50 { format!("{}...", &p[..50]) } else { p.clone() };
                format!(
                    "GrammarConstraint(vocab_size={}, type='{}', pattern='{}')",
                    self.vocabulary.len(),
                    t,
                    display
                )
            },
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
}
