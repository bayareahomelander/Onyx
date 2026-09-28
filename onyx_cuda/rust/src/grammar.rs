//! Internal grammar façade, opaque state registry, and valid-token scan cache.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use crate::constraint::{ConstraintEngine, ConstraintError, ScanKey};
use crate::json_engine::JsonEngine;
use crate::regex_engine::RegexEngine;

/// Scans kept per compiled grammar. A document revisits few distinct states
/// (13-18 in the measured long-text JSON cases); an entry is a vocabulary bitset.
const DEFAULT_SCAN_CACHE_ENTRIES: usize = 256;

/// One vocabulary scan. `dependent` lists tokens whose validity also depends on
/// state outside the scan key, with their validity when scanned.
struct ScanEntry {
    id: u64,
    bits: Vec<u64>,
    count: usize,
    dependent: Vec<(usize, bool)>,
}

impl ScanEntry {
    fn contains(&self, token_id: usize) -> bool {
        self.bits[token_id / 64] & (1 << (token_id % 64)) != 0
    }
}

/// A cached entry plus rechecked tokens that now differ from it. The id names
/// the exact token set, so callers may cache masks derived from it.
struct Scan {
    id: u64,
    entry: Arc<ScanEntry>,
    changed: Vec<(usize, bool)>,
}

impl Scan {
    fn is_valid(&self, token_id: usize) -> bool {
        match self.changed.iter().find(|(token, _)| *token == token_id) {
            Some(&(_, valid)) => valid,
            None => self.entry.contains(token_id),
        }
    }

    fn count(&self) -> usize {
        let added = self.changed.iter().filter(|(_, valid)| *valid).count();
        self.entry.count + added - (self.changed.len() - added)
    }

    fn token_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.entry.count);
        for (index, &word) in self.entry.bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                ids.push(index * 64 + word.trailing_zeros() as usize);
                word &= word - 1;
            }
        }
        if !self.changed.is_empty() {
            ids.retain(|&token| self.is_valid(token));
            ids.extend(self.changed.iter().filter(|(_, valid)| *valid).map(|(token, _)| *token));
            ids.sort_unstable();
        }
        ids
    }

    /// One byte per token: 1 where the token is not selectable.
    fn blocked_mask(&self, vocab_size: usize) -> Vec<u8> {
        let mut mask: Vec<u8> = (0..vocab_size)
            .map(|token| u8::from(!self.entry.contains(token)))
            .collect();
        for &(token, valid) in &self.changed {
            mask[token] = u8::from(!valid);
        }
        mask
    }
}

#[pyclass(module = "onyx_cuda._rust")]
pub struct GrammarConstraint {
    vocabulary: Vec<Vec<u8>>,
    initial_engine: Option<Box<dyn ConstraintEngine>>,
    states: HashMap<u32, Box<dyn ConstraintEngine>>,
    next_state_id: u32,
    scan_capacity: usize,
    scans: HashMap<ScanKey, Arc<ScanEntry>>,
    scan_order: VecDeque<ScanKey>,
    next_scan_id: u64,
}

fn constraint_error_to_value_error(error: ConstraintError) -> PyErr {
    PyValueError::new_err(error.to_string())
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
        self.scans.clear();
        self.scan_order.clear();
    }

    pub fn new(vocabulary: Vec<Vec<u8>>) -> Result<Self, ConstraintError> {
        Self::with_scan_cache(vocabulary, DEFAULT_SCAN_CACHE_ENTRIES)
    }

    /// A scan capacity of 0 disables the cache; every scan is then fresh.
    pub fn with_scan_cache(
        vocabulary: Vec<Vec<u8>>,
        scan_capacity: usize,
    ) -> Result<Self, ConstraintError> {
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
            scan_capacity,
            scans: HashMap::new(),
            scan_order: VecDeque::new(),
            next_scan_id: 1,
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

    /// Scan through the cache; a hit rechecks only its output-dependent tokens.
    fn scan(&mut self, state: u32) -> Result<Scan, ConstraintError> {
        let engine = self.states.get(&state).map(Box::as_ref).ok_or_else(|| {
            ConstraintError::InvalidState(format!("Unknown grammar state handle: {state}"))
        })?;
        let key = if self.scan_capacity > 0 { engine.scan_key() } else { None };
        if let Some(entry) = key.as_ref().and_then(|key| self.scans.get(key)) {
            let entry = Arc::clone(entry);
            let changed: Vec<(usize, bool)> = entry
                .dependent
                .iter()
                .filter_map(|&(token, valid)| {
                    let now = engine.is_valid_token(token);
                    (now != valid).then_some((token, now))
                })
                .collect();
            let id = if changed.is_empty() {
                entry.id
            } else {
                self.next_scan_id += 1;
                self.next_scan_id - 1
            };
            return Ok(Scan { id, entry, changed });
        }

        let (valid, dependent) = engine.scan();
        let mut bits = vec![0u64; self.vocabulary.len().div_ceil(64)];
        for &token in &valid {
            bits[token / 64] |= 1 << (token % 64);
        }
        let mut entry = ScanEntry {
            id: self.next_scan_id,
            bits,
            count: valid.len(),
            dependent: Vec::new(),
        };
        entry.dependent = dependent
            .into_iter()
            .map(|token| (token, entry.contains(token)))
            .collect();
        self.next_scan_id += 1;
        let entry = Arc::new(entry);
        if let Some(key) = key {
            if self.scans.len() >= self.scan_capacity {
                if let Some(oldest) = self.scan_order.pop_front() {
                    self.scans.remove(&oldest);
                }
            }
            self.scans.insert(key.clone(), Arc::clone(&entry));
            self.scan_order.push_back(key);
        }
        Ok(Scan { id: entry.id, entry, changed: Vec::new() })
    }

    pub fn get_valid_token_ids(&mut self, state: u32) -> Result<Vec<usize>, ConstraintError> {
        Ok(self.scan(state)?.token_ids())
    }

    /// An id naming the exact valid-token set, and its size.
    pub fn scan_valid_tokens(&mut self, state: u32) -> Result<(u64, usize), ConstraintError> {
        let scan = self.scan(state)?;
        Ok((scan.id, scan.count()))
    }

    pub fn blocked_token_mask(&mut self, state: u32) -> Result<Vec<u8>, ConstraintError> {
        let vocab_size = self.vocabulary.len();
        Ok(self.scan(state)?.blocked_mask(vocab_size))
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

#[pymethods]
impl GrammarConstraint {
    #[new]
    #[pyo3(signature = (vocabulary, scan_cache_entries = DEFAULT_SCAN_CACHE_ENTRIES))]
    fn py_new(vocabulary: Vec<Vec<u8>>, scan_cache_entries: usize) -> PyResult<Self> {
        Self::with_scan_cache(vocabulary, scan_cache_entries)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "compile_regex")]
    fn py_compile_regex(&mut self, pattern: &str) -> PyResult<()> {
        self.compile_regex(pattern)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "compile_json_schema")]
    fn py_compile_json_schema(&mut self, schema: &str) -> PyResult<()> {
        self.compile_json_schema(schema)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "init_state")]
    fn py_init_state(&mut self) -> PyResult<u32> {
        self.init_state().map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "advance_state")]
    fn py_advance_state(&mut self, state: u32, token_id: usize) -> PyResult<u32> {
        self.advance_state(state, token_id)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "get_valid_token_ids")]
    fn py_get_valid_token_ids(&mut self, state: u32) -> PyResult<Vec<usize>> {
        self.get_valid_token_ids(state)
            .map_err(constraint_error_to_value_error)
    }

    /// (id, count) of the state's valid tokens; equal ids mean equal sets.
    #[pyo3(name = "scan_valid_tokens")]
    fn py_scan_valid_tokens(&mut self, state: u32) -> PyResult<(u64, usize)> {
        self.scan_valid_tokens(state)
            .map_err(constraint_error_to_value_error)
    }

    /// One byte per vocabulary token, 1 where the token is not selectable.
    #[pyo3(name = "blocked_token_mask")]
    fn py_blocked_token_mask<'py>(
        &mut self,
        py: Python<'py>,
        state: u32,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let mask = self
            .blocked_token_mask(state)
            .map_err(constraint_error_to_value_error)?;
        Ok(PyByteArray::new_bound(py, &mask))
    }

    #[pyo3(name = "is_match_state")]
    fn py_is_match_state(&self, state: u32) -> PyResult<bool> {
        self.is_match_state(state)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "is_dead_state")]
    fn py_is_dead_state(&self, state: u32) -> PyResult<bool> {
        self.is_dead_state(state)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "reset")]
    fn py_reset(&mut self) -> PyResult<()> {
        self.reset().map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "vocab_size")]
    fn py_vocab_size(&self) -> usize {
        self.vocab_size()
    }

    #[pyo3(name = "release_state")]
    fn py_release_state(&mut self, state: u32) -> PyResult<()> {
        self.release_state(state)
            .map_err(constraint_error_to_value_error)
    }

    #[pyo3(name = "release_states")]
    fn py_release_states(&mut self, states: Vec<u32>) -> PyResult<()> {
        self.release_states(states)
            .map_err(constraint_error_to_value_error)
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

    fn byte_vocabulary() -> Vec<Vec<u8>> {
        let mut vocabulary: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
        vocabulary.extend([
            b"\"}".to_vec(),
            b"\"},".to_vec(),
            b"\",\"".to_vec(),
            b"\"} \n".to_vec(),
            b"ab".to_vec(),
            b" c".to_vec(),
            "é".as_bytes().to_vec(),
            b"\\n".to_vec(),
            b"\\u00".to_vec(),
            b"12".to_vec(),
            b"]}".to_vec(),
            b"}]".to_vec(),
            Vec::new(),
        ]);
        vocabulary
    }

    fn compiled(scan_capacity: usize, grammar: &str, json: bool) -> (GrammarConstraint, u32) {
        let mut constraint =
            GrammarConstraint::with_scan_cache(byte_vocabulary(), scan_capacity).unwrap();
        if json {
            constraint.compile_json_schema(grammar).unwrap();
        } else {
            constraint.compile_regex(grammar).unwrap();
        }
        let state = constraint.init_state().unwrap();
        (constraint, state)
    }

    #[test]
    fn cached_scans_match_fresh_scans_at_every_byte() {
        let cases = [
            (
                r#"{"type":"object","properties":{"title":{"type":"string"},"summary":{"type":"string"}},"required":["title","summary"],"additionalProperties":false}"#,
                r#"{"title":"ab ab","summary":"a \"q\" é é\n done"}"#,
                true,
            ),
            (
                r#"{"type":"array","items":{"type":"object","properties":{"name":{"type":"string","minLength":2},"note":{"type":"string","maxLength":4}},"required":["name"]},"minItems":2,"maxItems":3}"#,
                r#"[{"name":"aaa","note":"bcd"},{"name":"xy"}]"#,
                true,
            ),
            (r#"{"type":"string","pattern":"^a+b?$"}"#, r#""aaab""#, true),
            (r#"{"enum":["red","green",12]}"#, "12", true),
            (r#"{"type":"array","items":{"type":"number"}}"#, "[1,-2.5,30]", true),
            ("(ab|a)+c?", "abaabc", false),
        ];
        for (grammar, document, json) in cases {
            let (mut fresh, mut fresh_state) = compiled(0, grammar, json);
            let (mut cached, mut cached_state) = compiled(256, grammar, json);
            // One entry forces an eviction whenever two states alternate.
            let (mut tiny, mut tiny_state) = compiled(1, grammar, json);
            let tiny_initial = tiny.init_state().unwrap();
            for byte in document.bytes().map(Some).chain(std::iter::once(None)) {
                let expected = fresh.get_valid_token_ids(fresh_state).unwrap();
                for _ in 0..2 {
                    assert_eq!(cached.get_valid_token_ids(cached_state).unwrap(), expected, "{grammar}");
                    let (_, count) = cached.scan_valid_tokens(cached_state).unwrap();
                    assert_eq!(count, expected.len());
                    let mask = cached.blocked_token_mask(cached_state).unwrap();
                    let unblocked: Vec<usize> = (0..mask.len()).filter(|&t| mask[t] == 0).collect();
                    assert_eq!(unblocked, expected);
                    tiny.get_valid_token_ids(tiny_initial).unwrap();
                    assert_eq!(tiny.get_valid_token_ids(tiny_state).unwrap(), expected);
                }
                if let Some(byte) = byte {
                    let token = byte as usize;
                    fresh_state = fresh.advance_state(fresh_state, token).unwrap();
                    cached_state = cached.advance_state(cached_state, token).unwrap();
                    tiny_state = tiny.advance_state(tiny_state, token).unwrap();
                }
            }
        }
    }

    #[test]
    fn plain_string_positions_share_one_scan() {
        let schema = r#"{"type":"object","properties":{"text":{"type":"string"}}}"#;
        let (mut constraint, mut state) = compiled(256, schema, true);
        let mut ids = Vec::new();
        for (index, byte) in br#"{"text":"abcab"#.iter().enumerate() {
            state = constraint.advance_state(state, *byte as usize).unwrap();
            if index >= 8 {
                ids.push(constraint.scan_valid_tokens(state).unwrap().0);
            }
        }
        // The closing `"}` completes the document at every position; its
        // recheck agrees, so every position reuses the first scan.
        assert_eq!(ids.len(), 6);
        assert!(ids.iter().all(|id| *id == ids[0]), "{ids:?}");
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
