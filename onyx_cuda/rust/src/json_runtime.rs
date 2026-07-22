//! Transactional state-handle runtime for the independent Windows JSON core.

use crate::json_parser::JsonParserState;
use crate::json_schema::{compile_json_schema, CompiledSchema};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_JSON_OWNER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JsonStateHandle {
    owner_id: u64,
    epoch: u64,
    value: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum JsonRuntimeError {
    Compilation(String),
    InvalidInput(String),
    InvalidTokenId {
        token_id: usize,
        vocab_size: usize,
    },
    InvalidTransition {
        token_id: usize,
        byte_offset: usize,
        message: String,
    },
    State(String),
}

impl fmt::Display for JsonRuntimeError {
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
            Self::InvalidTransition {
                token_id,
                byte_offset,
                message,
            } => write!(
                formatter,
                "JSON token ID {token_id} is invalid at byte offset {byte_offset}: {message}"
            ),
        }
    }
}

pub(crate) struct JsonConstraintCore {
    vocabulary: Vec<Vec<u8>>,
    schema: Arc<CompiledSchema>,
    owner_id: u64,
    epoch: u64,
    next_state_id: u64,
    states: HashMap<u64, JsonParserState>,
}

impl JsonConstraintCore {
    pub(crate) fn compile(
        vocabulary: Vec<Vec<u8>>,
        schema_source: &str,
    ) -> Result<Self, JsonRuntimeError> {
        if vocabulary.is_empty() {
            return Err(JsonRuntimeError::InvalidInput(
                "grammar vocabulary cannot be empty".to_string(),
            ));
        }

        let schema = compile_json_schema(schema_source).map_err(|error| {
            JsonRuntimeError::Compilation(format!("failed to compile JSON schema: {error}"))
        })?;
        let owner_id = NEXT_JSON_OWNER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                JsonRuntimeError::State("JSON constraint owner counter overflowed".to_string())
            })?;

        Ok(Self {
            vocabulary,
            schema,
            owner_id,
            epoch: 1,
            next_state_id: 1,
            states: HashMap::new(),
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub(crate) fn init_state(&mut self) -> Result<JsonStateHandle, JsonRuntimeError> {
        self.insert_state(JsonParserState::new())
    }

    pub(crate) fn advance_state(
        &mut self,
        handle: JsonStateHandle,
        token_id: usize,
    ) -> Result<JsonStateHandle, JsonRuntimeError> {
        if token_id >= self.vocabulary.len() {
            return Err(JsonRuntimeError::InvalidTokenId {
                token_id,
                vocab_size: self.vocabulary.len(),
            });
        }

        let mut child = self.active_state(handle)?.clone();
        for (byte_offset, &byte) in self.vocabulary[token_id].iter().enumerate() {
            child
                .consume_token(&self.schema, &[byte])
                .map_err(|error| JsonRuntimeError::InvalidTransition {
                    token_id,
                    byte_offset,
                    message: error.to_string(),
                })?;
        }
        self.insert_state(child)
    }

    pub(crate) fn get_valid_token_ids(
        &self,
        handle: JsonStateHandle,
    ) -> Result<Vec<usize>, JsonRuntimeError> {
        let parent = self.active_state(handle)?;
        let mut valid = Vec::new();
        for (token_id, token_bytes) in self.vocabulary.iter().enumerate() {
            if token_bytes.is_empty() {
                continue;
            }
            let mut candidate = parent.clone();
            if candidate.consume_token(&self.schema, token_bytes).is_ok() {
                valid.push(token_id);
            }
        }
        Ok(valid)
    }

    pub(crate) fn is_match_state(&self, handle: JsonStateHandle) -> Result<bool, JsonRuntimeError> {
        Ok(self.active_state(handle)?.is_match(&self.schema))
    }

    pub(crate) fn is_dead_state(&self, handle: JsonStateHandle) -> Result<bool, JsonRuntimeError> {
        Ok(self.active_state(handle)?.is_dead())
    }

    pub(crate) fn release_state(
        &mut self,
        handle: JsonStateHandle,
    ) -> Result<(), JsonRuntimeError> {
        self.validate_known_handle(handle)?;
        self.states.remove(&handle.value);
        Ok(())
    }

    pub(crate) fn release_states(
        &mut self,
        handles: &[JsonStateHandle],
    ) -> Result<(), JsonRuntimeError> {
        for &handle in handles {
            self.validate_known_handle(handle)?;
        }
        for handle in handles {
            self.states.remove(&handle.value);
        }
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<(), JsonRuntimeError> {
        let next_epoch = self.epoch.checked_add(1).ok_or_else(|| {
            JsonRuntimeError::State("JSON state epoch counter overflowed".to_string())
        })?;
        self.epoch = next_epoch;
        self.next_state_id = 1;
        self.states.clear();
        Ok(())
    }

    fn insert_state(
        &mut self,
        state: JsonParserState,
    ) -> Result<JsonStateHandle, JsonRuntimeError> {
        let value = self.next_state_id;
        let next_value = value.checked_add(1).ok_or_else(|| {
            JsonRuntimeError::State("JSON state handle counter overflowed".to_string())
        })?;
        self.next_state_id = next_value;
        self.states.insert(value, state);
        Ok(JsonStateHandle {
            owner_id: self.owner_id,
            epoch: self.epoch,
            value,
        })
    }

    fn active_state(&self, handle: JsonStateHandle) -> Result<&JsonParserState, JsonRuntimeError> {
        self.validate_handle_identity(handle)?;
        self.states.get(&handle.value).ok_or_else(|| {
            JsonRuntimeError::State("JSON state handle is unknown or has been released".to_string())
        })
    }

    fn validate_known_handle(&self, handle: JsonStateHandle) -> Result<(), JsonRuntimeError> {
        self.validate_handle_identity(handle)?;
        if handle.value == 0 || handle.value >= self.next_state_id {
            return Err(JsonRuntimeError::State(
                "JSON state handle is unknown".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_handle_identity(&self, handle: JsonStateHandle) -> Result<(), JsonRuntimeError> {
        if handle.owner_id != self.owner_id {
            return Err(JsonRuntimeError::State(
                "JSON state belongs to another constraint".to_string(),
            ));
        }
        if handle.epoch != self.epoch {
            return Err(JsonRuntimeError::State(
                "JSON state belongs to an earlier reset epoch".to_string(),
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

    fn core(schema: &str, vocabulary: &[&[u8]]) -> JsonConstraintCore {
        JsonConstraintCore::compile(
            vocabulary.iter().map(|token| token.to_vec()).collect(),
            schema,
        )
        .unwrap()
    }

    fn advance_bytes(
        core: &mut JsonConstraintCore,
        mut state: JsonStateHandle,
        token_ids: &[usize],
    ) -> JsonStateHandle {
        for &token_id in token_ids {
            state = core.advance_state(state, token_id).unwrap();
        }
        state
    }

    fn assert_snapshot(
        core: &mut JsonConstraintCore,
        token_ids: &[usize],
        valid_token_ids: &[usize],
        is_match: bool,
    ) {
        let root = core.init_state().unwrap();
        let state = advance_bytes(core, root, token_ids);
        assert_eq!(
            core.get_valid_token_ids(state).unwrap(),
            valid_token_ids,
            "valid-token mismatch after {token_ids:?}"
        );
        assert_eq!(
            core.is_match_state(state).unwrap(),
            is_match,
            "match mismatch after {token_ids:?}"
        );
        assert!(!core.is_dead_state(state).unwrap());
    }

    fn assert_rejected_transition(
        core: &mut JsonConstraintCore,
        prefix_token_ids: &[usize],
        token_id: usize,
    ) {
        let root = core.init_state().unwrap();
        let parent = advance_bytes(core, root, prefix_token_ids);
        let valid_before = core.get_valid_token_ids(parent).unwrap();
        let match_before = core.is_match_state(parent).unwrap();
        let dead_before = core.is_dead_state(parent).unwrap();
        let active_before = core.active_state_count();
        assert!(!valid_before.contains(&token_id));

        assert!(matches!(
            core.advance_state(parent, token_id),
            Err(JsonRuntimeError::InvalidTransition { .. })
        ));
        assert_eq!(core.active_state_count(), active_before);
        assert_eq!(core.get_valid_token_ids(parent).unwrap(), valid_before);
        assert_eq!(core.is_match_state(parent).unwrap(), match_before);
        assert_eq!(core.is_dead_state(parent).unwrap(), dead_before);
    }

    #[test]
    fn branches_are_independent_and_invalid_transitions_are_error_only() {
        let mut core = core(
            r#"{"type":"boolean"}"#,
            &[b"", b"t", b"rue", b"f", b"alse", b"x"],
        );
        let root = core.init_state().unwrap();
        let true_prefix = core.advance_state(root, 1).unwrap();
        let false_prefix = core.advance_state(root, 3).unwrap();
        assert_eq!(core.active_state_count(), 3);

        let error = core.advance_state(true_prefix, 5).unwrap_err();
        assert!(matches!(
            error,
            JsonRuntimeError::InvalidTransition {
                token_id: 5,
                byte_offset: 0,
                ..
            }
        ));
        assert_eq!(core.active_state_count(), 3);

        let true_state = core.advance_state(true_prefix, 2).unwrap();
        let false_state = core.advance_state(false_prefix, 4).unwrap();
        assert!(core.is_match_state(true_state).unwrap());
        assert!(core.is_match_state(false_state).unwrap());
        assert!(!core.is_match_state(root).unwrap());
    }

    #[test]
    fn empty_tokens_clone_state_but_are_not_advertised() {
        let mut core = core(r#"{"type":"null"}"#, &[b"", b"null", b"n", b"x"]);
        let root = core.init_state().unwrap();
        assert_eq!(core.get_valid_token_ids(root).unwrap(), vec![1, 2]);

        let clone = core.advance_state(root, 0).unwrap();
        assert_ne!(root.value, clone.value);
        assert_eq!(core.get_valid_token_ids(clone).unwrap(), vec![1, 2]);
        assert_eq!(core.is_match_state(root), core.is_match_state(clone));
        assert_eq!(core.is_dead_state(root), core.is_dead_state(clone));
    }

    #[test]
    fn valid_token_filtering_is_sorted_and_does_not_allocate_states() {
        let mut core = core(
            r#"{"type":"string","minLength":1}"#,
            &[b"z", b"\"", b"\"a\"", b"", b"x", b"\"b"],
        );
        let root = core.init_state().unwrap();
        let before = core.active_state_count();

        assert_eq!(core.get_valid_token_ids(root).unwrap(), vec![1, 2, 5]);
        assert_eq!(core.get_valid_token_ids(root).unwrap(), vec![1, 2, 5]);
        assert_eq!(core.active_state_count(), before);
    }

    #[test]
    fn match_states_can_accept_continuations_and_active_states_are_not_dead() {
        let mut core = core(
            r#"{"type":"number"}"#,
            &[b"1", b"2", b"e", b"+", b"0", b" "],
        );
        let root = core.init_state().unwrap();
        let one = core.advance_state(root, 0).unwrap();
        assert!(core.is_match_state(one).unwrap());
        assert!(!core.is_dead_state(one).unwrap());

        let twelve = core.advance_state(one, 1).unwrap();
        assert!(core.is_match_state(twelve).unwrap());
        let exponent = advance_bytes(&mut core, twelve, &[2, 3, 4]);
        assert!(core.is_match_state(exponent).unwrap());
        let complete = core.advance_state(exponent, 5).unwrap();
        assert!(core.is_match_state(complete).unwrap());
        assert!(core.advance_state(complete, 1).is_err());
    }

    #[test]
    fn handles_are_owned_releasable_and_invalidated_by_reset() {
        let vocabulary: &[&[u8]] = &[b"null"];
        let mut left = core(r#"{"type":"null"}"#, vocabulary);
        let mut right = core(r#"{"type":"null"}"#, vocabulary);
        let first = left.init_state().unwrap();
        let second = left.advance_state(first, 0).unwrap();
        let foreign = right.init_state().unwrap();

        assert!(left.is_match_state(foreign).is_err());
        left.release_state(second).unwrap();
        left.release_state(second).unwrap();
        assert!(left.is_match_state(second).is_err());

        let before_atomic_release = left.active_state_count();
        assert!(left.release_states(&[first, foreign]).is_err());
        assert_eq!(left.active_state_count(), before_atomic_release);
        assert!(left.is_match_state(first).is_ok());

        left.reset().unwrap();
        assert_eq!(left.active_state_count(), 0);
        assert!(left.is_match_state(first).is_err());
        let after_reset = left.init_state().unwrap();
        assert_eq!(after_reset.value, 1);
        assert_ne!(after_reset.epoch, first.epoch);
    }

    #[test]
    fn invalid_token_ids_and_counter_overflow_are_reported_without_mutation() {
        let mut core = core(r#"{"type":"null"}"#, &[b"null"]);
        let root = core.init_state().unwrap();
        assert_eq!(
            core.advance_state(root, 1).unwrap_err(),
            JsonRuntimeError::InvalidTokenId {
                token_id: 1,
                vocab_size: 1,
            }
        );
        assert_eq!(core.active_state_count(), 1);

        core.next_state_id = u64::MAX;
        let error = core.advance_state(root, 0).unwrap_err();
        assert_eq!(
            error,
            JsonRuntimeError::State("JSON state handle counter overflowed".to_string())
        );
        assert_eq!(core.active_state_count(), 1);

        core.next_state_id = 2;
        core.epoch = u64::MAX;
        assert_eq!(
            core.reset().unwrap_err(),
            JsonRuntimeError::State("JSON state epoch counter overflowed".to_string())
        );
        assert_eq!(core.active_state_count(), 1);
    }

    #[test]
    fn repeated_branch_release_cycles_do_not_accumulate_states() {
        let mut core = core(
            r#"{"type":["boolean","null"]}"#,
            &[b"true", b"false", b"null"],
        );
        let root = core.init_state().unwrap();

        for cycle in 0..1_000 {
            let token_id = cycle % core.vocab_size();
            let child = core.advance_state(root, token_id).unwrap();
            assert!(core.is_match_state(child).unwrap());
            core.release_state(child).unwrap();
            assert_eq!(core.active_state_count(), 1);
        }
    }

    // These cases mirror the ten D19 `windows_parity` fixtures. Expected token sets apply the
    // selected Windows corrections (strict lexical transitions and trailing whitespace) instead
    // of reproducing the read-only root runtime's recorded over-approximations.

    #[test]
    fn d19_root_number_fraction_and_exponent_snapshots() {
        let mut core = core(
            r#"{"type":"number"}"#,
            &[
                b"-", b"1", b".", b"5", b"e", b"E", b"+", b"2", b" ", b"", b"-1.5e+2",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 1, 3, 7, 8, 10], false),
            (&[0], &[1, 3, 7], false),
            (&[0, 1], &[1, 2, 3, 4, 5, 7, 8], true),
            (&[1], &[1, 2, 3, 4, 5, 7, 8], true),
            (&[1, 2], &[1, 3, 7], false),
            (&[1, 2, 3], &[1, 3, 4, 5, 7, 8], true),
            (&[1, 4], &[0, 1, 3, 6, 7], false),
            (&[1, 4, 6], &[1, 3, 7], false),
            (&[1, 4, 6, 7], &[1, 3, 7, 8], true),
            (&[9], &[0, 1, 3, 7, 8, 10], false),
            (&[10], &[1, 3, 7, 8], true),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[], 2);
        assert_rejected_transition(&mut core, &[1, 4], 8);
    }

    #[test]
    fn d19_object_required_optional_branching_snapshots() {
        let mut core = core(
            r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]}"#,
            &[
                b"{",
                b"}",
                b"\"a\"",
                b"\"b\"",
                b":",
                b",",
                b"1",
                b"\"x\"",
                b" ",
                b"\"c\"",
                b"",
                b"{\"a\":1}",
                b"\t\n",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 8, 11, 12], false),
            (&[0], &[2, 3, 8, 12], false),
            (&[0, 2], &[4, 8, 12], false),
            (&[0, 2, 4], &[6, 8, 12], false),
            (&[0, 2, 4, 6], &[1, 5, 6, 8, 12], false),
            (&[0, 2, 4, 6, 1], &[8, 12], true),
            (&[0, 3, 4, 7], &[5, 8, 12], false),
            (&[0, 3, 4, 7, 5], &[2, 8, 12], false),
            (&[0, 3, 4, 7, 5, 2, 4, 6, 1], &[8, 12], true),
            (&[10], &[0, 8, 11, 12], false),
            (&[8], &[0, 8, 11, 12], false),
            (&[12], &[0, 8, 11, 12], false),
            (&[11], &[8, 12], true),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[0], 1);
        assert_rejected_transition(&mut core, &[0, 2, 4], 7);
        assert_rejected_transition(&mut core, &[0], 9);
    }

    #[test]
    fn d19_root_array_typed_items_and_bounds_snapshots() {
        let mut core = core(
            r#"{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}"#,
            &[
                b"[",
                b"]",
                b"\"a\"",
                b"\"b\"",
                b",",
                b" ",
                b"1",
                b"",
                b"[\"a\",\"b\"]",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 5, 8], false),
            (&[0], &[2, 3, 5], false),
            (&[0, 2], &[1, 4, 5], false),
            (&[0, 2, 4], &[2, 3, 5], false),
            (&[0, 2, 4, 3], &[1, 5], false),
            (&[0, 2, 4, 3, 1], &[5], true),
            (&[7], &[0, 5, 8], false),
            (&[8], &[5], true),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[0], 1);
        assert_rejected_transition(&mut core, &[0], 6);
        assert_rejected_transition(&mut core, &[0, 2, 4, 3], 4);
    }

    #[test]
    fn d19_root_string_pattern_and_lengths_snapshots() {
        let mut core = core(
            r#"{"type":"string","pattern":"^[A-Z]+$","minLength":2,"maxLength":2}"#,
            &[
                b"\"AB\"", b"\"A\"", b"\"ABC\"", b"\"ab\"", b"\"", b"A", b"B", b"C", b"",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 4], false),
            (&[4], &[5, 6, 7], false),
            (&[4, 5], &[5, 6, 7], false),
            (&[4, 5, 6], &[4], false),
            (&[4, 5, 6, 4], &[], true),
            (&[8], &[0, 4], false),
            (&[0], &[], true),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[4], 4);
        assert_rejected_transition(&mut core, &[4, 5], 4);
        assert_rejected_transition(&mut core, &[4, 5, 6], 7);
        assert_rejected_transition(&mut core, &[], 1);
        assert_rejected_transition(&mut core, &[], 3);
    }

    #[test]
    fn d19_root_integer_snapshots() {
        let mut core = core(
            r#"{"type":"integer"}"#,
            &[b"42", b".", b"e", b"-", b"1", b" ", b"", b"-7"],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 3, 4, 5, 7], false),
            (&[0], &[0, 4, 5], true),
            (&[3], &[0, 4], false),
            (&[3, 4], &[0, 4, 5], true),
            (&[6], &[0, 3, 4, 5, 7], false),
            (&[7], &[0, 4, 5], true),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[0], 1);
        assert_rejected_transition(&mut core, &[0], 2);
        assert_rejected_transition(&mut core, &[3], 5);
    }

    #[test]
    fn d19_root_boolean_null_union_snapshots() {
        let mut core = core(
            r#"{"type":["boolean","null"]}"#,
            &[
                b"true", b"false", b"null", b"t", b"f", b"n", b"rue", b"alse", b"ull", b"", b"1",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 1, 2, 3, 4, 5], false),
            (&[3], &[6], false),
            (&[3, 6], &[], true),
            (&[4], &[7], false),
            (&[4, 7], &[], true),
            (&[5], &[8], false),
            (&[5, 8], &[], true),
            (&[0], &[], true),
            (&[1], &[], true),
            (&[2], &[], true),
            (&[9], &[0, 1, 2, 3, 4, 5], false),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[], 10);
    }

    #[test]
    fn d19_root_string_enum_snapshots() {
        let mut core = core(
            r#"{"enum":["red","blue"]}"#,
            &[
                b"\"red\"",
                b"\"blue\"",
                b"\"green\"",
                b"\"",
                b"r",
                b"ed\"",
                b"",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 1, 3], false),
            (&[0], &[], true),
            (&[1], &[], true),
            (&[3], &[4], false),
            (&[3, 4], &[5], false),
            (&[3, 4, 5], &[], true),
            (&[6], &[0, 1, 3], false),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[], 2);
    }

    #[test]
    fn d19_nested_object_and_typed_array_snapshots() {
        let mut core = core(
            r#"{"type":"object","properties":{"person":{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}},"required":["name","tags"]}},"required":["person"]}"#,
            &[
                b"{",
                b"}",
                b"\"person\"",
                b":",
                b"\"name\"",
                b"\"tags\"",
                b"\"Ada\"",
                b"[",
                b"]",
                b"\"x\"",
                b",",
                b" ",
                b"{\"person\":{\"name\":\"Ada\",\"tags\":[\"x\"]}}",
                b"",
                b"\"other\"",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 11, 12], false),
            (&[0], &[2, 11], false),
            (&[0, 2, 3, 0], &[4, 5, 11], false),
            (&[0, 2, 3, 0, 4, 3, 6], &[10, 11], false),
            (&[0, 2, 3, 0, 4, 3, 6, 10], &[5, 11], false),
            (
                &[0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7],
                &[2, 4, 5, 6, 9, 11, 14],
                false,
            ),
            (&[0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7, 9], &[8, 10, 11], false),
            (&[0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7, 9, 8, 1, 1], &[11], true),
            (&[12], &[11], true),
            (&[13], &[0, 11, 12], false),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
        assert_rejected_transition(&mut core, &[0, 2, 3, 0], 1);
        assert_rejected_transition(&mut core, &[0, 2, 3, 0], 14);
        assert_rejected_transition(&mut core, &[0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7], 8);
    }

    #[test]
    fn d19_valid_escaped_quote_snapshots() {
        let mut core = core(
            r#"{"type":"string","minLength":1,"maxLength":1}"#,
            &[b"\"", b"\\\"", b"\"\\\"\"", b"", b"a"],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 2], false),
            (&[0], &[1, 4], false),
            (&[0, 1], &[0], false),
            (&[0, 1, 0], &[], true),
            (&[2], &[], true),
            (&[3], &[0, 2], false),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
    }

    #[test]
    fn d19_structural_whitespace_snapshots() {
        let mut core = core(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
            &[
                b" ",
                b"\t\n",
                b"{",
                b"}",
                b"\"a\"",
                b":",
                b"1",
                b"{ \"a\" : 1 }",
                b"",
            ],
        );
        let snapshots: &[(&[usize], &[usize], bool)] = &[
            (&[], &[0, 1, 2, 7], false),
            (&[0], &[0, 1, 2, 7], false),
            (&[1], &[0, 1, 2, 7], false),
            (&[2], &[0, 1, 4], false),
            (&[2, 0], &[0, 1, 4], false),
            (&[2, 0, 4, 0, 5, 0, 6, 0, 3], &[0, 1], true),
            (&[7], &[0, 1], true),
            (&[8], &[0, 1, 2, 7], false),
        ];
        for &(token_ids, valid_token_ids, is_match) in snapshots {
            assert_snapshot(&mut core, token_ids, valid_token_ids, is_match);
        }
    }

    #[test]
    fn d19_unicode_lengths_count_code_points_and_validate_utf8() {
        let vocabulary: &[&[u8]] = &[
            b"\"",
            b"\xC3",
            b"\xA9",
            b"\xC3\xA9",
            b"\"\xC3\xA9\"",
            b"",
            b"\"a\"",
            b"\"\xF0\x9F\x98\x80\"",
        ];
        let mut core = core(
            r#"{"type":"string","minLength":2,"maxLength":2}"#,
            vocabulary,
        );
        assert_snapshot(&mut core, &[], &[0], false);
        assert_snapshot(&mut core, &[0], &[1, 3], false);
        assert_snapshot(&mut core, &[0, 1], &[2], false);
        assert_snapshot(&mut core, &[0, 3], &[1, 3], false);
        assert_snapshot(&mut core, &[0, 3, 3], &[0], false);
        assert_snapshot(&mut core, &[0, 3, 3, 0], &[], true);
        assert_rejected_transition(&mut core, &[], 4);
        assert_rejected_transition(&mut core, &[], 6);
        assert_rejected_transition(&mut core, &[], 7);

        let root = core.init_state().unwrap();
        let parent = advance_bytes(&mut core, root, &[0]);
        let active_before = core.active_state_count();
        let error = core.advance_state(parent, 2).unwrap_err();
        assert!(matches!(
            error,
            JsonRuntimeError::InvalidTransition {
                token_id: 2,
                byte_offset: 0,
                ..
            }
        ));
        assert_eq!(core.active_state_count(), active_before);
    }

    #[test]
    fn d19_invalid_escape_forms_are_rejected() {
        let mut core = core(
            r#"{"type":"string","minLength":1,"maxLength":1}"#,
            &[b"\"", b"\\\"", b"\\x", b"\"\\\"\"", b"\"\\x\"", b"", b"a"],
        );
        assert_snapshot(&mut core, &[], &[0, 3], false);
        assert_snapshot(&mut core, &[0], &[1, 6], false);
        assert_rejected_transition(&mut core, &[0], 2);
        assert_rejected_transition(&mut core, &[], 4);
    }

    #[test]
    fn d19_single_byte_numeric_enum_candidates_complete() {
        let mut core = core(r#"{"enum":[2,3,null]}"#, &[b"2", b"3", b"null", b""]);
        assert_snapshot(&mut core, &[], &[0, 1, 2], false);
        assert_snapshot(&mut core, &[0], &[], true);
        assert_snapshot(&mut core, &[1], &[], true);
        assert_snapshot(&mut core, &[2], &[], true);
        assert_snapshot(&mut core, &[3], &[0, 1, 2], false);
    }

    #[test]
    fn d19_number_prefixes_are_strict_without_rejecting_incomplete_forms() {
        let mut core = core(
            r#"{"type":"number"}"#,
            &[
                b"0", b"1", b"01", b"-01", b"1.", b"1e", b"1e+", b"1e2", b"", b"1 ",
            ],
        );
        assert_snapshot(&mut core, &[], &[0, 1, 4, 5, 6, 7, 9], false);
        assert_rejected_transition(&mut core, &[], 2);
        assert_rejected_transition(&mut core, &[], 3);
        assert_snapshot(&mut core, &[4], &[0, 1, 2, 5, 6, 7, 9], false);
        assert_snapshot(&mut core, &[4, 0], &[0, 1, 2, 5, 6, 7, 9], true);
        assert_snapshot(&mut core, &[5], &[0, 1, 2, 3, 9], false);
        assert_snapshot(&mut core, &[5, 0], &[0, 1, 2, 9], true);
        assert_snapshot(&mut core, &[6], &[0, 1, 2, 9], false);
        assert_snapshot(&mut core, &[6, 0], &[0, 1, 2, 9], true);
        assert_snapshot(&mut core, &[9], &[], true);
    }

    #[test]
    fn compilation_errors_are_typed_and_do_not_create_a_runtime() {
        assert!(matches!(
            JsonConstraintCore::compile(Vec::new(), r#"{"type":"null"}"#),
            Err(JsonRuntimeError::InvalidInput(_))
        ));
        assert!(matches!(
            JsonConstraintCore::compile(vec![b"null".to_vec()], "not json"),
            Err(JsonRuntimeError::Compilation(_))
        ));
        assert!(matches!(
            JsonConstraintCore::compile(vec![b"x".to_vec()], r#"{"type":"string","const":"x"}"#),
            Err(JsonRuntimeError::Compilation(_))
        ));
    }

    #[test]
    fn enum_only_runtime_covers_canonical_recorded_json_values() {
        let mut core = core(
            r#"{"enum":[null,true,2,"x",[],{},[1],{"a":1}]}"#,
            &[
                b"null",
                b"true",
                b"2",
                b"\"x\"",
                b"[]",
                b"{}",
                b"[1]",
                b"{\"a\":1}",
                b"{ \"a\": 1 }",
                b" ",
                b"null ",
            ],
        );
        assert_snapshot(&mut core, &[], &[0, 1, 2, 3, 4, 5, 6, 7, 9, 10], false);
        for token_id in 0..8 {
            assert_snapshot(&mut core, &[token_id], &[9], true);
        }
        assert_rejected_transition(&mut core, &[], 8);
        assert_snapshot(&mut core, &[9], &[0, 1, 2, 3, 4, 5, 6, 7, 9, 10], false);
        assert_snapshot(&mut core, &[10], &[9], true);
    }
}
