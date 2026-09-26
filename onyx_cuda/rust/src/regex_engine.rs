//! DFA-based regex constraint engine.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::util::syntax;
use regex_automata::Anchored;
use regex_syntax::hir::{Hir, Look};
use regex_syntax::utf8::Utf8Sequences;

use crate::constraint::{ConstraintEngine, ConstraintError};

pub struct CompiledDfa {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
}

/// Client patterns can determinize exponentially: `(a|b)*a(a|b){24}` would take
/// minutes and gigabytes. Bound construction and the result; realistic patterns,
/// including Unicode classes such as `\w{1,200}` (about 31 MiB), stay below it.
const DFA_SIZE_LIMIT: usize = 64 << 20;

fn dfa_config() -> dense::Config {
    dfa_config_with_limit(DFA_SIZE_LIMIT)
}

fn dfa_config_with_limit(limit: usize) -> dense::Config {
    dense::Config::new()
        .start_kind(regex_automata::dfa::StartKind::Anchored)
        .match_kind(regex_automata::MatchKind::LeftmostFirst)
        .dfa_size_limit(Some(limit))
        .determinize_size_limit(Some(limit))
}

fn compile_error(error: impl std::fmt::Display) -> ConstraintError {
    ConstraintError::CompilationError(format!("Failed to compile regex: {error}"))
}

pub fn compile_pattern_dfa(pattern: &str) -> Result<CompiledDfa, String> {
    let dfa = dense::Builder::new()
        .configure(dfa_config())
        .build(pattern)
        .map_err(|error| format!("Failed to compile regex: {error}"))?;

    let start_config = StartConfig::new().anchored(Anchored::Yes);
    let initial_state = dfa
        .start_state(&start_config)
        .map_err(|error| format!("Failed to get start state: {error}"))?;

    Ok(CompiledDfa { dfa, initial_state })
}

/// Shared JSON string pattern and minimum completion lengths in characters.
#[derive(Debug)]
pub struct StringPattern {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
    remaining: HashMap<StateID, usize>,
    // Partial UTF-8 and JSON escapes repeatedly ask about the same character
    // intervals during vocabulary scans. Bound this schema-owned cache.
    range_remaining: Mutex<HashMap<(StateID, u32, u32), Option<usize>>>,
}

impl StringPattern {
    pub fn new(normalized: &str, bounded: bool) -> Self {
        let compiled = compile_pattern_dfa(&format!("(?s:.*(?:{normalized}).*)"))
            .expect("validated pattern DFA");
        let remaining = if bounded {
            completion_lengths(&compiled.dfa, compiled.initial_state)
        } else {
            HashMap::new()
        };
        Self {
            dfa: compiled.dfa,
            initial_state: compiled.initial_state,
            remaining,
            range_remaining: Mutex::new(HashMap::new()),
        }
    }

    pub fn can_finish(&self, state: StateID, budget: usize) -> bool {
        self.remaining
            .get(&state)
            .is_some_and(|&length| length <= budget)
    }

    pub fn can_finish_after_character(
        &self,
        state: StateID,
        low: u32,
        high: u32,
        budget: usize,
    ) -> bool {
        if budget == 0 || low > high {
            return false;
        }
        let key = (state, low, high);
        let cached = self
            .range_remaining
            .lock()
            .expect("pattern cache lock")
            .get(&key)
            .copied();
        let length = cached.unwrap_or_else(|| {
            let length = character_successors(&self.dfa, state, low, high)
                .iter()
                .filter_map(|state| self.remaining.get(state).copied())
                .min();
            let mut cache = self.range_remaining.lock().expect("pattern cache lock");
            if cache.len() < 4096 {
                cache.insert(key, length);
            }
            length
        });
        length.is_some_and(|length| length < budget)
    }
}

/// Advance by one Unicode scalar, without enumerating 1.1 million characters.
/// UTF-8 byte ranges exclude overlong encodings, surrogates and invalid scalars.
fn character_successors(
    dfa: &dense::DFA<Vec<u32>>,
    state: StateID,
    mut low: u32,
    mut high: u32,
) -> Vec<StateID> {
    if (0xd800..=0xdfff).contains(&low) {
        low = 0xe000;
    }
    if (0xd800..=0xdfff).contains(&high) {
        high = 0xd7ff;
    }
    high = high.min(0x10ffff);
    if low > high {
        return Vec::new();
    }
    let mut successors = Vec::new();
    for sequence in Utf8Sequences::new(char::from_u32(low).unwrap(), char::from_u32(high).unwrap())
    {
        let mut states = vec![state];
        for range in sequence.as_slice() {
            let mut next = Vec::new();
            for state in states {
                for byte in range.start..=range.end {
                    let state = dfa.next_state(state, byte);
                    if !dfa.is_dead_state(state) {
                        next.push(state);
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            states = next;
        }
        successors.extend(states);
    }
    successors.sort_unstable();
    successors.dedup();
    successors
}

/// Reverse breadth-first search over character transitions, computed once per
/// blueprint. The runtime check is independent of the requested length limit.
fn completion_lengths(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> HashMap<StateID, usize> {
    let mut states = vec![initial];
    let mut indices = HashMap::from([(initial, 0)]);
    let mut reverse = vec![Vec::new()];
    let mut index = 0;
    while index < states.len() {
        for next in character_successors(dfa, states[index], 0, 0x10ffff) {
            let next_index = *indices.entry(next).or_insert_with(|| {
                states.push(next);
                reverse.push(Vec::new());
                states.len() - 1
            });
            reverse[next_index].push(index);
        }
        index += 1;
    }
    let mut lengths = vec![usize::MAX; states.len()];
    let mut queue = VecDeque::new();
    for (index, &state) in states.iter().enumerate() {
        if dfa.is_match_state(dfa.next_eoi_state(state)) {
            lengths[index] = 0;
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        for &previous in &reverse[index] {
            if lengths[previous] == usize::MAX {
                lengths[previous] = lengths[index] + 1;
                queue.push_back(previous);
            }
        }
    }
    states
        .into_iter()
        .zip(lengths)
        .filter(|(_, length)| *length != usize::MAX)
        .collect()
}

pub struct RegexEngine {
    vocabulary: Arc<Vec<Vec<u8>>>,
    dfa: Arc<dense::DFA<Vec<u32>>>,
    current_state: StateID,
    initial_state: StateID,
}

impl RegexEngine {
    pub fn new(vocabulary: Vec<Vec<u8>>, pattern: &str) -> Result<Self, ConstraintError> {
        // Anchor the parsed pattern rather than the text: splicing it into
        // `\A(?:...)\z` accepts unbalanced input such as `a)|(b`, and a
        // trailing `(?x)` comment would swallow the closing anchor.
        let hir = syntax::parse(pattern).map_err(compile_error)?;
        let hir = Hir::concat(vec![Hir::look(Look::Start), hir, Hir::look(Look::End)]);
        let nfa = thompson::Compiler::new()
            .configure(thompson::Config::new().which_captures(thompson::WhichCaptures::None))
            .build_from_hir(&hir)
            .map_err(compile_error)?;
        let dfa = dense::Builder::new()
            .configure(dfa_config())
            .build_from_nfa(&nfa)
            .map_err(compile_error)?;

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
    fn test_every_compiled_pattern_is_size_limited() {
        let config = dfa_config();
        assert_eq!(config.get_dfa_size_limit(), Some(DFA_SIZE_LIMIT));
        assert_eq!(config.get_determinize_size_limit(), Some(DFA_SIZE_LIMIT));
    }

    #[test]
    fn test_exponential_pattern_fails_instead_of_exhausting_memory() {
        // Rejecting at the production limit takes seconds in a debug build.
        let build = |pattern: &str| {
            dense::Builder::new()
                .configure(dfa_config_with_limit(1 << 20))
                .build(pattern)
        };
        let error = build("(a|b)*a(a|b){24}").unwrap_err().to_string();
        assert!(error.contains("size limit"), "{error}");
        assert!(build("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{12}").is_ok());
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
    #[test]
    fn test_complete_marker_rejects_trailing_bytes_in_same_token() {
        let mut engine = RegexEngine::new(
            vec![b"<think>".to_vec(), b"ok".to_vec(), b"</think>".to_vec(),
                 b"</think>\n".to_vec()], "<think>ok</think>"
        ).unwrap();
        engine.advance(0).unwrap();
        engine.advance(1).unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![2]);
        engine.advance(2).unwrap();
        assert!(engine.is_finished());
    }

    #[test]
    fn test_unbalanced_pattern_is_rejected() {
        // Wrapped as text, `\A(?:a)|(b)\z` would compile with an unanchored end.
        assert!(RegexEngine::new(make_test_vocab(), "a)|(b").is_err());
        assert!(RegexEngine::new(make_test_vocab(), "a)(?:b").is_err());
    }

    #[test]
    fn test_verbose_pattern_with_trailing_comment_is_anchored() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"bc".to_vec()];
        let mut engine = RegexEngine::new(vocab, "(?x) a b  # letters").unwrap();
        engine.advance(0).unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![1]);
        engine.advance(1).unwrap();
        assert!(engine.is_finished());
    }
}
