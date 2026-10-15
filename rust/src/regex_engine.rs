//! regex constraint engine
//!
//! this module implements the DFA-based regex constraint engine.
//! it uses regex-automata to compile regex patterns into deterministic
//! finite automata for efficient vocabulary filtering during generation.

use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::util::syntax;
use regex_automata::Anchored;
use regex_syntax::hir::{Hir, Look};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

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
                .match_kind(regex_automata::MatchKind::LeftmostFirst),
        )
        .build(pattern)
        .map_err(|e| format!("Failed to compile regex: {}", e))?;

    let start_config = StartConfig::new().anchored(Anchored::Yes);
    let initial_state = dfa
        .start_state(&start_config)
        .map_err(|e| format!("Failed to get start state: {}", e))?;

    Ok(CompiledDfa { dfa, initial_state })
}

/// a JSON string pattern with its schema's length limits, and the distances
/// that decide whether a string can still close. a string's length here is
/// the number of bytes fed to the pattern, one per character or escape.
#[derive(Debug)]
pub struct StringPattern {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
    min_length: usize,
    max_length: Option<usize>,
    indices: HashMap<StateID, usize>,
    successors: Vec<Vec<usize>>,
    // fewest bytes to a match from each indexed state; usize::MAX when none
    // is reachable, such as after "ABX" for `^AB`.
    remaining: Vec<usize>,
    short_rows: ShortRows,
}

impl StringPattern {
    /// a compiled pattern for strings of `min_length..=max_length`; an error
    /// when no string within those limits matches it.
    pub fn new(
        compiled: CompiledDfa,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> Result<Self, String> {
        let graph = ByteGraph::new(&compiled.dfa, compiled.initial_state);
        let remaining = completion_lengths(&compiled.dfa, &graph);
        let min_length = min_length.unwrap_or(0);
        let short_rows = ShortRows::new(&graph, &remaining, min_length, max_length);
        let pattern = Self {
            dfa: compiled.dfa,
            initial_state: compiled.initial_state,
            min_length,
            max_length,
            indices: graph.indices,
            successors: graph.successors,
            remaining,
            short_rows,
        };
        // reject here, or generation opens a string it can never close.
        if pattern.remaining[0] == usize::MAX {
            return Err("pattern can never match".to_string());
        }
        if !pattern.can_finish(pattern.initial_state, 0) {
            return Err("pattern cannot match a string within minLength and maxLength".to_string());
        }
        Ok(pattern)
    }

    /// whether a string in `state` after `count` bytes can still close on a
    /// match whose length the schema allows.
    pub fn can_finish(&self, state: StateID, count: usize) -> bool {
        self.indices
            .get(&state)
            .is_some_and(|&index| self.can_finish_index(index, count))
    }

    /// like `can_finish`, after one more byte of any value, as an escape
    /// leaves open.
    pub fn can_finish_after_any_byte(&self, state: StateID, count: usize) -> bool {
        self.indices.get(&state).is_some_and(|&index| {
            self.successors[index]
                .iter()
                .any(|&next| self.can_finish_index(next, count + 1))
        })
    }

    fn can_finish_index(&self, index: usize, count: usize) -> bool {
        if count < self.min_length {
            return self.short_rows.row(self.min_length, count)[index];
        }
        finishes_within(self.remaining[index], count, self.max_length)
    }
}

/// from minLength on, a string closes as soon as its pattern matches, so only
/// the fewest remaining bytes matter.
fn finishes_within(remaining: usize, count: usize, max_length: Option<usize>) -> bool {
    remaining != usize::MAX
        && max_length.map_or(true, |maximum| {
            count <= maximum && remaining <= maximum - count
        })
}

/// below minLength a string cannot close yet, even on a match. row `count`
/// marks the states from which, after `count` bytes, some match of an allowed
/// length is reachable; matching lengths can have gaps (`^(A|BBBB)$` matches
/// lengths 1 and 4), so the fewest remaining bytes do not decide. each row
/// depends only on the next longer count, so once a row recurs the rows below
/// repeat with that period; only rows down to the first repeat are stored.
#[derive(Debug)]
struct ShortRows {
    // rows[k] is the row for count `min_length - 1 - k`.
    rows: Vec<Vec<bool>>,
    period: usize,
}

impl ShortRows {
    fn new(
        graph: &ByteGraph,
        remaining: &[usize],
        min_length: usize,
        max_length: Option<usize>,
    ) -> Self {
        let mut next: Vec<bool> = remaining
            .iter()
            .map(|&remaining| finishes_within(remaining, min_length, max_length))
            .collect();
        let mut seen = HashMap::new();
        let mut rows = Vec::new();
        let mut period = 0;
        for k in 0..min_length {
            let row: Vec<bool> = graph
                .successors
                .iter()
                .map(|successors| successors.iter().any(|&index| next[index]))
                .collect();
            if let Some(&first) = seen.get(&row) {
                period = k - first;
                break;
            }
            seen.insert(row.clone(), k);
            rows.push(row.clone());
            next = row;
        }
        Self { rows, period }
    }

    fn row(&self, min_length: usize, count: usize) -> &[bool] {
        let k = min_length - 1 - count;
        if k < self.rows.len() {
            return &self.rows[k];
        }
        let start = self.rows.len() - self.period;
        &self.rows[start + (k - start) % self.period]
    }
}

/// requested regexes can determinize exponentially, e.g. `(a|b)*a(a|b){24}`,
/// taking minutes and gigabytes. realistic patterns, including unicode classes
/// such as `\w{1,200}` (about 31 MiB), stay below this bound.
const REGEX_DFA_SIZE_LIMIT: usize = 64 << 20;

fn compile_error(e: impl std::fmt::Display) -> ConstraintError {
    ConstraintError::CompilationError(format!("Failed to compile regex: {}", e))
}

/// the DFA states reachable from `initial` one byte at a time, without the
/// dead state, and their byte transitions; `initial` has index 0.
struct ByteGraph {
    states: Vec<StateID>,
    indices: HashMap<StateID, usize>,
    successors: Vec<Vec<usize>>,
}

impl ByteGraph {
    fn new(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> Self {
        let mut states = vec![initial];
        let mut indices = HashMap::from([(initial, 0)]);
        let mut successors = Vec::new();
        let mut index = 0;
        while index < states.len() {
            let state = states[index];
            let mut next_indices: Vec<usize> = dfa
                .byte_classes()
                .representatives(..)
                .filter_map(|unit| unit.as_u8())
                .map(|byte| dfa.next_state(state, byte))
                .filter(|&next| !dfa.is_dead_state(next))
                .map(|next| {
                    *indices.entry(next).or_insert_with(|| {
                        states.push(next);
                        states.len() - 1
                    })
                })
                .collect();
            next_indices.sort_unstable();
            next_indices.dedup();
            successors.push(next_indices);
            index += 1;
        }
        Self {
            states,
            indices,
            successors,
        }
    }
}

/// fewest bytes from each graph state to a match at the end of input, by
/// reverse breadth-first search; usize::MAX when no match is reachable.
fn completion_lengths(dfa: &dense::DFA<Vec<u32>>, graph: &ByteGraph) -> Vec<usize> {
    let mut reverse = vec![Vec::new(); graph.states.len()];
    for (index, successors) in graph.successors.iter().enumerate() {
        for &next in successors {
            reverse[next].push(index);
        }
    }
    let mut lengths = vec![usize::MAX; graph.states.len()];
    let mut queue = VecDeque::new();
    for (index, &state) in graph.states.iter().enumerate() {
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
    lengths
}

/// whether a match at the end of input is still reachable from each state,
/// indexed by `state >> stride2`. determinization keeps states that can never
/// match apart from the dead state, such as the one after `a` for `a$b`; a
/// token entering one leaves generation with no valid continuation.
fn live_states(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> Vec<bool> {
    let graph = ByteGraph::new(dfa, initial);
    let lengths = completion_lengths(dfa, &graph);
    let slot = |state: StateID| state.as_usize() >> dfa.stride2();
    let slots = graph
        .states
        .iter()
        .map(|&state| slot(state) + 1)
        .max()
        .unwrap_or(0);
    let mut live = vec![false; slots];
    for (&state, &length) in graph.states.iter().zip(&lengths) {
        live[slot(state)] = length != usize::MAX;
    }
    live
}

/// a regex constraint engine using DFA traversal
///
/// this struct holds a compiled DFA and tracks the current state
/// during token generation. it implements the ConstraintEngine trait
/// to provide vocabulary filtering based on regex patterns.
pub struct RegexEngine {
    vocabulary: Arc<Vec<Vec<u8>>>,
    dfa: Arc<dense::DFA<Vec<u32>>>,
    live: Arc<Vec<bool>>,
    current_state: StateID,
    initial_state: StateID,
}

impl RegexEngine {
    /// create a new RegexEngine with the given vocabulary and pattern
    ///
    /// compiles the regex pattern into a DFA and initializes the state.
    /// the pattern must match the whole output: without the end anchor, a
    /// token that completes the match and then continues is not dead.
    pub fn new(vocabulary: Vec<Vec<u8>>, pattern: &str) -> Result<Self, ConstraintError> {
        // anchor the parsed pattern rather than the text: splicing it into
        // `\A(?:...)\z` accepts unbalanced input such as `a)|(b`, and a
        // trailing `(?x)` comment would swallow the closing anchor.
        let hir = syntax::parse(pattern).map_err(compile_error)?;
        let hir = Hir::concat(vec![Hir::look(Look::Start), hir, Hir::look(Look::End)]);
        let nfa = thompson::Compiler::new()
            .configure(thompson::Config::new().which_captures(thompson::WhichCaptures::None))
            .build_from_hir(&hir)
            .map_err(compile_error)?;

        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::LeftmostFirst)
                    .dfa_size_limit(Some(REGEX_DFA_SIZE_LIMIT))
                    .determinize_size_limit(Some(REGEX_DFA_SIZE_LIMIT)),
            )
            .build_from_nfa(&nfa)
            .map_err(compile_error)?;

        let start_config = StartConfig::new().anchored(Anchored::Yes);
        let initial_state = dfa.start_state(&start_config).map_err(|e| {
            ConstraintError::CompilationError(format!("Failed to get start state: {}", e))
        })?;

        let engine = RegexEngine {
            vocabulary: Arc::new(vocabulary),
            live: Arc::new(live_states(&dfa, initial_state)),
            dfa: Arc::new(dfa),
            current_state: initial_state,
            initial_state,
        };
        // reject before any model work instead of failing at the first token.
        if !engine.is_live(initial_state) {
            return Err(compile_error("pattern can never match"));
        }
        Ok(engine)
    }

    fn is_live(&self, state: StateID) -> bool {
        self.live
            .get(state.as_usize() >> self.dfa.stride2())
            .copied()
            .unwrap_or(false)
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

            // not dead is not enough: the token must leave a match reachable.
            if is_valid && self.is_live(test_state) {
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
        Box::new(RegexEngine {
            vocabulary: Arc::clone(&self.vocabulary),
            dfa: Arc::clone(&self.dfa),
            live: Arc::clone(&self.live),
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
        let vocab = make_test_vocab();
        let engine = RegexEngine::new(vocab, "The year is [0-9]{4}").unwrap();
        assert_eq!(engine.vocab_size(), 10);
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_exponential_regex_is_rejected() {
        let error = RegexEngine::new(make_test_vocab(), "(a|b)*a(a|b){24}")
            .err()
            .expect("an exponential DFA must be rejected");
        assert!(error.to_string().contains("size limit"), "{error}");
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
        engine.advance(0).unwrap(); // "The"
        engine.advance(1).unwrap(); // " year"
        engine.advance(2).unwrap(); // " is"
        engine.advance(3).unwrap(); // " "
        engine.advance(4).unwrap(); // "2"
        engine.advance(5).unwrap(); // "0"
        engine.advance(6).unwrap(); // "1"
        engine.advance(7).unwrap(); // "9"

        assert!(engine.is_finished());
        assert!(!engine.is_dead());
    }

    #[test]
    fn test_complete_match_rejects_trailing_bytes_in_same_token() {
        let vocab = vec![
            b"ABC-123".to_vec(),
            b"5.".to_vec(),
            b"5".to_vec(),
            b".".to_vec(),
        ];
        let mut engine = RegexEngine::new(vocab, "[A-Z]{3}-[0-9]{4}").unwrap();

        engine.advance(0).unwrap(); // "ABC-123"
        assert_eq!(engine.get_valid_tokens(), vec![2]);

        engine.advance(2).unwrap(); // "5"
        assert!(engine.is_finished());
        assert!(engine.get_valid_tokens().is_empty());
    }

    #[test]
    fn test_unbalanced_pattern_is_rejected() {
        // wrapped as text, `\A(?:a)|(b)\z` would compile with an unanchored end.
        assert!(RegexEngine::new(make_test_vocab(), "a)|(b").is_err());
        assert!(RegexEngine::new(make_test_vocab(), "a)(?:b").is_err());
    }

    #[test]
    fn test_verbose_pattern_with_trailing_comment_is_anchored() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"bc".to_vec()];
        let mut engine = RegexEngine::new(vocab, "(?x) a b  # letters").unwrap();

        engine.advance(0).unwrap(); // "a"
        assert_eq!(engine.get_valid_tokens(), vec![1]);
        engine.advance(1).unwrap(); // "b"
        assert!(engine.is_finished());
    }

    #[test]
    fn test_pattern_that_can_never_match_is_rejected() {
        for pattern in [r"[^\s\S]", "a$b", r"x(?-u:\b)y"] {
            let error = RegexEngine::new(make_test_vocab(), pattern)
                .err()
                .unwrap_or_else(|| panic!("{pattern} compiled"));
            assert!(
                error.to_string().contains("can never match"),
                "{pattern}: {error}"
            );
        }
        // matching only the empty string still matches.
        assert!(RegexEngine::new(make_test_vocab(), "(?:)")
            .unwrap()
            .is_finished());
    }

    #[test]
    fn test_tokens_into_branches_that_can_never_match_are_invalid() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"bc".to_vec()];
        // `b$c` never matches, so `b` and `bc` would leave no continuation.
        let engine = RegexEngine::new(vocab, "(?:a|b$c)").unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![0]);
    }
}
