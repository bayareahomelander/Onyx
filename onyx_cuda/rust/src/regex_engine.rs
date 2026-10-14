//! DFA-based regex constraint engine.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::util::syntax;
use regex_automata::Anchored;
use regex_syntax::hir::{Hir, Look};
use regex_syntax::utf8::Utf8Sequences;

use crate::constraint::{ConstraintEngine, ConstraintError, ScanKey};

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

/// JSON Schema patterns search: a match may occur anywhere in the string.
fn schema_pattern_search(normalized: &str) -> String {
    format!("(?s:.*(?:{normalized}).*)")
}

/// Reject a normalized schema pattern that no string satisfies within
/// minLength and maxLength, so generation never opens a string it can never
/// close.
pub fn check_pattern_satisfiable(
    normalized: &str,
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> Result<(), String> {
    let compiled = compile_pattern_dfa(&schema_pattern_search(normalized))?;
    let Some(bytes) = shortest_match_bytes(&compiled.dfa, compiled.initial_state) else {
        return Err("pattern can never match".to_string());
    };
    if min_length.is_some_and(|minimum| minimum > 0) {
        // Matching lengths can have gaps, such as 1 and 4 for `^(A|BBBB)$`,
        // so only the per-count rows decide.
        let pattern = StringPattern::from_compiled(compiled, min_length, max_length);
        return if pattern.can_finish(pattern.initial_state, 0) {
            Ok(())
        } else {
            Err("pattern cannot match a string within minLength and maxLength".to_string())
        };
    }
    // Characters never outnumber bytes, so the cheap byte search usually decides.
    let Some(maximum) = max_length.filter(|&maximum| bytes > maximum) else {
        return Ok(());
    };
    let graph = CharacterGraph::new(&compiled.dfa, compiled.initial_state);
    match completion_lengths(&compiled.dfa, &graph)[0] {
        characters if characters != usize::MAX && characters > maximum => Err(format!(
            "pattern needs at least {characters} characters, more than maxLength"
        )),
        _ => Ok(()),
    }
}

/// Bytes in the shortest input that reaches a match at the end of input, by
/// breadth-first search over byte classes; None when no match is reachable.
fn shortest_match_bytes(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> Option<usize> {
    let mut seen = HashSet::from([initial]);
    let mut layer = vec![initial];
    let mut length = 0;
    while !layer.is_empty() {
        if layer
            .iter()
            .any(|&state| dfa.is_match_state(dfa.next_eoi_state(state)))
        {
            return Some(length);
        }
        let mut next_layer = Vec::new();
        for state in layer {
            for byte in dfa.byte_classes().representatives(..).filter_map(|unit| unit.as_u8()) {
                let next = dfa.next_state(state, byte);
                if !dfa.is_dead_state(next) && seen.insert(next) {
                    next_layer.push(next);
                }
            }
        }
        layer = next_layer;
        length += 1;
    }
    None
}

/// Shared JSON string pattern with its schema's length limits, and the
/// character distances that decide whether a string can still close.
#[derive(Debug)]
pub struct StringPattern {
    pub dfa: dense::DFA<Vec<u32>>,
    pub initial_state: StateID,
    min_length: usize,
    max_length: Option<usize>,
    indices: HashMap<StateID, usize>,
    // Fewest characters to a match from each indexed state; usize::MAX when
    // none is reachable, such as any state after a leading `X` for `^AB`.
    remaining: Vec<usize>,
    short_rows: ShortRows,
    // Partial UTF-8 and JSON escapes repeatedly ask about the same character
    // intervals during vocabulary scans. Bound this schema-owned cache.
    range_successors: Mutex<HashMap<(StateID, u32, u32), Arc<[StateID]>>>,
}

impl StringPattern {
    pub fn new(normalized: &str, min_length: Option<usize>, max_length: Option<usize>) -> Self {
        let compiled = compile_pattern_dfa(&schema_pattern_search(normalized))
            .expect("validated pattern DFA");
        Self::from_compiled(compiled, min_length, max_length)
    }

    fn from_compiled(
        compiled: CompiledDfa,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> Self {
        let graph = CharacterGraph::new(&compiled.dfa, compiled.initial_state);
        let remaining = completion_lengths(&compiled.dfa, &graph);
        let min_length = min_length.unwrap_or(0);
        let short_rows = ShortRows::new(&graph, &remaining, min_length, max_length);
        Self {
            dfa: compiled.dfa,
            initial_state: compiled.initial_state,
            min_length,
            max_length,
            indices: graph.indices,
            remaining,
            short_rows,
            range_successors: Mutex::new(HashMap::new()),
        }
    }

    /// Whether a string in `state` after `count` characters can still close
    /// on a match whose length the schema allows.
    pub fn can_finish(&self, state: StateID, count: usize) -> bool {
        let Some(&index) = self.indices.get(&state) else {
            return false;
        };
        if count < self.min_length {
            return self.short_rows.row(self.min_length, count)[index];
        }
        finishes_within(self.remaining[index], count, self.max_length)
    }

    /// Like `can_finish`, after one more character from `low..=high`.
    pub fn can_finish_after_character(
        &self,
        state: StateID,
        low: u32,
        high: u32,
        count: usize,
    ) -> bool {
        let key = (state, low, high);
        let cached = self
            .range_successors
            .lock()
            .expect("pattern cache lock")
            .get(&key)
            .cloned();
        let successors = cached.unwrap_or_else(|| {
            let successors: Arc<[StateID]> = character_successors(&self.dfa, state, low, high).into();
            let mut cache = self.range_successors.lock().expect("pattern cache lock");
            if cache.len() < 4096 {
                cache.insert(key, successors.clone());
            }
            successors
        });
        successors
            .iter()
            .any(|&state| self.can_finish(state, count + 1))
    }
}

/// From minLength characters on, a string closes as soon as its pattern
/// matches, so only the fewest remaining characters matter.
fn finishes_within(remaining: usize, count: usize, max_length: Option<usize>) -> bool {
    remaining != usize::MAX
        && max_length.map_or(true, |maximum| {
            count <= maximum && remaining <= maximum - count
        })
}

/// Below minLength a string cannot close yet, even on a match. Row `count`
/// marks the states from which, after `count` characters, some match of an
/// allowed length is reachable; matching lengths can have gaps, so the
/// fewest remaining characters do not decide. Each row depends only on the
/// next longer count, so once a row recurs the rows below repeat with that
/// period; only rows down to the first repeat are stored.
#[derive(Debug)]
struct ShortRows {
    // rows[k] is the row for count `min_length - 1 - k`.
    rows: Vec<Vec<bool>>,
    period: usize,
}

impl ShortRows {
    fn new(
        graph: &CharacterGraph,
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

/// The DFA states reachable from `initial` one Unicode scalar at a time,
/// with their character transitions; `initial` has index 0.
struct CharacterGraph {
    states: Vec<StateID>,
    indices: HashMap<StateID, usize>,
    successors: Vec<Vec<usize>>,
}

impl CharacterGraph {
    fn new(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> Self {
        let mut states = vec![initial];
        let mut indices = HashMap::from([(initial, 0)]);
        let mut successors = Vec::new();
        let mut index = 0;
        while index < states.len() {
            let mut next_indices = Vec::new();
            for next in character_successors(dfa, states[index], 0, 0x10ffff) {
                next_indices.push(*indices.entry(next).or_insert_with(|| {
                    states.push(next);
                    states.len() - 1
                }));
            }
            successors.push(next_indices);
            index += 1;
        }
        Self { states, indices, successors }
    }
}

/// Fewest characters from each graph state to a match, by reverse
/// breadth-first search, computed once per blueprint; usize::MAX when no
/// match is reachable.
fn completion_lengths(dfa: &dense::DFA<Vec<u32>>, graph: &CharacterGraph) -> Vec<usize> {
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

/// Whether a match at the end of input is still reachable from each state,
/// indexed by `state >> stride2`. Determinization keeps states that can never
/// match apart from the dead state, such as the one after `a` for `a$b`; a
/// token entering one leaves generation with no valid continuation.
fn live_states(dfa: &dense::DFA<Vec<u32>>, initial: StateID) -> Vec<bool> {
    let mut states = vec![initial];
    let mut indices = HashMap::from([(initial, 0)]);
    let mut reverse = vec![Vec::new()];
    let mut index = 0;
    while index < states.len() {
        let mut successors: Vec<StateID> = dfa
            .byte_classes()
            .representatives(..)
            .filter_map(|unit| unit.as_u8())
            .map(|byte| dfa.next_state(states[index], byte))
            .filter(|&state| !dfa.is_dead_state(state))
            .collect();
        successors.sort_unstable();
        successors.dedup();
        for next in successors {
            let next_index = *indices.entry(next).or_insert_with(|| {
                states.push(next);
                reverse.push(Vec::new());
                states.len() - 1
            });
            reverse[next_index].push(index);
        }
        index += 1;
    }
    let mut reached = vec![false; states.len()];
    let mut queue: VecDeque<usize> = (0..states.len())
        .filter(|&index| dfa.is_match_state(dfa.next_eoi_state(states[index])))
        .collect();
    for &index in &queue {
        reached[index] = true;
    }
    while let Some(index) = queue.pop_front() {
        for &previous in &reverse[index] {
            if !reached[previous] {
                reached[previous] = true;
                queue.push_back(previous);
            }
        }
    }
    let slot = |state: StateID| state.as_usize() >> dfa.stride2();
    let mut live = vec![false; states.iter().map(|&state| slot(state) + 1).max().unwrap_or(0)];
    for (state, reached) in states.into_iter().zip(reached) {
        live[slot(state)] = reached;
    }
    live
}

pub struct RegexEngine {
    vocabulary: Arc<Vec<Vec<u8>>>,
    dfa: Arc<dense::DFA<Vec<u32>>>,
    live: Arc<Vec<bool>>,
    current_state: StateID,
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

        let engine = Self {
            vocabulary: Arc::new(vocabulary),
            live: Arc::new(live_states(&dfa, initial_state)),
            dfa: Arc::new(dfa),
            current_state: initial_state,
        };
        // Reject before any model work instead of failing at the first token.
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

    fn advance_state_by_token(&self, state: StateID, token_id: usize) -> StateID {
        let mut current = state;
        for &byte in &self.vocabulary[token_id] {
            current = self.dfa.next_state(current, byte);
        }
        current
    }
}

impl ConstraintEngine for RegexEngine {
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

            // Not dead is not enough: the token must leave a match reachable.
            if is_valid && self.is_live(state) {
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

    fn clone_box(&self) -> Box<dyn ConstraintEngine> {
        Box::new(Self {
            vocabulary: Arc::clone(&self.vocabulary),
            dfa: Arc::clone(&self.dfa),
            live: Arc::clone(&self.live),
            current_state: self.current_state,
        })
    }

    fn scan_key(&self) -> Option<ScanKey> {
        // A scan walks only the DFA from the current state.
        Some(ScanKey::Regex(self.current_state.as_u32()))
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
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_regex_engine_advance() {
        let mut engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();

        engine.advance(0).unwrap();
        assert!(!engine.is_dead());
        assert!(!engine.is_finished());

        let mut engine = RegexEngine::new(make_test_vocab(), "The year is [0-9]{4}").unwrap();
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

    #[test]
    fn test_pattern_that_can_never_match_is_rejected() {
        for pattern in [r"[^\s\S]", "a$b", r"x(?-u:\b)y"] {
            let error = RegexEngine::new(make_test_vocab(), pattern)
                .err()
                .unwrap_or_else(|| panic!("{pattern} compiled"));
            assert!(error.to_string().contains("can never match"), "{pattern}: {error}");
        }
        // Matching only the empty string still matches.
        assert!(RegexEngine::new(make_test_vocab(), "(?:)").unwrap().is_finished());
    }

    #[test]
    fn test_tokens_into_branches_that_can_never_match_are_invalid() {
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"bc".to_vec()];
        // `b$c` never matches, so `b` and `bc` would leave no continuation.
        let engine = RegexEngine::new(vocab, "(?:a|b$c)").unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![0]);
    }
}
