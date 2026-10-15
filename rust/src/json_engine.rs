//! JSON schema constraint engine with stack-based architecture
//!
//! this module implements a stack-based state machine for enforcing JSON schema
//! constraints during LLM generation. each level of JSON nesting has its own
//! scope on the stack, enabling proper tracking of nested objects and arrays.

use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::constraint::{ConstraintEngine, ConstraintError};
use crate::regex_engine::StringPattern;
use crate::schema::{PropertyBlueprint, SchemaBlueprint, SchemaType};

/// syntax state within an object scope
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectSyntaxState {
    /// expecting '"' to start a key or '}' for empty/end
    ExpectKeyOrEnd,
    /// after a comma, a key is mandatory
    ExpectKey,
    /// inside a key string, accumulating characters
    InKey,
    /// after key closing quote, expecting ':'
    ExpectColon,
    /// after ':', expecting a value
    ExpectValue,
    /// after a value, expecting ',' or '}'
    ExpectCommaOrEnd,
}

/// one byte of a json string: a decoded character, part of an unfinished
/// escape or UTF-8 sequence, or the closing quote
enum StringStep {
    Pending,
    Character(char),
    End,
}

/// state for parsing a json string value
#[derive(Debug, Clone)]
pub struct StringState {
    /// bytes of an unfinished escape or UTF-8 sequence
    pending: Vec<u8>,
    /// compiled pattern with its length limits
    pattern: Option<Arc<StringPattern>>,
    /// current dfa state
    dfa_state: Option<StateID>,
    /// unicode characters decoded so far
    char_count: usize,
    /// min chars required
    min_length: Option<usize>,
    /// max chars allowed
    max_length: Option<usize>,
}

impl StringState {
    /// new string state with opening quote seen
    pub fn new_started() -> Self {
        StringState {
            pending: Vec::new(),
            pattern: None,
            dfa_state: None,
            char_count: 0,
            min_length: None,
            max_length: None,
        }
    }

    /// string state with pattern and length constraints
    pub fn with_pattern_and_constraints(
        pattern: Option<&Arc<StringPattern>>,
        min_len: Option<usize>,
        max_len: Option<usize>,
    ) -> Self {
        StringState {
            pattern: pattern.cloned(),
            dfa_state: pattern.map(|p| p.initial_state),
            min_length: min_len,
            max_length: max_len,
            ..StringState::new_started()
        }
    }

    /// decode one byte of the string's JSON text; None when it is not valid
    /// JSON: a raw control character, invalid UTF-8, or a bad escape.
    fn decode_byte(&mut self, byte: u8) -> Option<StringStep> {
        if self.pending.is_empty() {
            return match byte {
                b'"' => Some(StringStep::End),
                b'\\' | 0xc2..=0xf4 => {
                    self.pending.push(byte);
                    Some(StringStep::Pending)
                }
                0x20..=0x7f => Some(StringStep::Character(byte as char)),
                _ => None,
            };
        }
        self.pending.push(byte);
        let character = if self.pending[0] == b'\\' {
            match self.pending.len() {
                2 => match byte {
                    b'"' | b'\\' | b'/' => byte as char,
                    b'b' => '\x08',
                    b'f' => '\x0c',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => return Some(StringStep::Pending),
                    _ => return None,
                },
                // the hex digits of one Unicode escape, or of a surrogate
                // pair's second escape
                3..=6 | 9..=12 => {
                    if !byte.is_ascii_hexdigit() {
                        return None;
                    }
                    let length = self.pending.len();
                    if length != 6 && length != 12 {
                        return Some(StringStep::Pending);
                    }
                    let first = hex_value(&self.pending[2..6]);
                    if length == 6 {
                        // a high surrogate waits for its low surrogate
                        if (0xd800..=0xdbff).contains(&first) {
                            return Some(StringStep::Pending);
                        }
                        char::from_u32(first)?
                    } else {
                        let second = hex_value(&self.pending[8..12]);
                        if !(0xdc00..=0xdfff).contains(&second) {
                            return None;
                        }
                        char::from_u32(0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00))?
                    }
                }
                7 if byte == b'\\' => return Some(StringStep::Pending),
                8 if byte == b'u' => return Some(StringStep::Pending),
                _ => return None,
            }
        } else {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => text.chars().next()?,
                Err(error) if error.error_len().is_none() => return Some(StringStep::Pending),
                Err(_) => return None,
            }
        };
        self.pending.clear();
        Some(StringStep::Character(character))
    }

    /// scalar ranges this unfinished escape or UTF-8 sequence can still
    /// encode; empty when it can encode none, such as a lone low surrogate.
    fn pending_character_ranges(&self) -> Vec<(u32, u32)> {
        fn hex_range(digits: &[u8]) -> (u32, u32) {
            let bits = 4 * (4 - digits.len() as u32);
            let prefix = hex_value(digits) << bits;
            (prefix, prefix | ((1 << bits) - 1))
        }

        if self.pending[0] != b'\\' {
            let (width, mask, minimum) = match self.pending[0] {
                0xc2..=0xdf => (2, 0x1f, 0x80),
                0xe0..=0xef => (3, 0x0f, 0x800),
                _ => (4, 0x07, 0x10000),
            };
            let prefix = self.pending[1..]
                .iter()
                .fold((self.pending[0] & mask) as u32, |value, &byte| {
                    (value << 6) | (byte & 0x3f) as u32
                });
            let bits = 6 * (width - self.pending.len() as u32);
            let low = prefix << bits;
            return vec![(low.max(minimum), (low | ((1 << bits) - 1)).min(0x10ffff))];
        }
        // a lone backslash can still begin a Unicode escape for any scalar.
        if self.pending.len() <= 2 {
            return vec![(0, 0x10ffff)];
        }
        let mut ranges = Vec::new();
        if self.pending.len() < 6 {
            let (low, high) = hex_range(&self.pending[2..]);
            ranges.push((low, high.min(0xd7ff)));
            ranges.push((low.max(0xe000), high));
            // high surrogates stand for the supplementary planes.
            let (first, last) = (low.max(0xd800), high.min(0xdbff));
            if first <= last {
                ranges.push((
                    0x10000 + ((first - 0xd800) << 10),
                    0x10000 + ((last - 0xd800) << 10) + 0x3ff,
                ));
            }
        } else {
            // a complete high surrogate, then its low surrogate's prefix
            let high = hex_value(&self.pending[2..6]);
            let (low_first, low_last) = hex_range(self.pending.get(8..).unwrap_or_default());
            let (first, last) = (low_first.max(0xdc00), low_last.min(0xdfff));
            let base = 0x10000 + ((high - 0xd800) << 10);
            ranges.push((base + first - 0xdc00, base + last - 0xdc00));
        }
        ranges.retain(|&(low, high)| low <= high);
        ranges
    }

    /// whether this unfinished character can still become one after which
    /// the pattern can close at an allowed length.
    fn can_complete_pending(&self) -> bool {
        let ranges = self.pending_character_ranges();
        match (&self.pattern, self.dfa_state) {
            (Some(pattern), Some(state)) => ranges.iter().any(|&(low, high)| {
                pattern.can_finish_after_character(state, low, high, self.char_count)
            }),
            _ => !ranges.is_empty(),
        }
    }

    /// whether this unfinished character can still spell the next character
    /// of an allowed key, raw or escaped; being valid JSON is not enough.
    fn can_complete_key(&self, remaining: &str) -> bool {
        if self.pending.is_empty() {
            return true;
        }
        let Some(character) = remaining.chars().next() else {
            return false;
        };
        let mut utf8 = [0; 4];
        if character
            .encode_utf8(&mut utf8)
            .as_bytes()
            .starts_with(&self.pending)
        {
            return true;
        }
        let quoted = serde_json::to_string(&character.to_string()).expect("JSON character");
        if quoted.as_bytes()[1..quoted.len() - 1].starts_with(&self.pending)
            || (character == '/' && b"\\/".starts_with(&self.pending))
        {
            return true;
        }
        let mut utf16 = [0; 2];
        let escaped: String = character
            .encode_utf16(&mut utf16)
            .iter()
            .map(|unit| format!("\\u{unit:04x}"))
            .collect();
        escaped
            .as_bytes()
            .get(..self.pending.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(&self.pending))
    }

    /// consume one byte; None refuses it. a character after which no match
    /// of an allowed length is reachable is refused: the string could never
    /// close. escapes feed the pattern the character they encode, as UTF-8,
    /// and count as one character.
    fn advance(&mut self, byte: u8) -> Option<StringStep> {
        if self.pending.is_empty()
            && byte != b'"'
            && self.max_length.is_some_and(|max| self.char_count >= max)
        {
            return None;
        }
        let step = self.decode_byte(byte)?;
        match step {
            StringStep::Character(character) => {
                self.char_count += 1;
                if let (Some(pattern), Some(state)) = (&self.pattern, &mut self.dfa_state) {
                    let mut buffer = [0; 4];
                    for &byte in character.encode_utf8(&mut buffer).as_bytes() {
                        *state = pattern.dfa.next_state(*state, byte);
                    }
                    if !pattern.can_finish(*state, self.char_count) {
                        return None;
                    }
                }
            }
            StringStep::End => {
                if self.min_length.is_some_and(|min| self.char_count < min) {
                    return None;
                }
                if let (Some(pattern), Some(state)) = (&self.pattern, self.dfa_state) {
                    let eoi_state = pattern.dfa.next_eoi_state(state);
                    if !pattern.dfa.is_match_state(eoi_state) {
                        return None;
                    }
                }
            }
            StringStep::Pending => {
                if !self.can_complete_pending() {
                    return None;
                }
            }
        }
        Some(step)
    }
}

/// the value of up to four validated hex digits
fn hex_value(digits: &[u8]) -> u32 {
    digits.iter().fold(0, |value, &byte| {
        (value << 4) | (byte as char).to_digit(16).expect("validated hex digit")
    })
}

/// state for parsing a JSON number value
#[derive(Debug, Clone)]
pub struct NumberState {
    /// buffer of digits seen so far
    buffer: String,
    /// whether we've seen a decimal point
    has_decimal: bool,
    /// whether we've seen an exponent (e/E)
    has_exponent: bool,
    /// whether we're expecting more digits after - or .
    expect_digit: bool,
    /// whether this is an integer (blocks decimal point and exponent)
    is_integer: bool,
}

impl Default for NumberState {
    fn default() -> Self {
        NumberState {
            buffer: String::new(),
            has_decimal: false,
            has_exponent: false,
            expect_digit: false,
            is_integer: false,
        }
    }
}

/// state for parsing a JSON boolean literal
#[derive(Debug, Clone)]
pub struct BooleanState {
    /// the literal we're matching ("true" or "false")
    target: &'static str,
    /// current position in the target
    position: usize,
}

/// state for parsing a JSON null literal
#[derive(Debug, Clone)]
pub struct NullState {
    /// current position in "null"
    position: usize,
}

/// syntax state within an array scope
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArraySyntaxState {
    /// expecting a value or ']' for empty/end
    ExpectValueOrEnd,
    /// after a comma, a value is mandatory
    ExpectValue,
    /// after a value, expecting ',' or ']'
    ExpectCommaOrEnd,
}

/// state for parsing a json array
#[derive(Debug, Clone)]
pub struct ArrayState {
    /// current syntax state
    pub syntax_state: ArraySyntaxState,
    /// schema for array items
    pub item_blueprint: Option<Arc<PropertyBlueprint>>,
    /// items consumed so far
    pub item_count: usize,
    /// min items required
    pub min_items: Option<usize>,
    /// max items allowed
    pub max_items: Option<usize>,
}

impl ArrayState {
    pub fn new(item_blueprint: Option<Arc<PropertyBlueprint>>) -> Self {
        ArrayState {
            syntax_state: ArraySyntaxState::ExpectValueOrEnd,
            item_blueprint,
            item_count: 0,
            min_items: None,
            max_items: None,
        }
    }

    pub fn with_constraints(
        item_blueprint: Option<Arc<PropertyBlueprint>>,
        min: Option<usize>,
        max: Option<usize>,
    ) -> Self {
        ArrayState {
            syntax_state: ArraySyntaxState::ExpectValueOrEnd,
            item_blueprint,
            item_count: 0,
            min_items: min,
            max_items: max,
        }
    }
}

/// state for parsing an enum value (one of a fixed set of allowed JSON values)
#[derive(Debug, Clone)]
pub struct EnumState {
    /// remaining valid candidates (as serialized JSON bytes)
    pub candidates: Vec<Vec<u8>>,
    /// how many bytes we have matched so far
    pub cursor: usize,
}

/// a scope on the parsing stack
#[derive(Debug, Clone)]
pub enum Scope {
    /// root scope before any value
    Root,
    /// inside a JSON object
    Object {
        /// the blueprint for this object (allowed keys, property types)
        blueprint: Arc<SchemaBlueprint>,
        /// current syntax state
        syntax_state: ObjectSyntaxState,
        /// buffer for the key being matched
        key_buffer: String,
        /// keys that have been used (for detecting duplicates)
        used_keys: Vec<String>,
        /// decoder for the key being read: escapes and UTF-8 sequences
        key_state: StringState,
        /// required keys that haven't been provided yet
        missing_required_keys: std::collections::HashSet<String>,
    },
    /// parsing a JSON string value
    String(StringState),
    /// parsing a JSON number value
    Number(NumberState),
    /// parsing a JSON boolean literal (true/false)
    Boolean(BooleanState),
    /// parsing a JSON null literal
    Null(NullState),
    /// inside a JSON array
    Array(ArrayState),
    /// parsing an enum value (one of a fixed set of allowed JSON values)
    Enum(EnumState),
}

/// a JSON schema constraint engine using stack-based scopes
pub struct JsonEngine {
    vocab: Arc<HashMap<usize, Vec<u8>>>,
    vocab_size: usize,
    root_blueprint: Arc<SchemaBlueprint>,
    root_value_blueprint: Arc<PropertyBlueprint>,
    stack: Vec<Scope>,
    finished: bool,
    dead: bool,
}

impl JsonEngine {
    /// create a new JsonEngine with the given vocabulary and schema
    pub fn new(vocabulary: Vec<Vec<u8>>, schema_str: &str) -> Result<Self, ConstraintError> {
        let schema: Value = serde_json::from_str(schema_str).map_err(|e| {
            ConstraintError::CompilationError(format!("Invalid JSON schema: {}", e))
        })?;

        let root_blueprint = Arc::new(SchemaBlueprint::from_value(&schema)?);
        let root_value_blueprint = Arc::new(PropertyBlueprint::from_value("_root", &schema)?);

        let vocab_size = vocabulary.len();
        let mut vocab = HashMap::with_capacity(vocab_size);
        for (id, bytes) in vocabulary.into_iter().enumerate() {
            vocab.insert(id, bytes);
        }

        let stack = vec![Scope::Root];

        Ok(JsonEngine {
            vocab: Arc::new(vocab),
            vocab_size,
            root_blueprint,
            root_value_blueprint,
            stack,
            finished: false,
            dead: false,
        })
    }

    #[inline]
    fn is_whitespace(byte: u8) -> bool {
        matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
    }

    /// check if a byte ends a number (separator or structural char)
    #[inline]
    fn is_number_terminator(byte: u8) -> bool {
        matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
    }

    #[inline]
    fn number_is_complete(state: &NumberState) -> bool {
        state.buffer.chars().any(|c| c.is_ascii_digit()) && !state.expect_digit
    }

    /// helper to update parent scope state after a value is complete
    fn update_parent_after_value(stack: &mut Vec<Scope>) {
        if let Some(parent) = stack.last_mut() {
            match parent {
                Scope::Object { syntax_state, .. } => {
                    *syntax_state = ObjectSyntaxState::ExpectCommaOrEnd;
                }
                Scope::Array(state) => {
                    state.syntax_state = ArraySyntaxState::ExpectCommaOrEnd;
                }
                _ => {}
            }
        }
    }

    /// helper function: push the appropriate value scope based on schema type
    fn push_value_scope(
        stack: &mut Vec<Scope>,
        item_bp: &Option<Arc<PropertyBlueprint>>,
        byte: u8,
    ) -> bool {
        // priority: check for enum values first
        if let Some(bp) = item_bp {
            if let Some(ref enum_vals) = bp.enum_values {
                // filter candidates that start with this byte
                let matching: Vec<Vec<u8>> = enum_vals
                    .iter()
                    .filter(|v| !v.is_empty() && v[0] == byte)
                    .cloned()
                    .collect();

                if !matching.is_empty() {
                    stack.push(Scope::Enum(EnumState {
                        candidates: matching,
                        cursor: 1, // matched the first byte
                    }));
                    return true;
                }
                return false; // byte doesn't match any enum candidate
            }
        }

        // get the allowed types
        let schema_types = item_bp
            .as_ref()
            .map(|bp| bp.schema_types.clone())
            .unwrap_or_else(|| vec![SchemaType::Any]);

        // helper function: check if a byte matches a type
        fn byte_matches_type(byte: u8, schema_type: &SchemaType) -> bool {
            match schema_type {
                SchemaType::String => byte == b'"',
                SchemaType::Number | SchemaType::Integer => byte.is_ascii_digit() || byte == b'-',
                SchemaType::Boolean => byte == b't' || byte == b'f',
                SchemaType::Null => byte == b'n',
                SchemaType::Object => byte == b'{',
                SchemaType::Array => byte == b'[',
                SchemaType::Any => {
                    byte == b'"'
                        || byte.is_ascii_digit()
                        || byte == b'-'
                        || byte == b't'
                        || byte == b'f'
                        || byte == b'n'
                        || byte == b'{'
                        || byte == b'['
                }
            }
        }

        // helper function: push a scope for a specific type
        fn push_scope_for_type(
            stack: &mut Vec<Scope>,
            schema_type: &SchemaType,
            byte: u8,
            item_bp: &Option<Arc<PropertyBlueprint>>,
        ) -> bool {
            match schema_type {
                SchemaType::String => {
                    if byte == b'"' {
                        // pass constraints from item blueprint
                        let string_state = if let Some(bp) = item_bp {
                            StringState::with_pattern_and_constraints(
                                bp.pattern.as_ref(),
                                bp.min_length,
                                bp.max_length,
                            )
                        } else {
                            StringState::new_started()
                        };
                        stack.push(Scope::String(string_state));
                        return true;
                    }
                }
                SchemaType::Number | SchemaType::Integer => {
                    if byte.is_ascii_digit() || byte == b'-' {
                        let mut ns = NumberState::default();
                        ns.buffer.push(byte as char);
                        ns.expect_digit = byte == b'-';
                        ns.is_integer = *schema_type == SchemaType::Integer;
                        stack.push(Scope::Number(ns));
                        return true;
                    }
                }
                SchemaType::Boolean => {
                    if byte == b't' {
                        stack.push(Scope::Boolean(BooleanState {
                            target: "true",
                            position: 1,
                        }));
                        return true;
                    }
                    if byte == b'f' {
                        stack.push(Scope::Boolean(BooleanState {
                            target: "false",
                            position: 1,
                        }));
                        return true;
                    }
                }
                SchemaType::Null => {
                    if byte == b'n' {
                        stack.push(Scope::Null(NullState { position: 1 }));
                        return true;
                    }
                }
                SchemaType::Object => {
                    if byte == b'{' {
                        let (nested_blueprint, missing_req) = if let Some(bp) = item_bp {
                            if let Some(obp) = &bp.object_blueprint {
                                (Arc::clone(obp), obp.required.clone())
                            } else {
                                let empty = Arc::new(SchemaBlueprint {
                                    root_type: SchemaType::Object,
                                    properties: std::collections::HashMap::new(),
                                    required: std::collections::HashSet::new(),
                                    allowed_keys: Vec::new(),
                                });
                                (empty, std::collections::HashSet::new())
                            }
                        } else {
                            let empty = Arc::new(SchemaBlueprint {
                                root_type: SchemaType::Object,
                                properties: std::collections::HashMap::new(),
                                required: std::collections::HashSet::new(),
                                allowed_keys: Vec::new(),
                            });
                            (empty, std::collections::HashSet::new())
                        };
                        stack.push(Scope::Object {
                            blueprint: nested_blueprint,
                            syntax_state: ObjectSyntaxState::ExpectKeyOrEnd,
                            key_buffer: String::new(),
                            used_keys: Vec::new(),
                            key_state: StringState::new_started(),
                            missing_required_keys: missing_req,
                        });
                        return true;
                    }
                }
                SchemaType::Array => {
                    if byte == b'[' {
                        let nested_items = item_bp.as_ref().and_then(|bp| bp.items.clone());
                        let min_items = item_bp.as_ref().and_then(|bp| bp.min_items);
                        let max_items = item_bp.as_ref().and_then(|bp| bp.max_items);
                        let arr_state =
                            ArrayState::with_constraints(nested_items, min_items, max_items);
                        stack.push(Scope::Array(arr_state));
                        return true;
                    }
                }
                SchemaType::Any => {
                    // for any, try all types
                    if byte == b'"' {
                        stack.push(Scope::String(StringState::new_started()));
                        return true;
                    }
                    if byte.is_ascii_digit() || byte == b'-' {
                        let mut ns = NumberState::default();
                        ns.buffer.push(byte as char);
                        ns.expect_digit = byte == b'-';
                        stack.push(Scope::Number(ns));
                        return true;
                    }
                    if byte == b't' {
                        stack.push(Scope::Boolean(BooleanState {
                            target: "true",
                            position: 1,
                        }));
                        return true;
                    }
                    if byte == b'f' {
                        stack.push(Scope::Boolean(BooleanState {
                            target: "false",
                            position: 1,
                        }));
                        return true;
                    }
                    if byte == b'n' {
                        stack.push(Scope::Null(NullState { position: 1 }));
                        return true;
                    }
                    if byte == b'{' {
                        let empty = Arc::new(SchemaBlueprint {
                            root_type: SchemaType::Object,
                            properties: std::collections::HashMap::new(),
                            required: std::collections::HashSet::new(),
                            allowed_keys: Vec::new(),
                        });
                        stack.push(Scope::Object {
                            blueprint: empty,
                            syntax_state: ObjectSyntaxState::ExpectKeyOrEnd,
                            key_buffer: String::new(),
                            used_keys: Vec::new(),
                            key_state: StringState::new_started(),
                            missing_required_keys: std::collections::HashSet::new(),
                        });
                        return true;
                    }
                    if byte == b'[' {
                        let arr_state = ArrayState::new(None);
                        stack.push(Scope::Array(arr_state));
                        return true;
                    }
                }
            }
            false
        }

        // find which types match this byte
        for schema_type in &schema_types {
            if byte_matches_type(byte, schema_type) {
                // a digit opens one number, which keeps the fraction and
                // exponent "number" allows even when "integer" is listed first
                let schema_type = if *schema_type == SchemaType::Integer
                    && schema_types.contains(&SchemaType::Number)
                {
                    &SchemaType::Number
                } else {
                    schema_type
                };
                return push_scope_for_type(stack, schema_type, byte, item_bp);
            }
        }

        false
    }

    /// check if a token is valid from the current state
    fn validate_token(&self, token_bytes: &[u8]) -> bool {
        if token_bytes.is_empty() || self.dead || self.finished {
            return false;
        }

        let mut temp_stack = self.stack.clone();
        let mut temp_finished = self.finished;

        for &byte in token_bytes {
            if !Self::validate_byte_static(
                &self.root_blueprint,
                &self.root_value_blueprint,
                &mut temp_stack,
                &mut temp_finished,
                byte,
            ) {
                return false;
            }
        }

        true
    }

    /// advance the actual state with a byte
    fn advance_byte(&mut self, byte: u8) -> Result<(), ConstraintError> {
        let valid = Self::validate_byte_static(
            &self.root_blueprint,
            &self.root_value_blueprint,
            &mut self.stack,
            &mut self.finished,
            byte,
        );

        if !valid {
            self.dead = true;
            return Err(ConstraintError::InvalidState(format!(
                "Invalid byte '{}' (0x{:02x}) in current state",
                byte as char, byte
            )));
        }
        Ok(())
    }

    /// static version of byte validation
    fn validate_byte_static(
        root_blueprint: &Arc<SchemaBlueprint>,
        root_value_blueprint: &Arc<PropertyBlueprint>,
        stack: &mut Vec<Scope>,
        finished: &mut bool,
        byte: u8,
    ) -> bool {
        loop {
            let scope = match stack.last_mut() {
                Some(s) => s,
                None => return false,
            };

            match scope {
                Scope::Root => {
                    if Self::is_whitespace(byte) {
                        return true;
                    }

                    if root_blueprint.root_type == SchemaType::Object && byte == b'{' {
                        *scope = Scope::Object {
                            blueprint: Arc::clone(root_blueprint),
                            syntax_state: ObjectSyntaxState::ExpectKeyOrEnd,
                            key_buffer: String::new(),
                            used_keys: Vec::new(),
                            key_state: StringState::new_started(),
                            missing_required_keys: root_blueprint.required.clone(),
                        };
                        return true;
                    }

                    let mut value_stack = Vec::new();
                    if Self::push_value_scope(
                        &mut value_stack,
                        &Some(Arc::clone(root_value_blueprint)),
                        byte,
                    ) {
                        if let Some(value_scope) = value_stack.pop() {
                            *scope = value_scope;
                            return true;
                        }
                    }
                    return false;
                }

                Scope::Object {
                    blueprint,
                    syntax_state,
                    key_buffer,
                    used_keys,
                    key_state,
                    missing_required_keys,
                } => {
                    match syntax_state {
                        ObjectSyntaxState::ExpectKeyOrEnd | ObjectSyntaxState::ExpectKey => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }
                            if byte == b'"' {
                                // only open a new key if at least one unused key exists
                                let has_unused_key = blueprint
                                    .allowed_keys
                                    .iter()
                                    .any(|k| !used_keys.contains(k));
                                if has_unused_key {
                                    *syntax_state = ObjectSyntaxState::InKey;
                                    key_buffer.clear();
                                    *key_state = StringState::new_started();
                                    return true;
                                }
                                return false;
                            }
                            // a trailing comma cannot be followed by '}'
                            if byte == b'}' && *syntax_state == ObjectSyntaxState::ExpectKeyOrEnd {
                                // only allow closing if all required keys have been provided
                                if !missing_required_keys.is_empty() {
                                    return false;
                                }
                                stack.pop();
                                if stack.is_empty() {
                                    *finished = true;
                                } else {
                                    Self::update_parent_after_value(stack);
                                }
                                return true;
                            }
                            return false;
                        }

                        ObjectSyntaxState::InKey => {
                            // keys are matched by the characters their JSON
                            // text decodes to, as string values are
                            match key_state.advance(byte) {
                                Some(StringStep::End) => {
                                    if !blueprint.is_key_allowed(key_buffer)
                                        || used_keys.contains(key_buffer)
                                    {
                                        return false;
                                    }
                                    *syntax_state = ObjectSyntaxState::ExpectColon;
                                    return true;
                                }
                                Some(StringStep::Character(character)) => {
                                    key_buffer.push(character)
                                }
                                Some(StringStep::Pending) => {}
                                None => return false,
                            }
                            // only accept a prefix of an unused key that an
                            // unfinished escape or UTF-8 sequence can still spell
                            return blueprint.allowed_keys.iter().any(|key| {
                                !used_keys.contains(key)
                                    && key.strip_prefix(key_buffer.as_str()).is_some_and(
                                        |remaining| key_state.can_complete_key(remaining),
                                    )
                            });
                        }

                        ObjectSyntaxState::ExpectColon => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }
                            if byte == b':' {
                                used_keys.push(key_buffer.clone());
                                // remove this key from missing required keys
                                missing_required_keys.remove(key_buffer.as_str());
                                *syntax_state = ObjectSyntaxState::ExpectValue;
                                return true;
                            }
                            return false;
                        }

                        ObjectSyntaxState::ExpectValue => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }

                            let key = key_buffer.clone();

                            // priority: check for enum values first
                            if let Some(prop) = blueprint.get_property(&key) {
                                if let Some(ref enum_vals) = prop.enum_values {
                                    // filter candidates that start with this byte
                                    let matching: Vec<Vec<u8>> = enum_vals
                                        .iter()
                                        .filter(|v| !v.is_empty() && v[0] == byte)
                                        .cloned()
                                        .collect();

                                    if !matching.is_empty() {
                                        *syntax_state = ObjectSyntaxState::ExpectCommaOrEnd;
                                        stack.push(Scope::Enum(EnumState {
                                            candidates: matching,
                                            cursor: 1,
                                        }));
                                        return true;
                                    }
                                    return false; // byte doesn't match any enum candidate
                                }
                            }

                            // get allowed types for this property
                            let schema_types = blueprint
                                .get_property(&key)
                                .map(|p| p.schema_types.clone())
                                .unwrap_or_else(|| vec![SchemaType::Any]);

                            // helper to check if byte matches a type
                            fn matches_type(byte: u8, t: &SchemaType) -> bool {
                                match t {
                                    SchemaType::String => byte == b'"',
                                    SchemaType::Number | SchemaType::Integer => {
                                        byte.is_ascii_digit() || byte == b'-'
                                    }
                                    SchemaType::Boolean => byte == b't' || byte == b'f',
                                    SchemaType::Null => byte == b'n',
                                    SchemaType::Object => byte == b'{',
                                    SchemaType::Array => byte == b'[',
                                    SchemaType::Any => {
                                        matches_type(byte, &SchemaType::String)
                                            || matches_type(byte, &SchemaType::Number)
                                            || matches_type(byte, &SchemaType::Boolean)
                                            || matches_type(byte, &SchemaType::Null)
                                            || matches_type(byte, &SchemaType::Object)
                                            || matches_type(byte, &SchemaType::Array)
                                    }
                                }
                            }

                            // Find which allowed type matches the byte
                            let matched_type = schema_types.iter().find(|t| matches_type(byte, t));

                            if let Some(prop_type) = matched_type {
                                *syntax_state = ObjectSyntaxState::ExpectCommaOrEnd;
                                match prop_type {
                                    SchemaType::String => {
                                        // get pattern and length constraints
                                        let string_state =
                                            if let Some(prop) = blueprint.get_property(&key) {
                                                StringState::with_pattern_and_constraints(
                                                    prop.pattern.as_ref(),
                                                    prop.min_length,
                                                    prop.max_length,
                                                )
                                            } else {
                                                StringState::new_started()
                                            };
                                        stack.push(Scope::String(string_state));
                                        return true;
                                    }
                                    SchemaType::Number | SchemaType::Integer => {
                                        let mut ns = NumberState::default();
                                        ns.buffer.push(byte as char);
                                        ns.expect_digit = byte == b'-';
                                        // as in push_value_scope, "number" anywhere in
                                        // the list keeps fractions and exponents
                                        ns.is_integer = *prop_type == SchemaType::Integer
                                            && !schema_types.contains(&SchemaType::Number);
                                        stack.push(Scope::Number(ns));
                                        return true;
                                    }
                                    SchemaType::Boolean => {
                                        if byte == b't' {
                                            stack.push(Scope::Boolean(BooleanState {
                                                target: "true",
                                                position: 1,
                                            }));
                                        } else {
                                            stack.push(Scope::Boolean(BooleanState {
                                                target: "false",
                                                position: 1,
                                            }));
                                        }
                                        return true;
                                    }
                                    SchemaType::Null => {
                                        stack.push(Scope::Null(NullState { position: 1 }));
                                        return true;
                                    }
                                    SchemaType::Object => {
                                        if let Some(prop) = blueprint.get_property(&key) {
                                            let (nested_blueprint, missing_req) = if let Some(obp) =
                                                &prop.object_blueprint
                                            {
                                                (Arc::clone(obp), obp.required.clone())
                                            } else {
                                                let empty = Arc::new(SchemaBlueprint {
                                                    root_type: SchemaType::Object,
                                                    properties: std::collections::HashMap::new(),
                                                    required: std::collections::HashSet::new(),
                                                    allowed_keys: Vec::new(),
                                                });
                                                (empty, std::collections::HashSet::new())
                                            };
                                            stack.push(Scope::Object {
                                                blueprint: nested_blueprint,
                                                syntax_state: ObjectSyntaxState::ExpectKeyOrEnd,
                                                key_buffer: String::new(),
                                                used_keys: Vec::new(),
                                                key_state: StringState::new_started(),
                                                missing_required_keys: missing_req,
                                            });
                                        }
                                        return true;
                                    }
                                    SchemaType::Array => {
                                        let prop = blueprint.get_property(&key);
                                        let items = prop.as_ref().and_then(|p| p.items.clone());
                                        let min_items = prop.as_ref().and_then(|p| p.min_items);
                                        let max_items = prop.as_ref().and_then(|p| p.max_items);
                                        let arr_state = ArrayState::with_constraints(
                                            items, min_items, max_items,
                                        );
                                        stack.push(Scope::Array(arr_state));
                                        return true;
                                    }
                                    SchemaType::Any => {
                                        // Handled by specific type match above
                                        return false;
                                    }
                                }
                            }
                            return false;
                        }

                        ObjectSyntaxState::ExpectCommaOrEnd => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }
                            if byte == b',' {
                                // only accept comma if there are unused keys to follow
                                let has_unused_key = blueprint
                                    .allowed_keys
                                    .iter()
                                    .any(|k| !used_keys.contains(k));
                                if has_unused_key {
                                    *syntax_state = ObjectSyntaxState::ExpectKey;
                                    return true;
                                }
                                return false;
                            }
                            if byte == b'}' {
                                // only allow closing if all required keys have been provided
                                if !missing_required_keys.is_empty() {
                                    return false;
                                }
                                stack.pop();
                                if stack.is_empty() {
                                    *finished = true;
                                } else {
                                    Self::update_parent_after_value(stack);
                                }
                                return true;
                            }
                            return false;
                        }
                    }
                }

                Scope::String(state) => {
                    // the string state checks minLength, maxLength and the
                    // pattern as it decodes
                    match state.advance(byte) {
                        None => return false,
                        Some(StringStep::End) => {
                            // string complete
                            stack.pop();
                            if stack.is_empty() {
                                *finished = true;
                            }
                            return true;
                        }
                        Some(_) => return true,
                    }
                }

                Scope::Number(state) => {
                    // Check for terminators first
                    if Self::is_number_terminator(byte) {
                        // Number is complete if we have at least one digit
                        if Self::number_is_complete(state) {
                            stack.pop();
                            // Re-process this byte with parent scope
                            continue;
                        }
                        return false;
                    }

                    // Handle number characters
                    if byte.is_ascii_digit() {
                        // json forbids leading zeros in the integer part
                        if !state.has_decimal
                            && !state.has_exponent
                            && matches!(state.buffer.as_str(), "0" | "-0")
                        {
                            return false;
                        }
                        state.buffer.push(byte as char);
                        state.expect_digit = false;
                        return true;
                    }
                    // '.' and 'e' must follow a digit, so "-.5" and "1.e5" are rejected
                    if byte == b'.'
                        && !state.expect_digit
                        && !state.has_decimal
                        && !state.has_exponent
                        && !state.is_integer
                    {
                        state.buffer.push('.');
                        state.has_decimal = true;
                        state.expect_digit = true;
                        return true;
                    }
                    if (byte == b'e' || byte == b'E')
                        && !state.expect_digit
                        && !state.has_exponent
                        && !state.is_integer
                    {
                        state.buffer.push(byte as char);
                        state.has_exponent = true;
                        state.expect_digit = true;
                        return true;
                    }
                    if (byte == b'+' || byte == b'-')
                        && state.buffer.ends_with(|c| c == 'e' || c == 'E')
                    {
                        state.buffer.push(byte as char);
                        state.expect_digit = true;
                        return true;
                    }
                    return false;
                }

                Scope::Boolean(state) => {
                    let expected_byte = state.target.as_bytes().get(state.position);
                    if expected_byte == Some(&byte) {
                        state.position += 1;
                        if state.position == state.target.len() {
                            // Boolean complete
                            stack.pop();
                            if stack.is_empty() {
                                *finished = true;
                            }
                        }
                        return true;
                    }
                    return false;
                }

                Scope::Null(state) => {
                    const NULL_LITERAL: &[u8] = b"null";
                    let expected_byte = NULL_LITERAL.get(state.position);
                    if expected_byte == Some(&byte) {
                        state.position += 1;
                        if state.position == 4 {
                            // Null complete
                            stack.pop();
                            if stack.is_empty() {
                                *finished = true;
                            }
                        }
                        return true;
                    }
                    return false;
                }

                Scope::Array(state) => {
                    match state.syntax_state {
                        ArraySyntaxState::ExpectValueOrEnd | ArraySyntaxState::ExpectValue => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }
                            if byte == b']' {
                                // a trailing comma cannot be followed by ']'
                                if state.syntax_state == ArraySyntaxState::ExpectValue {
                                    return false;
                                }
                                // check minitems before allowing close
                                if let Some(min) = state.min_items {
                                    if state.item_count < min {
                                        return false;
                                    }
                                }
                                stack.pop();
                                if stack.is_empty() {
                                    *finished = true;
                                } else {
                                    Self::update_parent_after_value(stack);
                                }
                                return true;
                            }
                            // check maxitems before allowing new item
                            if let Some(max) = state.max_items {
                                if state.item_count >= max {
                                    return false;
                                }
                            }
                            // push scope for item
                            state.syntax_state = ArraySyntaxState::ExpectCommaOrEnd;
                            state.item_count += 1;
                            let item_bp = state.item_blueprint.clone();
                            return Self::push_value_scope(stack, &item_bp, byte);
                        }

                        ArraySyntaxState::ExpectCommaOrEnd => {
                            if Self::is_whitespace(byte) {
                                return true;
                            }
                            if byte == b',' {
                                // check maxitems before allowing more
                                if let Some(max) = state.max_items {
                                    if state.item_count >= max {
                                        return false;
                                    }
                                }
                                state.syntax_state = ArraySyntaxState::ExpectValue;
                                return true;
                            }
                            if byte == b']' {
                                // check minitems
                                if let Some(min) = state.min_items {
                                    if state.item_count < min {
                                        return false;
                                    }
                                }
                                stack.pop();
                                if stack.is_empty() {
                                    *finished = true;
                                } else {
                                    Self::update_parent_after_value(stack);
                                }
                                return true;
                            }
                            return false;
                        }
                    }
                }

                Scope::Enum(state) => {
                    // check if any candidate expects this byte at current cursor
                    let cursor = state.cursor;

                    // check if we have a complete match at current cursor
                    // if so, pop on delimiters
                    let has_complete = state.candidates.iter().any(|c| c.len() == cursor);

                    if has_complete {
                        // allow delimiters that would end this value
                        if byte == b',' || byte == b'}' || byte == b']' || Self::is_whitespace(byte)
                        {
                            // pop the enum scope
                            stack.pop();
                            if stack.is_empty() {
                                // a root value may only be followed by whitespace
                                *finished = true;
                                return Self::is_whitespace(byte);
                            }
                            // reprocess this byte with the parent scope so its own
                            // comma, close, minitems and required-key rules apply
                            Self::update_parent_after_value(stack);
                            continue;
                        }
                    }

                    // check if byte matches any remaining candidate at current cursor
                    let new_candidates: Vec<Vec<u8>> = state
                        .candidates
                        .iter()
                        .filter(|c| c.len() > cursor && c[cursor] == byte)
                        .cloned()
                        .collect();

                    if new_candidates.is_empty() {
                        return false;
                    }

                    // advance cursor and update candidates
                    state.cursor += 1;
                    state.candidates = new_candidates;

                    // check if we just completed a match
                    let new_cursor = state.cursor;
                    if state.candidates.iter().any(|c| c.len() == new_cursor) {
                        // have a complete match, pop and update parent
                        stack.pop();
                        if stack.is_empty() {
                            *finished = true;
                        }
                    }

                    return true;
                }
            }
        }
    }
}

impl ConstraintEngine for JsonEngine {
    fn reset(&mut self) {
        self.stack = vec![Scope::Root];
        self.finished = false;
        self.dead = false;
    }

    fn get_valid_tokens(&self) -> Vec<usize> {
        if self.dead || self.finished {
            return Vec::new();
        }

        let mut valid_tokens = Vec::new();

        for token_id in 0..self.vocab_size {
            if let Some(bytes) = self.vocab.get(&token_id) {
                if !bytes.is_empty() && self.validate_token(bytes) {
                    valid_tokens.push(token_id);
                }
            }
        }

        valid_tokens
    }

    fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError> {
        let bytes = self
            .vocab
            .get(&token_id)
            .ok_or_else(|| ConstraintError::InvalidTokenId {
                token_id,
                vocab_size: self.vocab_size,
            })?
            .clone();

        for &byte in &bytes {
            self.advance_byte(byte)?;
        }

        Ok(())
    }

    fn is_finished(&self) -> bool {
        if self.finished && self.stack.is_empty() {
            return true;
        }

        // a root number or enum value has no closing delimiter, so it is complete
        // as soon as its buffer is a full value
        match self.stack.as_slice() {
            [Scope::Number(state)] => Self::number_is_complete(state),
            [Scope::Enum(state)] => state.candidates.iter().any(|c| c.len() == state.cursor),
            _ => false,
        }
    }

    fn is_dead(&self) -> bool {
        self.dead
    }

    fn current_state_id(&self) -> u32 {
        self.stack.len() as u32
    }

    fn clone_box(&self) -> Box<dyn ConstraintEngine> {
        Box::new(JsonEngine {
            vocab: Arc::clone(&self.vocab),
            vocab_size: self.vocab_size,
            root_blueprint: Arc::clone(&self.root_blueprint),
            root_value_blueprint: Arc::clone(&self.root_value_blueprint),
            stack: self.stack.clone(),
            finished: self.finished,
            dead: self.dead,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_typed_vocab() -> Vec<Vec<u8>> {
        vec![
            b"{".to_vec(),            // 0
            b"}".to_vec(),            // 1
            b"\"".to_vec(),           // 2
            b":".to_vec(),            // 3
            b",".to_vec(),            // 4
            b" ".to_vec(),            // 5
            b"\"name\"".to_vec(),     // 6 - valid key (string type)
            b"\"age\"".to_vec(),      // 7 - valid key (number type)
            b"\"is_admin\"".to_vec(), // 8 - valid key (boolean type)
            b"\"John\"".to_vec(),     // 9 - string value
            b"25".to_vec(),           // 10 - number value
            b"true".to_vec(),         // 11 - boolean value
            b"false".to_vec(),        // 12 - boolean value
            b"null".to_vec(),         // 13 - null value
            b"t".to_vec(),            // 14 - prefix of true
            b"f".to_vec(),            // 15 - prefix of false
            b"n".to_vec(),            // 16 - prefix of null/name
            b"1".to_vec(),            // 17 - single digit
            b"0".to_vec(),            // 18 - zero
            b"-".to_vec(),            // 19 - minus
            b".".to_vec(),            // 20 - decimal point
            b"123".to_vec(),          // 21 - multi-digit number
        ]
    }

    fn typed_schema() -> &'static str {
        r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "is_admin": {"type": "boolean"}}}"#
    }

    #[test]
    fn test_string_type_requires_quotes() {
        let vocab = make_typed_vocab();
        let mut engine = JsonEngine::new(vocab, typed_schema()).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(6).unwrap(); // '"name"'
        engine.advance(3).unwrap(); // ':'

        let valid = engine.get_valid_tokens();

        // for string type, must have quotes
        assert!(valid.contains(&2)); // '"' is valid
        assert!(!valid.contains(&10)); // '25' (unquoted number) is NOT valid
        assert!(!valid.contains(&11)); // 'true' is NOT valid
    }

    #[test]
    fn test_number_type_rejects_quotes() {
        let vocab = make_typed_vocab();
        let mut engine = JsonEngine::new(vocab, typed_schema()).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(7).unwrap(); // '"age"'
        engine.advance(3).unwrap(); // ':'

        let valid = engine.get_valid_tokens();

        // for number type, must NOT have quotes
        assert!(!valid.contains(&2)); // '"' is NOT valid
        assert!(valid.contains(&17)); // '1' is valid
        assert!(valid.contains(&18)); // '0' is valid
        assert!(valid.contains(&19)); // '-' is valid
        assert!(valid.contains(&21)); // '123' is valid
    }

    #[test]
    fn test_boolean_type_only_true_false() {
        let vocab = make_typed_vocab();
        let mut engine = JsonEngine::new(vocab, typed_schema()).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(8).unwrap(); // '"is_admin"'
        engine.advance(3).unwrap(); // ':'

        let valid = engine.get_valid_tokens();

        // for boolean type, only true/false
        assert!(valid.contains(&11)); // 'true' is valid
        assert!(valid.contains(&12)); // 'false' is valid
        assert!(valid.contains(&14)); // 't' (prefix of true) is valid
        assert!(valid.contains(&15)); // 'f' (prefix of false) is valid
        assert!(!valid.contains(&2)); // '"' is NOT valid
        assert!(!valid.contains(&17)); // '1' is NOT valid
    }

    #[test]
    fn test_complete_typed_object() {
        let vocab = make_typed_vocab();
        let mut engine = JsonEngine::new(vocab, typed_schema()).unwrap();

        // {"name":"John","age":25,"is_admin":true}
        engine.advance(0).unwrap(); // '{'
        engine.advance(6).unwrap(); // '"name"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(9).unwrap(); // '"John"'
        engine.advance(4).unwrap(); // ','
        engine.advance(7).unwrap(); // '"age"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(10).unwrap(); // '25'
        engine.advance(4).unwrap(); // ','
        engine.advance(8).unwrap(); // '"is_admin"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(11).unwrap(); // 'true'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_number_with_decimal() {
        let vocab = vec![
            b"{".to_vec(),
            b"}".to_vec(),
            b"\"age\"".to_vec(),
            b":".to_vec(),
            b"25".to_vec(),
            b".".to_vec(),
            b"5".to_vec(),
        ];
        let schema = r#"{"type": "object", "properties": {"age": {"type": "number"}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"age"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '25'
        engine.advance(5).unwrap(); // '.'
        engine.advance(6).unwrap(); // '5'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_reject_wrong_type() {
        let vocab = make_typed_vocab();
        let mut engine = JsonEngine::new(vocab, typed_schema()).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(7).unwrap(); // '"age"'
        engine.advance(3).unwrap(); // ':'

        // try to use a string for a number field - should fail
        let result = engine.advance(2); // trying to start a string for number field
        assert!(result.is_err());
    }

    #[test]
    fn test_array_of_strings() {
        let vocab = vec![
            b"{".to_vec(),        // 0
            b"}".to_vec(),        // 1
            b"[".to_vec(),        // 2
            b"]".to_vec(),        // 3
            b"\"tags\"".to_vec(), // 4
            b":".to_vec(),        // 5
            b",".to_vec(),        // 6
            b"\"tag1\"".to_vec(), // 7
            b"\"tag2\"".to_vec(), // 8
        ];
        let schema = r#"{"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"tags":["tag1","tag2"]}
        engine.advance(0).unwrap(); // '{'
        engine.advance(4).unwrap(); // '"tags"'
        engine.advance(5).unwrap(); // ':'
        engine.advance(2).unwrap(); // '['
        engine.advance(7).unwrap(); // '"tag1"'
        engine.advance(6).unwrap(); // ','
        engine.advance(8).unwrap(); // '"tag2"'
        engine.advance(3).unwrap(); // ']'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_empty_array() {
        let vocab = vec![
            b"{".to_vec(),        // 0
            b"}".to_vec(),        // 1
            b"[".to_vec(),        // 2
            b"]".to_vec(),        // 3
            b"\"tags\"".to_vec(), // 4
            b":".to_vec(),        // 5
        ];
        let schema = r#"{"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"tags":[]}
        engine.advance(0).unwrap(); // '{'
        engine.advance(4).unwrap(); // '"tags"'
        engine.advance(5).unwrap(); // ':'
        engine.advance(2).unwrap(); // '['
        engine.advance(3).unwrap(); // ']'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_array_rejects_wrong_item_type() {
        let vocab = vec![
            b"{".to_vec(),        // 0
            b"}".to_vec(),        // 1
            b"[".to_vec(),        // 2
            b"]".to_vec(),        // 3
            b"\"tags\"".to_vec(), // 4
            b":".to_vec(),        // 5
            b"123".to_vec(),      // 6 - number, should fail
        ];
        let schema = r#"{"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(4).unwrap(); // '"tags"'
        engine.advance(5).unwrap(); // ':'
        engine.advance(2).unwrap(); // '['

        // try to add a number to a string array - should fail
        let result = engine.advance(6);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_field_rejects_empty_object() {
        let vocab = vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b"\"b\"".to_vec(), // 3
            b":".to_vec(),     // 4
            b"1".to_vec(),     // 5
            b",".to_vec(),     // 6
        ];
        let schema = r#"{"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a"]}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '{'

        // try to close empty object - should fail because "a" is required
        let result = engine.advance(1); // '}'
        assert!(result.is_err());
    }

    #[test]
    fn test_required_field_rejects_missing_key() {
        let vocab = vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b"\"b\"".to_vec(), // 3
            b":".to_vec(),     // 4
            b"1".to_vec(),     // 5
            b",".to_vec(),     // 6
        ];
        let schema = r#"{"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a"]}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"b":1} - should fail because "a" is required
        engine.advance(0).unwrap(); // '{'
        engine.advance(3).unwrap(); // '"b"'
        engine.advance(4).unwrap(); // ':'
        engine.advance(5).unwrap(); // '1'

        // try to close - should fail because "a" is required
        let result = engine.advance(1); // '}'
        assert!(result.is_err());
    }

    #[test]
    fn test_required_field_accepts_complete_object() {
        let vocab = vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b"\"b\"".to_vec(), // 3
            b":".to_vec(),     // 4
            b"1".to_vec(),     // 5
            b",".to_vec(),     // 6
        ];
        let schema = r#"{"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a"]}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"b":1,"a":1} - should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(3).unwrap(); // '"b"'
        engine.advance(4).unwrap(); // ':'
        engine.advance(5).unwrap(); // '1'
        engine.advance(6).unwrap(); // ','
        engine.advance(2).unwrap(); // '"a"'
        engine.advance(4).unwrap(); // ':'
        engine.advance(5).unwrap(); // '1'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_required_field_accepts_only_required() {
        let vocab = vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b":".to_vec(),     // 3
            b"1".to_vec(),     // 4
        ];
        let schema = r#"{"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a"]}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"a":1} - should succeed (only required key)
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"a"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '1'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_enum_accepts_valid_value() {
        // enum values are serialized as JSON, so "red" becomes b'"red"'
        let vocab = vec![
            b"{".to_vec(),         // 0
            b"}".to_vec(),         // 1
            b"\"color\"".to_vec(), // 2
            b":".to_vec(),         // 3
            b"\"red\"".to_vec(),   // 4
            b"\"blue\"".to_vec(),  // 5
            b"\"green\"".to_vec(), // 6
        ];
        let schema = r#"{"type": "object", "properties": {"color": {"enum": ["red", "blue"]}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"color":"red"} - should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"color"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '"red"'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_enum_accepts_second_value() {
        let vocab = vec![
            b"{".to_vec(),         // 0
            b"}".to_vec(),         // 1
            b"\"color\"".to_vec(), // 2
            b":".to_vec(),         // 3
            b"\"red\"".to_vec(),   // 4
            b"\"blue\"".to_vec(),  // 5
        ];
        let schema = r#"{"type": "object", "properties": {"color": {"enum": ["red", "blue"]}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"color":"blue"} - should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"color"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(5).unwrap(); // '"blue"'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_enum_rejects_invalid_value() {
        let vocab = vec![
            b"{".to_vec(),         // 0
            b"}".to_vec(),         // 1
            b"\"color\"".to_vec(), // 2
            b":".to_vec(),         // 3
            b"\"red\"".to_vec(),   // 4
            b"\"blue\"".to_vec(),  // 5
            b"\"green\"".to_vec(), // 6 - not in enum
        ];
        let schema = r#"{"type": "object", "properties": {"color": {"enum": ["red", "blue"]}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"color"'
        engine.advance(3).unwrap(); // ':'

        // try to use "green" - not in enum, should fail
        let result = engine.advance(6);
        assert!(result.is_err());
    }

    #[test]
    fn test_union_type_accepts_string() {
        // union type: ["string", "null"]
        let vocab = vec![
            b"{".to_vec(),               // 0
            b"}".to_vec(),               // 1
            b"\"description\"".to_vec(), // 2
            b":".to_vec(),               // 3
            b"\"text\"".to_vec(),        // 4 - string value
            b"null".to_vec(),            // 5 - null value
        ];
        let schema =
            r#"{"type": "object", "properties": {"description": {"type": ["string", "null"]}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"description":"text"} - string value should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"description"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '"text"'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_union_type_accepts_null() {
        // union type: ["string", "null"]
        let vocab = vec![
            b"{".to_vec(),               // 0
            b"}".to_vec(),               // 1
            b"\"description\"".to_vec(), // 2
            b":".to_vec(),               // 3
            b"\"text\"".to_vec(),        // 4 - string value
            b"null".to_vec(),            // 5 - null value
        ];
        let schema =
            r#"{"type": "object", "properties": {"description": {"type": ["string", "null"]}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"description":null} - null value should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"description"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(5).unwrap(); // 'null'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_union_type_with_number_allows_fractions() {
        // "integer" listed before "number" must not drop the fraction and
        // exponent that "number" allows.
        for types in [
            r#"["integer", "number"]"#,
            r#"["number", "integer"]"#,
            r#"["integer", "null", "number"]"#,
        ] {
            let schema = format!(r#"{{"type": {types}}}"#);
            for doc in ["2.25", "-0.5", "1e5", "7"] {
                assert!(accepts_complete(&schema, doc), "{types} {doc}");
            }
        }
        let schema = r#"{"type": ["integer", "null"]}"#;
        assert_eq!(rejected_at(schema, "2.25"), Some(1));
        assert!(accepts_complete(schema, "7"));
        let property =
            r#"{"type": "object", "properties": {"a": {"type": ["integer", "number"]}}}"#;
        assert!(accepts_complete(property, r#"{"a":-0.5}"#));
    }

    #[test]
    fn test_integer_blocks_decimal_point() {
        let vocab = vec![
            b"{".to_vec(),      // 0
            b"}".to_vec(),      // 1
            b"\"id\"".to_vec(), // 2
            b":".to_vec(),      // 3
            b"10".to_vec(),     // 4
            b".".to_vec(),      // 5 - decimal point
        ];
        let schema = r#"{"type": "object", "properties": {"id": {"type": "integer"}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"id"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '10'

        // decimal point should not be valid for integer
        let valid = engine.get_valid_tokens();
        assert!(!valid.contains(&5)); // '.' should NOT be valid
    }

    #[test]
    fn test_number_allows_decimal_point() {
        let vocab = vec![
            b"{".to_vec(),         // 0
            b"}".to_vec(),         // 1
            b"\"score\"".to_vec(), // 2
            b":".to_vec(),         // 3
            b"10".to_vec(),        // 4
            b".".to_vec(),         // 5 - decimal point
            b"5".to_vec(),         // 6
        ];
        let schema = r#"{"type": "object", "properties": {"score": {"type": "number"}}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        // {"score":10.5} should succeed
        engine.advance(0).unwrap(); // '{'
        engine.advance(2).unwrap(); // '"score"'
        engine.advance(3).unwrap(); // ':'
        engine.advance(4).unwrap(); // '10'
        engine.advance(5).unwrap(); // '.'
        engine.advance(6).unwrap(); // '5'
        engine.advance(1).unwrap(); // '}'

        assert!(engine.is_finished());
    }

    #[test]
    fn test_root_array_schema() {
        let vocab = vec![
            b"[".to_vec(),     // 0
            b"]".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b",".to_vec(),     // 3
            b"\"b\"".to_vec(), // 4
        ];
        let schema =
            r#"{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        engine.advance(0).unwrap(); // '['
        engine.advance(2).unwrap(); // '"a"'
        engine.advance(3).unwrap(); // ','
        engine.advance(4).unwrap(); // '"b"'

        let valid = engine.get_valid_tokens();
        assert!(valid.contains(&1));
        assert!(!valid.contains(&3));

        engine.advance(1).unwrap(); // ']'
        assert!(engine.is_finished());
    }

    #[test]
    fn test_root_integer_schema_finishes_without_delimiter() {
        let vocab = vec![b"42".to_vec(), b".".to_vec()];
        let schema = r#"{"type":"integer"}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        let initial_valid = engine.get_valid_tokens();
        assert!(initial_valid.contains(&0));
        assert!(!initial_valid.contains(&1));

        engine.advance(0).unwrap(); // '42'
        assert!(engine.is_finished());
    }

    #[test]
    fn test_root_string_schema_uses_pattern_and_length_constraints() {
        let vocab = vec![
            b"\"AB\"".to_vec(),  // 0
            b"\"ABC\"".to_vec(), // 1
            b"\"ab\"".to_vec(),  // 2
        ];
        let schema = r#"{"type":"string","pattern":"^[A-Z]+$","minLength":2,"maxLength":2}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();

        let valid = engine.get_valid_tokens();
        assert!(valid.contains(&0));
        assert!(!valid.contains(&1));
        assert!(!valid.contains(&2));

        engine.advance(0).unwrap();
        assert!(engine.is_finished());
    }

    #[test]
    fn test_unbounded_pattern_rejects_characters_that_can_never_match() {
        let vocab = vec![b"\"".to_vec(), b"A".to_vec(), b"B".to_vec(), b"X".to_vec()];
        let mut engine = JsonEngine::new(vocab, r#"{"type":"string","pattern":"^AB"}"#).unwrap();
        engine.advance(0).unwrap();
        // after a leading X the string could never match, so never close.
        assert_eq!(engine.get_valid_tokens(), vec![1]);
        engine.advance(1).unwrap();
        engine.advance(2).unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_pattern_matches_anywhere_in_the_string() {
        let schema =
            |pattern: &str| serde_json::json!({"type": "string", "pattern": pattern}).to_string();
        assert!(accepts_complete(&schema("AB"), r#""XABX""#));
        assert!(accepts_complete(&schema("^AB"), r#""ABX""#));
        assert!(accepts_complete(&schema("AB$"), r#""XAB""#));
        assert_eq!(rejected_at(&schema("AB$"), r#""ABX""#), Some(4));
        // the search wrapper must not change what the pattern's `.` matches.
        assert_eq!(rejected_at(&schema("^A.B$"), r#""A\nB""#), Some(3));
        assert!(accepts_complete(&schema("^A.B$"), r#""A\tB""#));
    }

    #[test]
    fn test_uncompilable_pattern_is_rejected() {
        // `a)|(b` is unbalanced; spliced into a wrapper it would compile.
        for pattern in ["(", "a)|(b", "[z-a]"] {
            let schema = serde_json::json!({"type": "string", "pattern": pattern}).to_string();
            let error = JsonEngine::new(vec![b"\"".to_vec()], &schema)
                .err()
                .unwrap_or_else(|| panic!("{pattern} compiled"));
            assert!(
                error.to_string().contains("Failed to compile regex"),
                "{pattern}: {error}"
            );
        }
    }

    #[test]
    fn test_unsupported_pattern_syntax_is_rejected() {
        // syntax that Rust reads differently from ECMA-262, or that a schema
        // pattern does not support, fails rather than changing its meaning.
        for (pattern, message) in [
            (r"\s", r"unsupported escape \s"),
            (r"a\b", r"unsupported escape \b"),
            (r"(a)\1", r"unsupported escape \1"),
            (r"\u0041", r"unsupported escape \u"),
            (r"\p{L}", r"unsupported escape \p"),
            (r"\x41", r"unsupported escape \x"),
            (r"a\", "unfinished escape"),
            ("[[a]]", "nested character classes"),
            ("[a-z&&[^x]]", "set operations are unsupported"),
            ("[a--b]", "set operations are unsupported"),
            ("[a~~b]", "set operations are unsupported"),
            ("(?=x)", "noncapturing groups are supported"),
            ("(?<=a)b", "noncapturing groups are supported"),
            ("(?P<n>a)", "noncapturing groups are supported"),
            ("(?i)a", "noncapturing groups are supported"),
        ] {
            let schema = serde_json::json!({"type": "string", "pattern": pattern}).to_string();
            let error = JsonEngine::new(vec![b"\"".to_vec()], &schema)
                .err()
                .unwrap_or_else(|| panic!("{pattern} compiled"));
            assert!(error.to_string().contains(message), "{pattern}: {error}");
        }
    }

    #[test]
    fn test_pattern_shorthand_classes_are_ascii() {
        let schema =
            |pattern: &str| serde_json::json!({"type": "string", "pattern": pattern}).to_string();
        assert!(accepts_complete(&schema(r"^\d\w[\d]$"), r#""7_7""#));
        // other scripts' digits and letters are `\D` and `\W`, as in ECMA-262.
        assert_eq!(rejected_at(&schema(r"^\d$"), r#""٣""#), Some(1));
        assert_eq!(rejected_at(&schema(r"^[\d]$"), r#""٣""#), Some(1));
        assert_eq!(rejected_at(&schema(r"^\w$"), r#""é""#), Some(1));
        assert!(accepts_complete(&schema(r"^\D\W$"), r#""٣é""#));
        assert_eq!(rejected_at(&schema(r"^\D$"), r#""7""#), Some(1));
        assert_eq!(rejected_at(&schema(r"^\W$"), r#""_""#), Some(1));
    }

    #[test]
    fn test_pattern_dot_excludes_line_terminators() {
        let dot = serde_json::json!({"type": "string", "pattern": "^a.b$"}).to_string();
        for doc in [r#""a\tb""#, r#""aéb""#, r#""a🚀b""#, r#""a.b""#] {
            assert!(accepts_complete(&dot, doc), "{doc}");
        }
        assert_eq!(rejected_at(&dot, r#""a\nb""#), Some(3));
        assert_eq!(rejected_at(&dot, r#""a\rb""#), Some(3));
        assert_eq!(rejected_at(&dot, b"\"a\xe2\x80\xa8b\""), Some(4));
        assert_eq!(rejected_at(&dot, b"\"a\xe2\x80\xa9b\""), Some(4));
        assert_eq!(rejected_at(&dot, r#""a\u2028b""#), Some(7));
        assert_eq!(rejected_at(&dot, r#""a\u2029b""#), Some(7));
    }

    #[test]
    fn test_portable_pattern_syntax_keeps_its_meaning() {
        let schema =
            |pattern: &str| serde_json::json!({"type": "string", "pattern": pattern}).to_string();
        let groups = schema("^(?:ab)+(c|d)$");
        assert!(accepts_complete(&groups, r#""ababd""#));
        assert_eq!(rejected_at(&groups, r#""abac""#), Some(4));
        // escaped punctuation is itself, and `.` in a class is literal.
        let punctuation = schema(r"^\.\/\-\$\[\]\{\}[.]$");
        assert!(accepts_complete(&punctuation, r#""./-$[]{}.""#));
        assert_eq!(rejected_at(&punctuation, r#""./-$[]{}x""#), Some(9));
        assert!(accepts_complete(&schema(r"^\t\f$"), r#""\t\f""#));
    }

    #[test]
    fn test_string_lengths_count_characters() {
        let bounded = r#"{"type":"string","minLength":2,"maxLength":2}"#;
        assert!(accepts_complete(bounded, r#""éé""#));
        assert!(accepts_complete(bounded, r#""🚀A""#));
        assert_eq!(rejected_at(bounded, r#""é""#), Some(3));
        assert_eq!(rejected_at(bounded, r#""ééé""#), Some(5));
        // an escape is one character, however many bytes encode it.
        assert!(accepts_complete(bounded, r#""\u00e9\n""#));
        let pattern = r#"{"type":"string","pattern":"^é+$","maxLength":2}"#;
        assert!(accepts_complete(pattern, r#""éé""#));
        assert_eq!(rejected_at(pattern, r#""ééé""#), Some(5));
    }

    #[test]
    fn test_string_escapes_decode_to_the_characters_they_encode() {
        let schema =
            |pattern: &str| serde_json::json!({"type": "string", "pattern": pattern}).to_string();
        for (pattern, doc) in [
            (r"^\n$", r#""\n""#),
            (r"^\t\r$", r#""\t\r""#),
            ("^\x08\\f$", r#""\b\f""#),
            (r#"^"\\/$"#, r#""\"\\\/""#),
            ("^A$", r#""\u0041""#),
            ("^é$", r#""\u00E9""#),
            ("^🚀$", r#""\uD83D\uDE80""#),
        ] {
            assert!(accepts_complete(&schema(pattern), doc), "{pattern} {doc}");
        }
        // the escaped byte itself is not the character.
        assert_eq!(rejected_at(&schema("^n$"), r#""\n""#), Some(2));
        assert_eq!(rejected_at(&schema("^u0041$"), r#""\u0041""#), Some(5));
    }

    #[test]
    fn test_invalid_string_contents_are_rejected() {
        let string = r#"{"type":"string"}"#;
        for (doc, offset) in [
            (&br#""\x""#[..], 2),
            (br#""\u00G0""#, 5),
            // a low surrogate cannot stand alone, and a high one needs one
            // next; refused at the first byte that leaves no scalar to encode.
            (br#""\uDC00""#, 4),
            (br#""\uD800x""#, 7),
            (br#""\uD800\u0041""#, 9),
            // raw control characters must be escaped.
            (b"\"a\nb\"", 2),
            // invalid UTF-8: a continuation byte alone, and an overlong lead.
            (b"\"\x80\"", 1),
            (b"\"\xc0\xaf\"", 1),
            (b"\"\xed\xa0\x80\"", 2),
        ] {
            assert_eq!(rejected_at(string, doc), Some(offset), "{doc:?}");
        }
        assert!(accepts_complete(string, r#""\u00e9\uD83D\uDE80é🚀""#));
    }

    #[test]
    fn test_pending_characters_must_keep_the_pattern_reachable() {
        // a partial escape or UTF-8 sequence is allowed only if some
        // character it can still become keeps a match reachable.
        for (pattern, token, expected) in [
            ("^AB", &br"\"[..], true),
            ("^AB", br"\u004", true),
            ("^AB", br"\u005", false),
            ("^AB", b"\xc3", false),
            ("^é", b"\xc3", true),
            ("^é", b"\xc2", false),
            ("^é", br"\u00E", true),
            ("^é", br"\u00F", false),
            ("^🚀", b"\xf0\x9f", true),
            ("^🚀", b"\xf0\x90", false),
            ("^🚀", br"\uD83", true),
            ("^🚀", br"\uD84", false),
            ("^🚀", br"\uD83D\uDE8", true),
            ("^🚀", br"\uD83D\uDC", false),
        ] {
            let schema = serde_json::json!({"type": "string", "pattern": pattern}).to_string();
            let mut engine =
                JsonEngine::new(vec![b"\"".to_vec(), token.to_vec()], &schema).unwrap();
            engine.advance(0).unwrap();
            assert_eq!(
                engine.get_valid_tokens().contains(&1),
                expected,
                "{pattern} {token:?}"
            );
        }
    }

    #[test]
    fn test_min_length_pattern_rejects_branches_that_end_too_short() {
        let vocab = vec![b"\"".to_vec(), b"A".to_vec(), b"B".to_vec()];
        let schema = r#"{"type":"string","pattern":"^(A|BBB)$","minLength":2}"#;
        let mut engine = JsonEngine::new(vocab, schema).unwrap();
        engine.advance(0).unwrap();
        // "A" matches but is shorter than minLength and cannot grow.
        assert_eq!(engine.get_valid_tokens(), vec![2]);
        engine.advance(2).unwrap();
        engine.advance(2).unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![2]);
        engine.advance(2).unwrap();
        assert_eq!(engine.get_valid_tokens(), vec![0]);
    }

    #[test]
    fn test_min_length_outside_pattern_lengths_is_rejected() {
        let vocab = || vec![b"\"".to_vec(), b"A".to_vec()];
        for (schema, accepted) in [
            (
                r#"{"type":"string","pattern":"^[A-Z]{3}$","minLength":5}"#,
                false,
            ),
            // two characters are four bytes: lengths count characters.
            (
                r#"{"type":"string","pattern":"^é{2}$","minLength":3}"#,
                false,
            ),
            (
                r#"{"type":"string","pattern":"^é{2}$","minLength":2}"#,
                true,
            ),
            // lengths 1 and 4 match, but none in 2..=3.
            (
                r#"{"type":"string","pattern":"^(A|BBBB)$","minLength":2,"maxLength":3}"#,
                false,
            ),
            (
                r#"{"type":"string","pattern":"^(A|BBBB)$","minLength":2,"maxLength":4}"#,
                true,
            ),
            // search semantics: anything may follow the match.
            (r#"{"type":"string","pattern":"^AB","minLength":5}"#, true),
        ] {
            assert_eq!(
                JsonEngine::new(vocab(), schema).is_ok(),
                accepted,
                "{schema}"
            );
        }
    }

    #[test]
    fn test_root_boolean_and_null_schemas() {
        let bool_vocab = vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()];
        let mut bool_engine = JsonEngine::new(bool_vocab, r#"{"type":"boolean"}"#).unwrap();
        let valid_bool = bool_engine.get_valid_tokens();
        assert!(valid_bool.contains(&0));
        assert!(valid_bool.contains(&1));
        assert!(!valid_bool.contains(&2));
        bool_engine.advance(0).unwrap();
        assert!(bool_engine.is_finished());

        let null_vocab = vec![b"true".to_vec(), b"null".to_vec()];
        let mut null_engine = JsonEngine::new(null_vocab, r#"{"type":"null"}"#).unwrap();
        let valid_null = null_engine.get_valid_tokens();
        assert!(!valid_null.contains(&0));
        assert!(valid_null.contains(&1));
        null_engine.advance(1).unwrap();
        assert!(null_engine.is_finished());
    }

    const INT_ARRAY: &str = r#"{"type":"array","items":{"type":"integer"}}"#;
    const INT_OBJECT: &str =
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#;
    const INTEGER: &str = r#"{"type":"integer"}"#;
    const NUMBER: &str = r#"{"type":"number"}"#;

    /// feed `doc` one byte at a time through a byte-level vocabulary, requiring
    /// each byte to be in the valid token mask before advancing; returns the
    /// engine, or the offset of the first byte the mask rejected
    fn feed_bytes(schema: &str, doc: impl AsRef<[u8]>) -> Result<JsonEngine, usize> {
        let vocab = (0..=255u8).map(|b| vec![b]).collect();
        let mut engine = JsonEngine::new(vocab, schema).unwrap();
        for (offset, &byte) in doc.as_ref().iter().enumerate() {
            if !engine.get_valid_tokens().contains(&(byte as usize)) {
                return Err(offset);
            }
            engine.advance(byte as usize).unwrap();
        }
        Ok(engine)
    }

    fn rejected_at(schema: &str, doc: impl AsRef<[u8]>) -> Option<usize> {
        feed_bytes(schema, doc).err()
    }

    fn accepts_complete(schema: &str, doc: impl AsRef<[u8]>) -> bool {
        feed_bytes(schema, doc).is_ok_and(|engine| engine.is_finished())
    }

    #[test]
    fn test_array_rejects_trailing_comma() {
        assert!(serde_json::from_str::<Value>("[1,]").is_err());
        assert_eq!(rejected_at(INT_ARRAY, "[1,]"), Some(3));
        assert_eq!(rejected_at(INT_ARRAY, "[1, ]"), Some(4));
        assert_eq!(rejected_at(INT_ARRAY, "[1,2,]"), Some(5));
    }

    #[test]
    fn test_object_rejects_trailing_comma() {
        assert!(serde_json::from_str::<Value>(r#"{"a":1,}"#).is_err());
        assert_eq!(rejected_at(INT_OBJECT, r#"{"a":1,}"#), Some(7));
        assert_eq!(rejected_at(INT_OBJECT, r#"{"a":1, }"#), Some(8));
    }

    fn keyed_schema(key: &str) -> String {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {key: {"type": "integer"}},
            "required": [key]
        });
        schema.to_string()
    }

    #[test]
    fn test_key_escapes_decode_to_the_characters_they_encode() {
        for (key, doc) in [
            ("name", r#"{"name":1}"#),
            ("name", r#"{"\u006Eame":1}"#),
            ("é", r#"{"é":1}"#),
            ("é", r#"{"\u00e9":1}"#),
            ("🚀", r#"{"🚀":1}"#),
            ("🚀", r#"{"\uD83D\uDE80":1}"#),
            ("a/b", r#"{"a\/b":1}"#),
            ("a\"b", r#"{"a\"b":1}"#),
            ("a\nb", r#"{"a\nb":1}"#),
        ] {
            assert!(accepts_complete(&keyed_schema(key), doc), "{key:?} {doc}");
        }
    }

    #[test]
    fn test_key_prefixes_must_still_spell_an_allowed_key() {
        for (key, doc, offset) in [
            // "\n" is a newline, not the "n" of "name".
            ("name", &br#"{"\name":1}"#[..], 3),
            ("name", br#"{"\q"#, 3),
            ("name", br#"{"name\"#, 6),
            // a UTF-8 lead byte is allowed only if it starts the next character.
            ("é", b"{\"\xc2", 2),
            ("é", b"{\"\xc3\xa8", 3),
            ("🚀", br#"{"\ud83d\udc"#, 11),
        ] {
            assert_eq!(
                rejected_at(&keyed_schema(key), doc),
                Some(offset),
                "{key:?} {doc:?}"
            );
        }
        // an escaped key is used once decoded, so it cannot repeat raw.
        let schema =
            r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#;
        assert_eq!(rejected_at(schema, r#"{"\u0061":1,"a"#), Some(13));
    }

    #[test]
    fn test_number_rejects_leading_zeros() {
        assert!(serde_json::from_str::<Value>("007").is_err());
        assert_eq!(rejected_at(INTEGER, "007"), Some(1));
        assert_eq!(rejected_at(INTEGER, "-01"), Some(2));
        assert_eq!(rejected_at(NUMBER, "00.5"), Some(1));
        assert_eq!(rejected_at(INT_ARRAY, "[01]"), Some(2));
        // zeros after the integer part are fine
        assert!(accepts_complete(NUMBER, "0.05"));
        assert!(accepts_complete(NUMBER, "1e05"));
    }

    #[test]
    fn test_number_requires_digit_before_decimal_or_exponent() {
        assert_eq!(rejected_at(NUMBER, "-.5"), Some(1));
        assert_eq!(rejected_at(NUMBER, "-e5"), Some(1));
        assert_eq!(rejected_at(NUMBER, "1.e5"), Some(2));
        assert_eq!(rejected_at(NUMBER, "1.E5"), Some(2));
    }

    #[test]
    fn test_enum_value_rejects_trailing_comma_and_defers_to_parent() {
        // single-byte candidates stay on the stack until a delimiter arrives
        let enum_object =
            r#"{"type":"object","properties":{"n":{"enum":[1,2]},"b":{"type":"integer"}}}"#;
        let enum_array = r#"{"type":"array","items":{"enum":[1,2]}}"#;
        let enum_array_min = r#"{"type":"array","minItems":2,"items":{"enum":[1,2]}}"#;
        let root_enum = r#"{"enum":[1,2]}"#;

        assert_eq!(rejected_at(enum_object, r#"{"n":1,}"#), Some(7));
        assert_eq!(rejected_at(enum_array, "[1,]"), Some(3));
        // the parent's minitems check applies after an enum item
        assert_eq!(rejected_at(enum_array_min, "[1]"), Some(2));
        // a root value cannot be followed by a delimiter
        assert_eq!(rejected_at(root_enum, "1,"), Some(1));
        assert_eq!(rejected_at(root_enum, "1]"), Some(1));

        assert!(accepts_complete(enum_object, r#"{"n":1,"b":2}"#));
        assert!(accepts_complete(enum_object, r#"{"n":2 }"#));
        assert!(accepts_complete(enum_array, "[1,2]"));
        assert!(accepts_complete(enum_array_min, "[1, 2]"));
        assert!(accepts_complete(root_enum, "1"));
        assert!(accepts_complete(root_enum, "2 "));
    }

    #[test]
    fn test_accepts_valid_json_documents() {
        let cases = [
            (INT_ARRAY, "[1,2]"),
            (INT_ARRAY, "[ 1 , 2 ]"),
            (INT_ARRAY, "[]"),
            (INT_ARRAY, "[0,-0,10]"),
            (INT_OBJECT, r#"{"a":1,"b":2}"#),
            (INT_OBJECT, r#"{ "a" : 1 , "b" : 2 }"#),
            (INT_OBJECT, "{}"),
            (INTEGER, "0"),
            (INTEGER, "-0"),
            (INTEGER, "10"),
            (NUMBER, "-0.5"),
            (NUMBER, "1e+2"),
            (NUMBER, "1E-2"),
            (NUMBER, "0e5"),
            (NUMBER, "12.50e10"),
        ];
        for (schema, doc) in cases {
            assert!(
                serde_json::from_str::<Value>(doc).is_ok(),
                "{doc} is not JSON"
            );
            assert!(accepts_complete(schema, doc), "rejected {doc} for {schema}");
        }
    }

    #[test]
    fn test_number_syntax_matches_serde() {
        // every string up to 4 bytes over the number alphabet completes a
        // number schema exactly when serde parses it as a JSON number
        let alphabet = b"01-+.eE";
        let mut docs = Vec::new();
        let mut frontier = vec![String::new()];
        for _ in 0..4 {
            frontier = frontier
                .iter()
                .flat_map(|doc| alphabet.iter().map(move |&b| format!("{doc}{}", b as char)))
                .collect();
            docs.extend(frontier.iter().cloned());
        }
        for doc in &docs {
            let is_number = serde_json::from_str::<serde_json::Number>(doc).is_ok();
            let is_integer = is_number && !doc.contains(['.', 'e', 'E']);
            assert_eq!(accepts_complete(NUMBER, doc), is_number, "number {doc}");
            assert_eq!(accepts_complete(INTEGER, doc), is_integer, "integer {doc}");
        }
    }
}
