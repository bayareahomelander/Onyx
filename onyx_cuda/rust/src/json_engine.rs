//! JSON schema constraint engine with stack-based architecture
//!
//! this module implements a stack-based state machine for enforcing JSON schema
//! constraints during LLM generation. each level of JSON nesting has its own
//! scope on the stack, enabling proper tracking of nested objects and arrays.

use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::constraint::{ConstraintEngine, ConstraintError};
use crate::regex_engine::compile_pattern_dfa;
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

/// A decoded JSON character, an unfinished escape/UTF-8 sequence, or a closing quote.
enum StringStep {
    Pending,
    Character(char),
    End,
}

#[derive(Debug, Clone)]
pub struct StringState {
    pending: Vec<u8>,
    pattern_dfa: Option<Arc<dense::DFA<Vec<u32>>>>,
    dfa_state: Option<StateID>,
    char_count: usize,
    min_length: Option<usize>,
    max_length: Option<usize>,
}

impl StringState {
    pub fn new_started() -> Self {
        Self {
            pending: Vec::new(),
            pattern_dfa: None,
            dfa_state: None,
            char_count: 0,
            min_length: None,
            max_length: None,
        }
    }

    pub fn with_pattern_and_constraints(
        pattern: Option<&str>,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> Self {
        let mut state = Self::new_started();
        if let Some(pattern) = pattern {
            // Schema compilation validates every pattern before constructing states.
            let normalized = crate::schema::schema_pattern(pattern).expect("validated pattern");
            let compiled =
                compile_pattern_dfa(&format!("(?s:.*(?:{normalized}).*)")).expect("validated DFA");
            state.pattern_dfa = Some(Arc::new(compiled.dfa));
            state.dfa_state = Some(compiled.initial_state);
        }
        state.min_length = min_length;
        state.max_length = max_length;
        state
    }

    fn decode_byte(&mut self, byte: u8) -> Option<StringStep> {
        if self.pending.is_empty() {
            match byte {
                b'"' => return Some(StringStep::End),
                b'\\' | 0xc2..=0xf4 => {
                    self.pending.push(byte);
                    return Some(StringStep::Pending);
                }
                0x20..=0x7f => return Some(StringStep::Character(byte as char)),
                _ => return None,
            }
        }
        self.pending.push(byte);
        let character = if self.pending[0] == b'\\' {
            match self.pending.len() {
                2 => match byte {
                    b'"' | b'\\' | b'/' => byte as char,
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => return Some(StringStep::Pending),
                    _ => return None,
                },
                3..=6 | 9..=12 => {
                    if !byte.is_ascii_hexdigit() {
                        return None;
                    }
                    let length = self.pending.len();
                    if length != 6 && length != 12 {
                        return Some(StringStep::Pending);
                    }
                    let first =
                        u16::from_str_radix(std::str::from_utf8(&self.pending[2..6]).ok()?, 16)
                            .ok()?;
                    if length == 6 {
                        if (0xd800..=0xdbff).contains(&first) {
                            return Some(StringStep::Pending);
                        }
                        char::from_u32(first as u32)?
                    } else {
                        let second = u16::from_str_radix(
                            std::str::from_utf8(&self.pending[8..12]).ok()?,
                            16,
                        )
                        .ok()?;
                        if !(0xdc00..=0xdfff).contains(&second) {
                            return None;
                        }
                        char::from_u32(
                            0x10000 + (((first as u32) - 0xd800) << 10) + (second as u32) - 0xdc00,
                        )?
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
                if let (Some(dfa), Some(state)) = (&self.pattern_dfa, &mut self.dfa_state) {
                    let mut buffer = [0; 4];
                    for &byte in character.encode_utf8(&mut buffer).as_bytes() {
                        *state = dfa.next_state(*state, byte);
                        if dfa.is_dead_state(*state) {
                            return None;
                        }
                    }
                }
            }
            StringStep::End => {
                if self.min_length.is_some_and(|min| self.char_count < min) {
                    return None;
                }
                if let (Some(dfa), Some(state)) = (&self.pattern_dfa, self.dfa_state) {
                    if !dfa.is_match_state(dfa.next_eoi_state(state)) {
                        return None;
                    }
                }
            }
            StringStep::Pending => {}
        }
        Some(step)
    }
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
        /// whether we're in an escape sequence (for key parsing)
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
    schema: Arc<Value>,
    output: Vec<u8>,
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
        let root_value_blueprint = Arc::new(PropertyBlueprint::from_value("_root", &schema));

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
            schema: Arc::new(schema),
            output: Vec::new(),
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
        !state.expect_digit && serde_json::from_str::<serde_json::Number>(&state.buffer).is_ok()
    }

    fn complete(stack: &[Scope], finished: bool) -> bool {
        (finished && stack.is_empty())
            || matches!(stack, [Scope::Number(state)] if Self::number_is_complete(state))
            || matches!(stack, [Scope::Enum(state)] if state.candidates.iter().any(|c| c.len() == state.cursor))
    }

    fn valid_completion(&self, bytes: &[u8]) -> bool {
        serde_json::from_slice::<Value>(bytes)
            .is_ok_and(|value| crate::schema::value_matches_schema(&value, &self.schema))
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
                                bp.pattern.as_deref(),
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

        if Self::complete(&temp_stack, temp_finished) {
            let mut output = self.output.clone();
            output.extend_from_slice(token_bytes);
            return self.valid_completion(&output);
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
        _root_blueprint: &Arc<SchemaBlueprint>,
        root_value_blueprint: &Arc<PropertyBlueprint>,
        stack: &mut Vec<Scope>,
        finished: &mut bool,
        byte: u8,
    ) -> bool {
        if *finished {
            return Self::is_whitespace(byte);
        }
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
                            return blueprint.allowed_keys.iter().any(|key| {
                                key.starts_with(key_buffer.as_str()) && !used_keys.contains(key)
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
                            let property =
                                blueprint.get_property(key_buffer).cloned().map(Arc::new);
                            *syntax_state = ObjectSyntaxState::ExpectCommaOrEnd;
                            return Self::push_value_scope(stack, &property, byte);
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

                Scope::String(state) => match state.advance(byte) {
                    Some(StringStep::End) => {
                        stack.pop();
                        if stack.is_empty() {
                            *finished = true;
                        } else {
                            Self::update_parent_after_value(stack);
                        }
                        return true;
                    }
                    Some(_) => return true,
                    None => return false,
                },

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

                    if has_complete && Self::is_number_terminator(byte) {
                        stack.pop();
                        if stack.is_empty() {
                            *finished = true;
                            return Self::is_whitespace(byte);
                        }
                        Self::update_parent_after_value(stack);
                        continue;
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
        self.output.clear();
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

        if !self.validate_token(&bytes) {
            self.dead = true;
            return Err(ConstraintError::InvalidState(
                "Token violates the JSON schema".into(),
            ));
        }

        for &byte in &bytes {
            self.advance_byte(byte)?;
        }
        self.output.extend_from_slice(&bytes);

        Ok(())
    }

    fn is_finished(&self) -> bool {
        !self.dead
            && Self::complete(&self.stack, self.finished)
            && self.valid_completion(&self.output)
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
            schema: Arc::clone(&self.schema),
            output: self.output.clone(),
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
        let schema = r#"{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}}"#;
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
}
