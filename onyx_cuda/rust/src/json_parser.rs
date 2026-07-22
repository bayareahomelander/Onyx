//! Cloneable RFC 8259 byte parser for the Deliverable 20 JSON core.

use crate::json_schema::{
    CompiledPattern, CompiledSchema, JsonValueType, SchemaNodeId, StringConstraints,
};
use regex_automata::util::primitives::StateID;
use std::error::Error;
use std::fmt;
use std::str;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct JsonParserError {
    message: String,
}

impl JsonParserError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for JsonParserError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for JsonParserError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootPhase {
    ExpectValue,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectPhase {
    FirstKeyOrEnd,
    KeyAfterComma,
    InKey,
    ExpectColon,
    ExpectValue,
    CommaOrEnd,
}

#[derive(Clone)]
struct ObjectFrame {
    node_id: SchemaNodeId,
    phase: ObjectPhase,
    used_properties: Vec<bool>,
    missing_required: usize,
    current_property: Option<usize>,
    key_decoder: Option<JsonStringDecoder>,
    key_buffer: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArrayPhase {
    FirstValueOrEnd,
    ValueAfterComma,
    CommaOrEnd,
}

#[derive(Clone)]
struct ArrayFrame {
    node_id: SchemaNodeId,
    phase: ArrayPhase,
    item_count: usize,
}

#[derive(Clone)]
struct StringFrame {
    decoder: JsonStringDecoder,
    char_count: usize,
    min_length: Option<usize>,
    max_length: Option<usize>,
    pattern: Option<Arc<CompiledPattern>>,
    pattern_state: Option<StateID>,
}

impl StringFrame {
    fn new(constraints: Option<&StringConstraints>) -> Self {
        let pattern = constraints.and_then(|value| value.pattern.clone());
        let pattern_state = pattern.as_ref().map(|value| value.initial_state());
        Self {
            decoder: JsonStringDecoder::new(),
            char_count: 0,
            min_length: constraints.and_then(|value| value.min_length),
            max_length: constraints.and_then(|value| value.max_length),
            pattern,
            pattern_state,
        }
    }

    fn consume(&mut self, byte: u8) -> Result<StringFrameAction, JsonParserError> {
        if self
            .max_length
            .is_some_and(|maximum| self.char_count >= maximum)
            && !(byte == b'"' && self.decoder.can_close())
        {
            return Err(JsonParserError::new(
                "decoded string cannot continue after maxLength is reached",
            ));
        }
        match self.decoder.consume(byte)? {
            StringEvent::Pending => Ok(StringFrameAction::Consumed),
            StringEvent::CodePoint(character) => {
                if self
                    .max_length
                    .is_some_and(|maximum| self.char_count >= maximum)
                {
                    return Err(JsonParserError::new(
                        "decoded string exceeds its maxLength constraint",
                    ));
                }
                if let (Some(pattern), Some(state)) = (&self.pattern, self.pattern_state) {
                    let mut encoded = [0_u8; 4];
                    let bytes = character.encode_utf8(&mut encoded).as_bytes();
                    self.pattern_state = Some(pattern.advance(state, bytes).ok_or_else(|| {
                        JsonParserError::new("decoded string entered the pattern DFA dead state")
                    })?);
                }
                self.char_count = self.char_count.checked_add(1).ok_or_else(|| {
                    JsonParserError::new("decoded string length counter overflowed")
                })?;
                Ok(StringFrameAction::Consumed)
            }
            StringEvent::Closed => {
                if self
                    .min_length
                    .is_some_and(|minimum| self.char_count < minimum)
                {
                    return Err(JsonParserError::new(
                        "decoded string does not satisfy its minLength constraint",
                    ));
                }
                if let (Some(pattern), Some(state)) = (&self.pattern, self.pattern_state) {
                    if !pattern.is_match(state) {
                        return Err(JsonParserError::new(
                            "decoded string does not satisfy its anchored pattern",
                        ));
                    }
                }
                Ok(StringFrameAction::Complete)
            }
        }
    }
}

enum StringFrameAction {
    Consumed,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumberPhase {
    Minus,
    Zero,
    IntegerDigits,
    DecimalPoint,
    FractionDigits,
    ExponentMarker,
    ExponentSign,
    ExponentDigits,
}

#[derive(Clone)]
struct NumberFrame {
    phase: NumberPhase,
    integer_only: bool,
}

impl NumberFrame {
    fn start(byte: u8, integer_only: bool) -> Result<Self, JsonParserError> {
        let phase = match byte {
            b'-' => NumberPhase::Minus,
            b'0' => NumberPhase::Zero,
            b'1'..=b'9' => NumberPhase::IntegerDigits,
            _ => return Err(JsonParserError::new("invalid first byte for a JSON number")),
        };
        Ok(Self {
            phase,
            integer_only,
        })
    }

    fn is_complete(&self) -> bool {
        matches!(
            self.phase,
            NumberPhase::Zero
                | NumberPhase::IntegerDigits
                | NumberPhase::FractionDigits
                | NumberPhase::ExponentDigits
        )
    }

    fn consume(&mut self, byte: u8) -> Result<NumberAction, JsonParserError> {
        if is_value_terminator(byte) {
            return if self.is_complete() {
                Ok(NumberAction::CompleteBeforeByte)
            } else {
                Err(JsonParserError::new(
                    "incomplete JSON number cannot terminate",
                ))
            };
        }

        self.phase = match self.phase {
            NumberPhase::Minus => match byte {
                b'0' => NumberPhase::Zero,
                b'1'..=b'9' => NumberPhase::IntegerDigits,
                _ => return Err(JsonParserError::new("minus must be followed by a digit")),
            },
            NumberPhase::Zero => {
                if self.integer_only {
                    return Err(JsonParserError::new(
                        "integer values cannot contain a fraction or exponent",
                    ));
                }
                match byte {
                    b'.' => NumberPhase::DecimalPoint,
                    b'e' | b'E' => NumberPhase::ExponentMarker,
                    b'0'..=b'9' => {
                        return Err(JsonParserError::new(
                            "leading zeroes are not valid JSON numbers",
                        ));
                    }
                    _ => return Err(JsonParserError::new("invalid JSON number byte")),
                }
            }
            NumberPhase::IntegerDigits => match byte {
                b'0'..=b'9' => NumberPhase::IntegerDigits,
                b'.' if !self.integer_only => NumberPhase::DecimalPoint,
                b'e' | b'E' if !self.integer_only => NumberPhase::ExponentMarker,
                _ => return Err(JsonParserError::new("invalid JSON number byte")),
            },
            NumberPhase::DecimalPoint => match byte {
                b'0'..=b'9' => NumberPhase::FractionDigits,
                _ => {
                    return Err(JsonParserError::new(
                        "decimal point must be followed by a digit",
                    ));
                }
            },
            NumberPhase::FractionDigits => match byte {
                b'0'..=b'9' => NumberPhase::FractionDigits,
                b'e' | b'E' => NumberPhase::ExponentMarker,
                _ => return Err(JsonParserError::new("invalid fraction byte")),
            },
            NumberPhase::ExponentMarker => match byte {
                b'+' | b'-' => NumberPhase::ExponentSign,
                b'0'..=b'9' => NumberPhase::ExponentDigits,
                _ => {
                    return Err(JsonParserError::new(
                        "exponent marker must be followed by a sign or digit",
                    ));
                }
            },
            NumberPhase::ExponentSign => match byte {
                b'0'..=b'9' => NumberPhase::ExponentDigits,
                _ => {
                    return Err(JsonParserError::new(
                        "exponent sign must be followed by a digit",
                    ));
                }
            },
            NumberPhase::ExponentDigits => match byte {
                b'0'..=b'9' => NumberPhase::ExponentDigits,
                _ => return Err(JsonParserError::new("invalid exponent byte")),
            },
        };
        Ok(NumberAction::Consumed)
    }
}

enum NumberAction {
    Consumed,
    CompleteBeforeByte,
}

#[derive(Clone)]
struct LiteralFrame {
    target: &'static [u8],
    position: usize,
}

#[derive(Clone)]
struct EnumFrame {
    node_id: SchemaNodeId,
    candidate_indices: Vec<usize>,
    cursor: usize,
}

impl EnumFrame {
    fn has_complete_candidate(&self, schema: &CompiledSchema) -> bool {
        let candidates = schema
            .node(self.node_id)
            .enum_candidates
            .as_ref()
            .expect("enum frame must reference enum candidates");
        self.candidate_indices
            .iter()
            .any(|&index| candidates[index].len() == self.cursor)
    }

    fn consume(
        &mut self,
        schema: &CompiledSchema,
        byte: u8,
    ) -> Result<EnumAction, JsonParserError> {
        let candidates = schema
            .node(self.node_id)
            .enum_candidates
            .as_ref()
            .expect("enum frame must reference enum candidates");
        let matching = self
            .candidate_indices
            .iter()
            .copied()
            .filter(|&index| candidates[index].get(self.cursor) == Some(&byte))
            .collect::<Vec<_>>();
        if !matching.is_empty() {
            self.candidate_indices = matching;
            self.cursor += 1;
            return Ok(EnumAction::Consumed);
        }
        if self.has_complete_candidate(schema) && is_value_terminator(byte) {
            return Ok(EnumAction::CompleteBeforeByte);
        }
        Err(JsonParserError::new(
            "byte does not continue any serialized enum candidate",
        ))
    }
}

enum EnumAction {
    Consumed,
    CompleteBeforeByte,
}

#[derive(Clone)]
enum Frame {
    Root(RootPhase),
    Object(ObjectFrame),
    Array(ArrayFrame),
    String(StringFrame),
    Number(NumberFrame),
    Literal(LiteralFrame),
    Enum(EnumFrame),
}

enum ParserAction {
    Consumed,
    StartValue { node_id: SchemaNodeId, byte: u8 },
    CompleteValueConsumed,
    CompleteValueBeforeByte,
}

#[derive(Clone)]
pub(crate) struct JsonParserState {
    stack: Vec<Frame>,
}

impl JsonParserState {
    pub(crate) fn new() -> Self {
        Self {
            stack: vec![Frame::Root(RootPhase::ExpectValue)],
        }
    }

    pub(crate) fn consume_token(
        &mut self,
        schema: &CompiledSchema,
        token_bytes: &[u8],
    ) -> Result<(), JsonParserError> {
        for &byte in token_bytes {
            self.consume_byte(schema, byte)?;
        }
        Ok(())
    }

    pub(crate) fn is_match(&self, schema: &CompiledSchema) -> bool {
        match self.stack.as_slice() {
            [Frame::Root(RootPhase::Complete)] => true,
            [Frame::Root(RootPhase::ExpectValue), Frame::Number(number)] => number.is_complete(),
            [Frame::Root(RootPhase::ExpectValue), Frame::Enum(state)] => {
                state.has_complete_candidate(schema)
            }
            _ => false,
        }
    }

    pub(crate) fn is_dead(&self) -> bool {
        false
    }

    fn consume_byte(&mut self, schema: &CompiledSchema, byte: u8) -> Result<(), JsonParserError> {
        loop {
            let action = self.next_action(schema, byte)?;
            match action {
                ParserAction::Consumed => return Ok(()),
                ParserAction::StartValue { node_id, byte } => {
                    self.start_value(schema, node_id, byte)?;
                    return Ok(());
                }
                ParserAction::CompleteValueConsumed => {
                    self.stack.pop();
                    self.finish_value()?;
                    return Ok(());
                }
                ParserAction::CompleteValueBeforeByte => {
                    self.stack.pop();
                    self.finish_value()?;
                }
            }
        }
    }

    fn next_action(
        &mut self,
        schema: &CompiledSchema,
        byte: u8,
    ) -> Result<ParserAction, JsonParserError> {
        let frame = self
            .stack
            .last_mut()
            .ok_or_else(|| JsonParserError::new("JSON parser stack is empty"))?;
        match frame {
            Frame::Root(RootPhase::ExpectValue) => {
                if is_whitespace(byte) {
                    Ok(ParserAction::Consumed)
                } else {
                    Ok(ParserAction::StartValue {
                        node_id: schema.root(),
                        byte,
                    })
                }
            }
            Frame::Root(RootPhase::Complete) => {
                if is_whitespace(byte) {
                    Ok(ParserAction::Consumed)
                } else {
                    Err(JsonParserError::new(
                        "only structural whitespace may follow a complete JSON root",
                    ))
                }
            }
            Frame::Object(object) => object_action(object, schema, byte),
            Frame::Array(array) => array_action(array, schema, byte),
            Frame::String(string) => match string.consume(byte)? {
                StringFrameAction::Consumed => Ok(ParserAction::Consumed),
                StringFrameAction::Complete => Ok(ParserAction::CompleteValueConsumed),
            },
            Frame::Number(number) => match number.consume(byte)? {
                NumberAction::Consumed => Ok(ParserAction::Consumed),
                NumberAction::CompleteBeforeByte => Ok(ParserAction::CompleteValueBeforeByte),
            },
            Frame::Literal(literal) => {
                if literal.target.get(literal.position) != Some(&byte) {
                    return Err(JsonParserError::new("invalid JSON literal byte"));
                }
                literal.position += 1;
                if literal.position == literal.target.len() {
                    Ok(ParserAction::CompleteValueConsumed)
                } else {
                    Ok(ParserAction::Consumed)
                }
            }
            Frame::Enum(state) => match state.consume(schema, byte)? {
                EnumAction::Consumed => Ok(ParserAction::Consumed),
                EnumAction::CompleteBeforeByte => Ok(ParserAction::CompleteValueBeforeByte),
            },
        }
    }

    fn start_value(
        &mut self,
        schema: &CompiledSchema,
        node_id: SchemaNodeId,
        byte: u8,
    ) -> Result<(), JsonParserError> {
        let node = schema.node(node_id);
        if let Some(candidates) = &node.enum_candidates {
            let candidate_indices = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    (candidate.first() == Some(&byte)).then_some(index)
                })
                .collect::<Vec<_>>();
            if candidate_indices.is_empty() {
                return Err(JsonParserError::new(
                    "first byte does not match any serialized enum candidate",
                ));
            }
            self.stack.push(Frame::Enum(EnumFrame {
                node_id,
                candidate_indices,
                cursor: 1,
            }));
            return Ok(());
        }

        let value_type = select_value_type(&node.allowed_types, byte).ok_or_else(|| {
            JsonParserError::new("byte cannot begin any value allowed by the schema")
        })?;
        let frame = match value_type {
            JsonValueType::Object => {
                let constraints = node
                    .object
                    .as_ref()
                    .expect("explicit object schemas have object constraints");
                Frame::Object(ObjectFrame {
                    node_id,
                    phase: ObjectPhase::FirstKeyOrEnd,
                    used_properties: vec![false; constraints.properties.len()],
                    missing_required: constraints
                        .properties
                        .iter()
                        .filter(|property| property.required)
                        .count(),
                    current_property: None,
                    key_decoder: None,
                    key_buffer: String::new(),
                })
            }
            JsonValueType::Array => Frame::Array(ArrayFrame {
                node_id,
                phase: ArrayPhase::FirstValueOrEnd,
                item_count: 0,
            }),
            JsonValueType::String => Frame::String(StringFrame::new(node.string.as_ref())),
            JsonValueType::Number => Frame::Number(NumberFrame::start(byte, false)?),
            JsonValueType::Integer => Frame::Number(NumberFrame::start(byte, true)?),
            JsonValueType::Boolean => Frame::Literal(LiteralFrame {
                target: if byte == b't' { b"true" } else { b"false" },
                position: 1,
            }),
            JsonValueType::Null => Frame::Literal(LiteralFrame {
                target: b"null",
                position: 1,
            }),
        };
        self.stack.push(frame);
        Ok(())
    }

    fn finish_value(&mut self) -> Result<(), JsonParserError> {
        let parent = self
            .stack
            .last_mut()
            .ok_or_else(|| JsonParserError::new("completed JSON value has no parent frame"))?;
        match parent {
            Frame::Root(phase @ RootPhase::ExpectValue) => {
                *phase = RootPhase::Complete;
                Ok(())
            }
            Frame::Object(object) if object.phase == ObjectPhase::ExpectValue => {
                object.phase = ObjectPhase::CommaOrEnd;
                object.current_property = None;
                Ok(())
            }
            Frame::Array(array)
                if matches!(
                    array.phase,
                    ArrayPhase::FirstValueOrEnd | ArrayPhase::ValueAfterComma
                ) =>
            {
                array.item_count = array
                    .item_count
                    .checked_add(1)
                    .ok_or_else(|| JsonParserError::new("JSON array item counter overflowed"))?;
                array.phase = ArrayPhase::CommaOrEnd;
                Ok(())
            }
            _ => Err(JsonParserError::new(
                "completed JSON value encountered an invalid parent phase",
            )),
        }
    }
}

fn object_action(
    frame: &mut ObjectFrame,
    schema: &CompiledSchema,
    byte: u8,
) -> Result<ParserAction, JsonParserError> {
    let constraints = schema
        .node(frame.node_id)
        .object
        .as_ref()
        .expect("object frame must reference object constraints");
    match frame.phase {
        ObjectPhase::FirstKeyOrEnd | ObjectPhase::KeyAfterComma => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            if byte == b'}' {
                if frame.phase == ObjectPhase::KeyAfterComma {
                    return Err(JsonParserError::new(
                        "trailing object comma is not valid JSON",
                    ));
                }
                if frame.missing_required != 0 {
                    return Err(JsonParserError::new(
                        "object cannot close before all required properties are present",
                    ));
                }
                return Ok(ParserAction::CompleteValueConsumed);
            }
            if byte != b'"' {
                return Err(JsonParserError::new(
                    "object expects a quoted property name",
                ));
            }
            if !frame
                .used_properties
                .iter()
                .enumerate()
                .any(|(index, used)| !used && index < constraints.properties.len())
            {
                return Err(JsonParserError::new(
                    "object has no unused declared property to generate",
                ));
            }
            frame.phase = ObjectPhase::InKey;
            frame.key_decoder = Some(JsonStringDecoder::new());
            frame.key_buffer.clear();
            Ok(ParserAction::Consumed)
        }
        ObjectPhase::InKey => {
            let event = frame
                .key_decoder
                .as_mut()
                .expect("in-key phase must have a decoder")
                .consume(byte)?;
            match event {
                StringEvent::Pending => Ok(ParserAction::Consumed),
                StringEvent::CodePoint(character) => {
                    frame.key_buffer.push(character);
                    let has_prefix =
                        constraints
                            .properties
                            .iter()
                            .enumerate()
                            .any(|(index, property)| {
                                !frame.used_properties[index]
                                    && property.name.starts_with(&frame.key_buffer)
                            });
                    if !has_prefix {
                        return Err(JsonParserError::new(
                            "decoded object key is not a prefix of an unused declared property",
                        ));
                    }
                    Ok(ParserAction::Consumed)
                }
                StringEvent::Closed => {
                    let property_index = constraints
                        .properties
                        .binary_search_by(|property| {
                            property.name.as_str().cmp(frame.key_buffer.as_str())
                        })
                        .map_err(|_| {
                            JsonParserError::new("decoded object key is not declared by the schema")
                        })?;
                    if frame.used_properties[property_index] {
                        return Err(JsonParserError::new(
                            "duplicate object properties are not allowed",
                        ));
                    }
                    frame.current_property = Some(property_index);
                    frame.phase = ObjectPhase::ExpectColon;
                    Ok(ParserAction::Consumed)
                }
            }
        }
        ObjectPhase::ExpectColon => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            if byte != b':' {
                return Err(JsonParserError::new(
                    "object property name must be followed by ':'",
                ));
            }
            let property_index = frame
                .current_property
                .expect("colon phase must reference a property");
            frame.used_properties[property_index] = true;
            if constraints.properties[property_index].required {
                frame.missing_required = frame
                    .missing_required
                    .checked_sub(1)
                    .ok_or_else(|| JsonParserError::new("required-property counter underflowed"))?;
            }
            frame.phase = ObjectPhase::ExpectValue;
            Ok(ParserAction::Consumed)
        }
        ObjectPhase::ExpectValue => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            let property_index = frame
                .current_property
                .expect("value phase must reference a property");
            Ok(ParserAction::StartValue {
                node_id: constraints.properties[property_index].schema,
                byte,
            })
        }
        ObjectPhase::CommaOrEnd => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            if byte == b',' {
                if frame.used_properties.iter().all(|used| *used) {
                    return Err(JsonParserError::new(
                        "object comma cannot be followed by an undeclared property",
                    ));
                }
                frame.phase = ObjectPhase::KeyAfterComma;
                return Ok(ParserAction::Consumed);
            }
            if byte == b'}' {
                if frame.missing_required != 0 {
                    return Err(JsonParserError::new(
                        "object cannot close before all required properties are present",
                    ));
                }
                return Ok(ParserAction::CompleteValueConsumed);
            }
            Err(JsonParserError::new("object expects ',' or '}'"))
        }
    }
}

fn array_action(
    frame: &mut ArrayFrame,
    schema: &CompiledSchema,
    byte: u8,
) -> Result<ParserAction, JsonParserError> {
    let constraints = schema
        .node(frame.node_id)
        .array
        .as_ref()
        .expect("array frame must reference array constraints");
    match frame.phase {
        ArrayPhase::FirstValueOrEnd | ArrayPhase::ValueAfterComma => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            if byte == b']' {
                if frame.phase == ArrayPhase::ValueAfterComma {
                    return Err(JsonParserError::new(
                        "trailing array comma is not valid JSON",
                    ));
                }
                if constraints
                    .min_items
                    .is_some_and(|minimum| frame.item_count < minimum)
                {
                    return Err(JsonParserError::new(
                        "array cannot close before minItems is satisfied",
                    ));
                }
                return Ok(ParserAction::CompleteValueConsumed);
            }
            if constraints
                .max_items
                .is_some_and(|maximum| frame.item_count >= maximum)
            {
                return Err(JsonParserError::new(
                    "array cannot start another item after maxItems",
                ));
            }
            Ok(ParserAction::StartValue {
                node_id: constraints.items,
                byte,
            })
        }
        ArrayPhase::CommaOrEnd => {
            if is_whitespace(byte) {
                return Ok(ParserAction::Consumed);
            }
            if byte == b',' {
                if constraints
                    .max_items
                    .is_some_and(|maximum| frame.item_count >= maximum)
                {
                    return Err(JsonParserError::new(
                        "array comma cannot be followed by an item after maxItems",
                    ));
                }
                frame.phase = ArrayPhase::ValueAfterComma;
                return Ok(ParserAction::Consumed);
            }
            if byte == b']' {
                if constraints
                    .min_items
                    .is_some_and(|minimum| frame.item_count < minimum)
                {
                    return Err(JsonParserError::new(
                        "array cannot close before minItems is satisfied",
                    ));
                }
                return Ok(ParserAction::CompleteValueConsumed);
            }
            Err(JsonParserError::new("array expects ',' or ']'"))
        }
    }
}

fn select_value_type(types: &[JsonValueType], byte: u8) -> Option<JsonValueType> {
    match byte {
        b'{' if types.contains(&JsonValueType::Object) => Some(JsonValueType::Object),
        b'[' if types.contains(&JsonValueType::Array) => Some(JsonValueType::Array),
        b'"' if types.contains(&JsonValueType::String) => Some(JsonValueType::String),
        b'-' | b'0'..=b'9' if types.contains(&JsonValueType::Number) => Some(JsonValueType::Number),
        b'-' | b'0'..=b'9' if types.contains(&JsonValueType::Integer) => {
            Some(JsonValueType::Integer)
        }
        b't' | b'f' if types.contains(&JsonValueType::Boolean) => Some(JsonValueType::Boolean),
        b'n' if types.contains(&JsonValueType::Null) => Some(JsonValueType::Null),
        _ => None,
    }
}

fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_value_terminator(byte: u8) -> bool {
    is_whitespace(byte) || matches!(byte, b',' | b'}' | b']')
}

#[derive(Clone)]
struct JsonStringDecoder {
    mode: StringMode,
    pending_utf8: Vec<u8>,
}

impl JsonStringDecoder {
    fn new() -> Self {
        Self {
            mode: StringMode::Normal,
            pending_utf8: Vec::new(),
        }
    }

    fn can_close(&self) -> bool {
        matches!(self.mode, StringMode::Normal) && self.pending_utf8.is_empty()
    }

    fn consume(&mut self, byte: u8) -> Result<StringEvent, JsonParserError> {
        match self.mode {
            StringMode::Normal => self.consume_normal(byte),
            StringMode::Escape => self.consume_escape(byte),
            StringMode::Unicode { value, digits } => {
                self.consume_unicode(byte, value, digits, false, 0)
            }
            StringMode::ExpectLowSurrogateSlash { high } => {
                if byte != b'\\' {
                    return Err(JsonParserError::new(
                        "high surrogate must be followed by a low-surrogate escape",
                    ));
                }
                self.mode = StringMode::ExpectLowSurrogateU { high };
                Ok(StringEvent::Pending)
            }
            StringMode::ExpectLowSurrogateU { high } => {
                if byte != b'u' {
                    return Err(JsonParserError::new(
                        "high surrogate must be followed by \\u",
                    ));
                }
                self.mode = StringMode::LowSurrogate {
                    high,
                    value: 0,
                    digits: 0,
                };
                Ok(StringEvent::Pending)
            }
            StringMode::LowSurrogate {
                high,
                value,
                digits,
            } => self.consume_unicode(byte, value, digits, true, high),
        }
    }

    fn consume_normal(&mut self, byte: u8) -> Result<StringEvent, JsonParserError> {
        if !self.pending_utf8.is_empty() {
            return self.consume_utf8(byte);
        }
        match byte {
            b'"' => Ok(StringEvent::Closed),
            b'\\' => {
                self.mode = StringMode::Escape;
                Ok(StringEvent::Pending)
            }
            0x00..=0x1F => Err(JsonParserError::new(
                "unescaped control byte is not valid in a JSON string",
            )),
            0x20..=0x7F => Ok(StringEvent::CodePoint(char::from(byte))),
            _ => self.consume_utf8(byte),
        }
    }

    fn consume_utf8(&mut self, byte: u8) -> Result<StringEvent, JsonParserError> {
        self.pending_utf8.push(byte);
        match str::from_utf8(&self.pending_utf8) {
            Ok(text) => {
                let mut characters = text.chars();
                let character = characters.next().ok_or_else(|| {
                    JsonParserError::new("UTF-8 decoder produced an empty scalar")
                })?;
                if characters.next().is_some() {
                    return Err(JsonParserError::new(
                        "UTF-8 decoder produced more than one scalar",
                    ));
                }
                self.pending_utf8.clear();
                Ok(StringEvent::CodePoint(character))
            }
            Err(error) if error.error_len().is_none() && self.pending_utf8.len() < 4 => {
                Ok(StringEvent::Pending)
            }
            Err(_) => Err(JsonParserError::new("invalid UTF-8 in JSON string")),
        }
    }

    fn consume_escape(&mut self, byte: u8) -> Result<StringEvent, JsonParserError> {
        let character = match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{0008}',
            b'f' => '\u{000C}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                self.mode = StringMode::Unicode {
                    value: 0,
                    digits: 0,
                };
                return Ok(StringEvent::Pending);
            }
            _ => return Err(JsonParserError::new("invalid JSON string escape")),
        };
        self.mode = StringMode::Normal;
        Ok(StringEvent::CodePoint(character))
    }

    fn consume_unicode(
        &mut self,
        byte: u8,
        value: u16,
        digits: u8,
        low_surrogate: bool,
        high: u16,
    ) -> Result<StringEvent, JsonParserError> {
        let digit = hex_value(byte).ok_or_else(|| {
            JsonParserError::new("Unicode escape must contain exactly four hexadecimal digits")
        })?;
        let next_value = (value << 4) | u16::from(digit);
        let next_digits = digits + 1;
        if next_digits < 4 {
            self.mode = if low_surrogate {
                StringMode::LowSurrogate {
                    high,
                    value: next_value,
                    digits: next_digits,
                }
            } else {
                StringMode::Unicode {
                    value: next_value,
                    digits: next_digits,
                }
            };
            return Ok(StringEvent::Pending);
        }

        if low_surrogate {
            if !(0xDC00..=0xDFFF).contains(&next_value) {
                return Err(JsonParserError::new(
                    "high surrogate was not followed by a valid low surrogate",
                ));
            }
            let scalar =
                0x10000 + ((u32::from(high) - 0xD800) << 10) + (u32::from(next_value) - 0xDC00);
            let character = char::from_u32(scalar)
                .ok_or_else(|| JsonParserError::new("surrogate pair is not a Unicode scalar"))?;
            self.mode = StringMode::Normal;
            return Ok(StringEvent::CodePoint(character));
        }

        if (0xD800..=0xDBFF).contains(&next_value) {
            self.mode = StringMode::ExpectLowSurrogateSlash { high: next_value };
            return Ok(StringEvent::Pending);
        }
        if (0xDC00..=0xDFFF).contains(&next_value) {
            return Err(JsonParserError::new(
                "lone low surrogate is not valid JSON Unicode",
            ));
        }
        let character = char::from_u32(u32::from(next_value))
            .ok_or_else(|| JsonParserError::new("Unicode escape is not a scalar value"))?;
        self.mode = StringMode::Normal;
        Ok(StringEvent::CodePoint(character))
    }
}

#[derive(Clone, Copy)]
enum StringMode {
    Normal,
    Escape,
    Unicode { value: u16, digits: u8 },
    ExpectLowSurrogateSlash { high: u16 },
    ExpectLowSurrogateU { high: u16 },
    LowSurrogate { high: u16, value: u16, digits: u8 },
}

enum StringEvent {
    Pending,
    CodePoint(char),
    Closed,
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_schema::compile_json_schema;

    fn parse(schema: &str, chunks: &[&[u8]]) -> Result<JsonParserState, JsonParserError> {
        let compiled = compile_json_schema(schema).unwrap();
        let mut state = JsonParserState::new();
        for chunk in chunks {
            state.consume_token(&compiled, chunk)?;
        }
        Ok(state)
    }

    #[test]
    fn accepts_nested_objects_arrays_and_structural_whitespace() {
        let schema = r#"{"type":"object","properties":{"person":{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}},"required":["name","tags"]}},"required":["person"]}"#;
        let compiled = compile_json_schema(schema).unwrap();
        let mut state = JsonParserState::new();
        state
            .consume_token(
                &compiled,
                b" \n{\"person\": {\"tags\":[\"x\"],\"name\":\"Ada\"}} \t",
            )
            .unwrap();
        assert!(state.is_match(&compiled));
        assert!(!state.is_dead());
    }

    #[test]
    fn rejects_unknown_duplicate_missing_and_trailing_object_properties() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]}"#;
        for invalid in [
            br#"{}"#.as_slice(),
            br#"{"c":1}"#.as_slice(),
            br#"{"a":1,"a":2}"#.as_slice(),
            br#"{"a":1,}"#.as_slice(),
        ] {
            assert!(parse(schema, &[invalid]).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn decodes_property_names_and_keeps_objects_closed() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#;
        let compiled = compile_json_schema(schema).unwrap();
        let state = parse(schema, &[br#"{"\u0061":1}"#]).unwrap();
        assert!(state.is_match(&compiled));

        let empty = r#"{"type":"object"}"#;
        let compiled = compile_json_schema(empty).unwrap();
        assert!(parse(empty, &[b"{}"]).unwrap().is_match(&compiled));
        assert!(parse(empty, &[br#"{"a":1}"#]).is_err());
    }

    #[test]
    fn enforces_typed_array_bounds_and_no_trailing_comma() {
        let schema = r#"{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}"#;
        for valid in [br#"["x"]"#.as_slice(), br#"["x","y"]"#.as_slice()] {
            let compiled = compile_json_schema(schema).unwrap();
            let state = parse(schema, &[valid]).unwrap();
            assert!(state.is_match(&compiled));
        }
        for invalid in [
            br#"[]"#.as_slice(),
            br#"[1]"#.as_slice(),
            br#"["x","y","z"]"#.as_slice(),
            br#"["x",]"#.as_slice(),
        ] {
            assert!(parse(schema, &[invalid]).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn number_machine_is_rfc8259_strict_across_chunks() {
        let number = r#"{"type":"number"}"#;
        let compiled = compile_json_schema(number).unwrap();
        for chunks in [
            vec![b"0".as_slice()],
            vec![b"-0".as_slice()],
            vec![b"12".as_slice(), b".34".as_slice(), b"e-2".as_slice()],
        ] {
            let state = parse(number, &chunks).unwrap();
            assert!(state.is_match(&compiled));
        }
        for invalid in [b"01".as_slice(), b"-01", b"+1", b"1.", b"1e", b"1e+"] {
            let result = parse(number, &[invalid]);
            if matches!(invalid, b"1." | b"1e" | b"1e+") {
                assert!(!result.unwrap().is_match(&compiled));
            } else {
                assert!(result.is_err(), "{invalid:?}");
            }
        }

        let integer = r#"{"type":"integer"}"#;
        assert!(parse(integer, &[b"12.0"]).is_err());
        assert!(parse(integer, &[b"12e1"]).is_err());
    }

    #[test]
    fn decodes_utf8_escapes_and_surrogate_pairs_as_code_points() {
        let schema = r#"{"type":"string","minLength":1,"maxLength":1}"#;
        let compiled = compile_json_schema(schema).unwrap();
        for chunks in [
            vec![b"\"\xC3".as_slice(), b"\xA9\"".as_slice()],
            vec![br#""\u00E9""#.as_slice()],
            vec![b"\"\\uD83D".as_slice(), b"\\uDE00\"".as_slice()],
            vec![br#""\n""#.as_slice()],
        ] {
            let state = parse(schema, &chunks).unwrap();
            assert!(state.is_match(&compiled));
        }

        for invalid in [
            br#""\x""#.as_slice(),
            br#""\uD83D""#.as_slice(),
            br#""\uD83D\u0041""#.as_slice(),
            br#""\uDE00""#.as_slice(),
            b"\"\x80\"".as_slice(),
            b"\"\x01\"".as_slice(),
        ] {
            assert!(parse(schema, &[invalid]).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn accepts_every_simple_escape_and_rejects_partial_content_at_max_length() {
        let schema = r#"{"type":"string","minLength":8,"maxLength":8}"#;
        let compiled = compile_json_schema(schema).unwrap();
        let state = parse(schema, &[br#""\"\\\/\b\f\n\r\t""#]).unwrap();
        assert!(state.is_match(&compiled));

        let empty = r#"{"type":"string","maxLength":0}"#;
        assert!(parse(empty, &[b"\"\\"]).is_err());
        assert!(parse(empty, &[b"\"\xC3"]).is_err());
    }

    #[test]
    fn pattern_receives_decoded_unicode_not_escape_designators() {
        let schema = r#"{"type":"string","pattern":"^\\n$","minLength":1,"maxLength":1}"#;
        let compiled = compile_json_schema(schema).unwrap();
        let state = parse(schema, &[br#""\n""#]).unwrap();
        assert!(state.is_match(&compiled));

        let designator = r#"{"type":"string","pattern":"^n$","minLength":1,"maxLength":1}"#;
        assert!(parse(designator, &[br#""\n""#]).is_err());
    }

    #[test]
    fn root_completion_accepts_only_structural_whitespace() {
        let schema = r#"{"type":"boolean"}"#;
        let compiled = compile_json_schema(schema).unwrap();
        let state = parse(schema, &[b"true \n\t"]).unwrap();
        assert!(state.is_match(&compiled));
        assert!(parse(schema, &[b"true false"]).is_err());
    }

    #[test]
    fn numeric_and_prefix_enum_matches_can_retain_continuations() {
        let number_schema = compile_json_schema(r#"{"type":"number"}"#).unwrap();
        let mut number = JsonParserState::new();
        number.consume_token(&number_schema, b"1").unwrap();
        assert!(number.is_match(&number_schema));
        number.consume_token(&number_schema, b"2").unwrap();
        assert!(number.is_match(&number_schema));

        let enum_schema = compile_json_schema(r#"{"enum":[1,10]}"#).unwrap();
        let mut value = JsonParserState::new();
        value.consume_token(&enum_schema, b"1").unwrap();
        assert!(value.is_match(&enum_schema));
        value.consume_token(&enum_schema, b"0").unwrap();
        assert!(value.is_match(&enum_schema));
    }
}
