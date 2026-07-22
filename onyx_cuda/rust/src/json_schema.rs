//! Strict compilation of the Deliverable 19 Windows JSON Schema subset.

use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start::Config as StartConfig;
use regex_automata::{Anchored, MatchKind};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

pub(crate) type SchemaNodeId = usize;

const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "enum",
    "pattern",
    "minLength",
    "maxLength",
    "items",
    "minItems",
    "maxItems",
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum JsonValueType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl JsonValueType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "object" => Some(Self::Object),
            "array" => Some(Self::Array),
            "string" => Some(Self::String),
            "number" => Some(Self::Number),
            "integer" => Some(Self::Integer),
            "boolean" => Some(Self::Boolean),
            "null" => Some(Self::Null),
            _ => None,
        }
    }

    fn from_json_value(value: &Value) -> Self {
        match value {
            Value::Object(_) => Self::Object,
            Value::Array(_) => Self::Array,
            Value::String(_) => Self::String,
            Value::Number(number) if number.is_i64() || number.is_u64() => Self::Integer,
            Value::Number(_) => Self::Number,
            Value::Bool(_) => Self::Boolean,
            Value::Null => Self::Null,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct JsonSchemaError {
    message: String,
}

impl JsonSchemaError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for JsonSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for JsonSchemaError {}

pub(crate) struct CompiledPattern {
    dfa: dense::DFA<Vec<u32>>,
    initial_state: StateID,
}

impl CompiledPattern {
    fn compile(pattern: &str, path: &str) -> Result<Self, JsonSchemaError> {
        let config = dense::Config::new()
            .start_kind(StartKind::Anchored)
            .match_kind(MatchKind::LeftmostFirst);
        let dfa = dense::Builder::new()
            .configure(config)
            .build(pattern)
            .map_err(|error| {
                JsonSchemaError::new(format!(
                    "{path}.pattern is not a valid Windows regex-automata pattern: {error}"
                ))
            })?;
        let start = StartConfig::new().anchored(Anchored::Yes);
        let initial_state = dfa.start_state(&start).map_err(|error| {
            JsonSchemaError::new(format!(
                "{path}.pattern could not derive an anchored start state: {error}"
            ))
        })?;
        Ok(Self { dfa, initial_state })
    }

    pub(crate) fn initial_state(&self) -> StateID {
        self.initial_state
    }

    pub(crate) fn advance(&self, mut state: StateID, bytes: &[u8]) -> Option<StateID> {
        for &byte in bytes {
            state = self.dfa.next_state(state, byte);
            if self.dfa.is_dead_state(state) {
                return None;
            }
        }
        Some(state)
    }

    pub(crate) fn is_match(&self, state: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(state))
    }
}

pub(crate) struct CompiledProperty {
    pub(crate) name: String,
    pub(crate) schema: SchemaNodeId,
    pub(crate) required: bool,
}

pub(crate) struct ObjectConstraints {
    pub(crate) properties: Vec<CompiledProperty>,
}

pub(crate) struct ArrayConstraints {
    pub(crate) items: SchemaNodeId,
    pub(crate) min_items: Option<usize>,
    pub(crate) max_items: Option<usize>,
}

pub(crate) struct StringConstraints {
    pub(crate) pattern: Option<Arc<CompiledPattern>>,
    pub(crate) min_length: Option<usize>,
    pub(crate) max_length: Option<usize>,
}

pub(crate) struct SchemaNode {
    pub(crate) allowed_types: Vec<JsonValueType>,
    pub(crate) enum_candidates: Option<Arc<[Vec<u8>]>>,
    pub(crate) object: Option<ObjectConstraints>,
    pub(crate) array: Option<ArrayConstraints>,
    pub(crate) string: Option<StringConstraints>,
}

pub(crate) struct CompiledSchema {
    nodes: Vec<SchemaNode>,
    root: SchemaNodeId,
}

impl CompiledSchema {
    pub(crate) fn root(&self) -> SchemaNodeId {
        self.root
    }

    pub(crate) fn node(&self, node_id: SchemaNodeId) -> &SchemaNode {
        &self.nodes[node_id]
    }

    fn value_matches_node(&self, node_id: SchemaNodeId, value: &Value, enforce_enum: bool) -> bool {
        let node = self.node(node_id);
        if enforce_enum {
            if let Some(candidates) = &node.enum_candidates {
                let Ok(serialized) = serde_json::to_vec(value) else {
                    return false;
                };
                if !candidates.contains(&serialized) {
                    return false;
                }
            }
        }

        node.allowed_types
            .iter()
            .any(|value_type| self.value_matches_type(node, *value_type, value))
    }

    fn value_matches_type(
        &self,
        node: &SchemaNode,
        value_type: JsonValueType,
        value: &Value,
    ) -> bool {
        match (value_type, value) {
            (JsonValueType::Null, Value::Null) => true,
            (JsonValueType::Boolean, Value::Bool(_)) => true,
            (JsonValueType::Number, Value::Number(_)) => true,
            (JsonValueType::Integer, Value::Number(number)) => number.is_i64() || number.is_u64(),
            (JsonValueType::String, Value::String(text)) => node
                .string
                .as_ref()
                .is_none_or(|constraints| string_matches(constraints, text)),
            (JsonValueType::Array, Value::Array(values)) => {
                let Some(constraints) = &node.array else {
                    return true;
                };
                if constraints
                    .min_items
                    .is_some_and(|minimum| values.len() < minimum)
                    || constraints
                        .max_items
                        .is_some_and(|maximum| values.len() > maximum)
                {
                    return false;
                }
                values
                    .iter()
                    .all(|item| self.value_matches_node(constraints.items, item, true))
            }
            (JsonValueType::Object, Value::Object(values)) => {
                let Some(constraints) = &node.object else {
                    return true;
                };
                for key in values.keys() {
                    if constraints
                        .properties
                        .binary_search_by(|property| property.name.as_str().cmp(key.as_str()))
                        .is_err()
                    {
                        return false;
                    }
                }
                constraints.properties.iter().all(|property| {
                    values
                        .get(&property.name)
                        .map_or(!property.required, |child| {
                            self.value_matches_node(property.schema, child, true)
                        })
                })
            }
            _ => false,
        }
    }
}

struct PendingEnumValidation {
    node_id: SchemaNodeId,
    values: Vec<Value>,
    path: String,
}

struct SchemaBuilder {
    nodes: Vec<SchemaNode>,
    pending_enums: Vec<PendingEnumValidation>,
}

impl SchemaBuilder {
    fn compile_node(&mut self, value: &Value, path: &str) -> Result<SchemaNodeId, JsonSchemaError> {
        let object = value
            .as_object()
            .ok_or_else(|| JsonSchemaError::new(format!("{path} must be a schema object")))?;
        for keyword in object.keys() {
            if !SUPPORTED_KEYWORDS.contains(&keyword.as_str()) {
                return Err(JsonSchemaError::new(format!(
                    "{path} contains unsupported keyword {keyword:?}"
                )));
            }
        }

        let explicit_type = object.contains_key("type");
        let enum_values = parse_enum_values(object, path)?;
        if !explicit_type && enum_values.is_none() {
            return Err(JsonSchemaError::new(format!(
                "{path} must contain type or enum; unconstrained schemas are unsupported"
            )));
        }
        if !explicit_type && object.keys().any(|keyword| keyword != "enum") {
            return Err(JsonSchemaError::new(format!(
                "{path} may omit type only when enum is its sole keyword"
            )));
        }

        let allowed_types = if explicit_type {
            parse_explicit_types(object, path)?
        } else {
            let mut types = BTreeSet::new();
            for value in enum_values.as_ref().expect("enum presence was checked") {
                types.insert(JsonValueType::from_json_value(value));
            }
            types.into_iter().collect()
        };

        validate_keyword_applicability(object, path, &allowed_types)?;

        let object_constraints = if explicit_type && allowed_types.contains(&JsonValueType::Object)
        {
            Some(self.compile_object_constraints(object, path)?)
        } else {
            None
        };
        let array_constraints = if explicit_type && allowed_types.contains(&JsonValueType::Array) {
            Some(self.compile_array_constraints(object, path)?)
        } else {
            None
        };
        let string_constraints = if explicit_type && allowed_types.contains(&JsonValueType::String)
        {
            Some(compile_string_constraints(object, path)?)
        } else {
            None
        };

        let enum_candidates = enum_values.as_ref().map(|values| {
            let candidates = values
                .iter()
                .map(|candidate| {
                    serde_json::to_vec(candidate)
                        .expect("serde_json Value serialization cannot fail")
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            Arc::<[Vec<u8>]>::from(candidates)
        });

        let node_id = self.nodes.len();
        self.nodes.push(SchemaNode {
            allowed_types,
            enum_candidates,
            object: object_constraints,
            array: array_constraints,
            string: string_constraints,
        });
        if explicit_type {
            if let Some(values) = enum_values {
                self.pending_enums.push(PendingEnumValidation {
                    node_id,
                    values,
                    path: path.to_string(),
                });
            }
        }
        Ok(node_id)
    }

    fn compile_object_constraints(
        &mut self,
        object: &Map<String, Value>,
        path: &str,
    ) -> Result<ObjectConstraints, JsonSchemaError> {
        let required = parse_required(object, path)?;
        let properties = match object.get("properties") {
            Some(Value::Object(properties)) => properties,
            Some(_) => {
                return Err(JsonSchemaError::new(format!(
                    "{path}.properties must be an object"
                )));
            }
            None => {
                if let Some(name) = required.first() {
                    return Err(JsonSchemaError::new(format!(
                        "{path}.required entry {name:?} is not declared in properties"
                    )));
                }
                return Ok(ObjectConstraints {
                    properties: Vec::new(),
                });
            }
        };

        for name in &required {
            if !properties.contains_key(name) {
                return Err(JsonSchemaError::new(format!(
                    "{path}.required entry {name:?} is not declared in properties"
                )));
            }
        }

        let mut sorted_properties = properties.iter().collect::<Vec<_>>();
        sorted_properties.sort_by_key(|(name, _)| (*name).clone());
        let mut compiled = Vec::with_capacity(sorted_properties.len());
        for (name, child) in sorted_properties {
            let child_path = format!("{path}.properties[{name:?}]");
            compiled.push(CompiledProperty {
                name: name.clone(),
                schema: self.compile_node(child, &child_path)?,
                required: required.contains(name),
            });
        }
        Ok(ObjectConstraints {
            properties: compiled,
        })
    }

    fn compile_array_constraints(
        &mut self,
        object: &Map<String, Value>,
        path: &str,
    ) -> Result<ArrayConstraints, JsonSchemaError> {
        let item_schema = object.get("items").ok_or_else(|| {
            JsonSchemaError::new(format!(
                "{path}.items is required for the first supported typed-array subset"
            ))
        })?;
        let min_items = parse_limit(object, "minItems", path)?;
        let max_items = parse_limit(object, "maxItems", path)?;
        validate_ordered_limits(min_items, max_items, path, "minItems", "maxItems")?;
        Ok(ArrayConstraints {
            items: self.compile_node(item_schema, &format!("{path}.items"))?,
            min_items,
            max_items,
        })
    }
}

pub(crate) fn compile_json_schema(schema: &str) -> Result<Arc<CompiledSchema>, JsonSchemaError> {
    if schema.trim().is_empty() {
        return Err(JsonSchemaError::new("JSON Schema cannot be empty"));
    }
    let value: Value = serde_json::from_str(schema)
        .map_err(|error| JsonSchemaError::new(format!("JSON Schema is not valid JSON: {error}")))?;
    if !value.is_object() {
        return Err(JsonSchemaError::new("JSON Schema root must be an object"));
    }

    let mut builder = SchemaBuilder {
        nodes: Vec::new(),
        pending_enums: Vec::new(),
    };
    let root = builder.compile_node(&value, "$")?;
    let compiled = CompiledSchema {
        nodes: builder.nodes,
        root,
    };
    for pending in builder.pending_enums {
        for candidate in pending.values {
            if !compiled.value_matches_node(pending.node_id, &candidate, false) {
                return Err(JsonSchemaError::new(format!(
                    "{}.enum candidate {} does not satisfy its declared type and constraints",
                    pending.path, candidate
                )));
            }
        }
    }
    Ok(Arc::new(compiled))
}

fn parse_explicit_types(
    object: &Map<String, Value>,
    path: &str,
) -> Result<Vec<JsonValueType>, JsonSchemaError> {
    let value = object
        .get("type")
        .expect("explicit type presence was checked");
    let names = match value {
        Value::String(name) => vec![name.as_str()],
        Value::Array(values) => {
            if values.is_empty() {
                return Err(JsonSchemaError::new(format!(
                    "{path}.type union cannot be empty"
                )));
            }
            values
                .iter()
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        JsonSchemaError::new(format!("{path}.type union entries must be strings"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(JsonSchemaError::new(format!(
                "{path}.type must be a string or nonempty string array"
            )));
        }
    };

    let mut types = BTreeSet::new();
    for name in names {
        let value_type = JsonValueType::parse(name).ok_or_else(|| {
            JsonSchemaError::new(format!("{path}.type contains unsupported value {name:?}"))
        })?;
        if !types.insert(value_type) {
            return Err(JsonSchemaError::new(format!(
                "{path}.type contains duplicate value {name:?}"
            )));
        }
    }
    if types.contains(&JsonValueType::Number) {
        types.remove(&JsonValueType::Integer);
    }
    Ok(types.into_iter().collect())
}

fn parse_enum_values(
    object: &Map<String, Value>,
    path: &str,
) -> Result<Option<Vec<Value>>, JsonSchemaError> {
    match object.get("enum") {
        None => Ok(None),
        Some(Value::Array(values)) if values.is_empty() => {
            Err(JsonSchemaError::new(format!("{path}.enum cannot be empty")))
        }
        Some(Value::Array(values)) => Ok(Some(values.clone())),
        Some(_) => Err(JsonSchemaError::new(format!(
            "{path}.enum must be a nonempty array"
        ))),
    }
}

fn parse_required(
    object: &Map<String, Value>,
    path: &str,
) -> Result<BTreeSet<String>, JsonSchemaError> {
    let Some(value) = object.get("required") else {
        return Ok(BTreeSet::new());
    };
    let values = value.as_array().ok_or_else(|| {
        JsonSchemaError::new(format!(
            "{path}.required must be an array of unique strings"
        ))
    })?;
    let mut required = BTreeSet::new();
    for value in values {
        let name = value.as_str().ok_or_else(|| {
            JsonSchemaError::new(format!("{path}.required entries must be strings"))
        })?;
        if !required.insert(name.to_string()) {
            return Err(JsonSchemaError::new(format!(
                "{path}.required contains duplicate entry {name:?}"
            )));
        }
    }
    Ok(required)
}

fn compile_string_constraints(
    object: &Map<String, Value>,
    path: &str,
) -> Result<StringConstraints, JsonSchemaError> {
    let pattern = match object.get("pattern") {
        None => None,
        Some(Value::String(pattern)) => Some(Arc::new(CompiledPattern::compile(pattern, path)?)),
        Some(_) => {
            return Err(JsonSchemaError::new(format!(
                "{path}.pattern must be a string"
            )));
        }
    };
    let min_length = parse_limit(object, "minLength", path)?;
    let max_length = parse_limit(object, "maxLength", path)?;
    validate_ordered_limits(min_length, max_length, path, "minLength", "maxLength")?;
    Ok(StringConstraints {
        pattern,
        min_length,
        max_length,
    })
}

fn parse_limit(
    object: &Map<String, Value>,
    keyword: &str,
    path: &str,
) -> Result<Option<usize>, JsonSchemaError> {
    let Some(value) = object.get(keyword) else {
        return Ok(None);
    };
    let raw = value.as_u64().ok_or_else(|| {
        JsonSchemaError::new(format!("{path}.{keyword} must be a nonnegative integer"))
    })?;
    usize::try_from(raw).map(Some).map_err(|_| {
        JsonSchemaError::new(format!("{path}.{keyword} is too large for this platform"))
    })
}

fn validate_ordered_limits(
    minimum: Option<usize>,
    maximum: Option<usize>,
    path: &str,
    minimum_name: &str,
    maximum_name: &str,
) -> Result<(), JsonSchemaError> {
    if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
        if minimum > maximum {
            return Err(JsonSchemaError::new(format!(
                "{path}.{minimum_name} cannot exceed {maximum_name}"
            )));
        }
    }
    Ok(())
}

fn validate_keyword_applicability(
    object: &Map<String, Value>,
    path: &str,
    types: &[JsonValueType],
) -> Result<(), JsonSchemaError> {
    validate_keyword_group(
        object,
        path,
        types,
        JsonValueType::Object,
        &["properties", "required"],
    )?;
    validate_keyword_group(
        object,
        path,
        types,
        JsonValueType::Array,
        &["items", "minItems", "maxItems"],
    )?;
    validate_keyword_group(
        object,
        path,
        types,
        JsonValueType::String,
        &["pattern", "minLength", "maxLength"],
    )
}

fn validate_keyword_group(
    object: &Map<String, Value>,
    path: &str,
    types: &[JsonValueType],
    required_type: JsonValueType,
    keywords: &[&str],
) -> Result<(), JsonSchemaError> {
    if !types.contains(&required_type) {
        if let Some(keyword) = keywords
            .iter()
            .find(|keyword| object.contains_key(**keyword))
        {
            return Err(JsonSchemaError::new(format!(
                "{path}.{keyword} does not apply without type {required_type:?}"
            )));
        }
    }
    Ok(())
}

fn string_matches(constraints: &StringConstraints, value: &str) -> bool {
    let length = value.chars().count();
    if constraints
        .min_length
        .is_some_and(|minimum| length < minimum)
        || constraints
            .max_length
            .is_some_and(|maximum| length > maximum)
    {
        return false;
    }
    let Some(pattern) = &constraints.pattern else {
        return true;
    };
    let mut state = pattern.initial_state();
    for character in value.chars() {
        let mut encoded = [0_u8; 4];
        let bytes = character.encode_utf8(&mut encoded).as_bytes();
        let Some(next) = pattern.advance(state, bytes) else {
            return false;
        };
        state = next;
    }
    pattern.is_match(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARITY_SCHEMAS: &[&str] = &[
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]}"#,
        r#"{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}"#,
        r#"{"type":"string","pattern":"^[A-Z]+$","minLength":2,"maxLength":2}"#,
        r#"{"type":"number"}"#,
        r#"{"type":"integer"}"#,
        r#"{"type":["boolean","null"]}"#,
        r#"{"enum":["red","blue"]}"#,
        r#"{"type":"object","properties":{"person":{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string"}}},"required":["name","tags"]}},"required":["person"]}"#,
        r#"{"type":"string","minLength":1,"maxLength":1}"#,
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
    ];

    #[test]
    fn compiles_every_windows_parity_schema() {
        for schema in PARITY_SCHEMAS {
            compile_json_schema(schema).unwrap_or_else(|error| panic!("{schema}: {error}"));
        }
    }

    #[test]
    fn rejects_malformed_unsupported_and_silently_weakened_schemas() {
        let rejected = [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"type":"string","const":"x"}"#,
            r##"{"$ref":"#/defs/value","defs":{"value":{"type":"string"}}}"##,
            r#"{"oneOf":[{"type":"string"},{"type":"null"}]}"#,
            r#"{"type":"mystery"}"#,
            r#"{"type":"string","pattern":"("}"#,
            r#"{"type":"string","minLength":"2"}"#,
            r#"{"type":"number","minimum":10}"#,
            r#"{"type":"object","properties":{},"additionalProperties":true}"#,
            r#"{"type":"object","properties":{},"required":["x"]}"#,
            r#"{"type":"object","required":["x"]}"#,
        ];
        for schema in rejected {
            assert!(compile_json_schema(schema).is_err(), "{schema}");
        }
    }

    #[test]
    fn enforces_recursive_keyword_types_and_applicability() {
        let rejected = [
            r#"{"type":1}"#,
            r#"{"type":["string","string"]}"#,
            r#"{"type":["string",1]}"#,
            r#"{"type":["string","mystery"]}"#,
            r#"{"type":[]}"#,
            r#"{"type":"array"}"#,
            r#"{"type":"number","pattern":".*"}"#,
            r#"{"type":"string","properties":{}}"#,
            r#"{"type":"object","items":{"type":"string"}}"#,
            r#"{"type":"object","minLength":1}"#,
            r#"{"type":"object","properties":[]}"#,
            r#"{"type":"object","properties":{"x":{}}}"#,
            r#"{"type":"object","properties":{"x":{"type":"string","const":"x"}}}"#,
            r#"{"type":"object","properties":{"x":{"type":"string"}},"required":"x"}"#,
            r#"{"type":"object","properties":{"x":{"type":"string"}},"required":[1]}"#,
            r#"{"type":"object","properties":{"x":{"type":"string"}},"required":["x","x"]}"#,
            r#"{"type":"array","items":[]}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":-1}"#,
            r#"{"type":"array","items":{"type":"string"},"maxItems":1.5}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":1}"#,
            r#"{"type":"string","pattern":1}"#,
            r#"{"type":"string","minLength":-1}"#,
            r#"{"type":"string","maxLength":true}"#,
            r#"{"type":"string","minLength":2,"maxLength":1}"#,
            r#"{"enum":[]}"#,
            r#"{"enum":"x"}"#,
            r#"{"enum":["x"],"minLength":1}"#,
        ];
        for schema in rejected {
            assert!(compile_json_schema(schema).is_err(), "{schema}");
        }
    }

    #[test]
    fn validates_enum_intersection_and_deduplicates_candidates() {
        let schema = compile_json_schema(
            r#"{"type":"string","enum":["AA","AA"],"pattern":"^[A-Z]+$","minLength":2,"maxLength":2}"#,
        )
        .unwrap();
        let node = schema.node(schema.root());
        assert_eq!(node.enum_candidates.as_ref().unwrap().len(), 1);

        assert!(compile_json_schema(r#"{"type":"integer","enum":[1,"x"]}"#).is_err());
        assert!(
            compile_json_schema(r#"{"type":"string","enum":["a"],"pattern":"^[A-Z]+$"}"#).is_err()
        );
    }

    #[test]
    fn enum_only_values_cover_the_recorded_json_value_surface() {
        let schema =
            compile_json_schema(r#"{"enum":[null,true,2,"x",[],{},[1],{"a":1}]}"#).unwrap();
        let node = schema.node(schema.root());
        assert_eq!(node.enum_candidates.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn number_integer_union_normalizes_to_number() {
        let schema = compile_json_schema(r#"{"type":["integer","number"]}"#).unwrap();
        assert_eq!(
            schema.node(schema.root()).allowed_types,
            vec![JsonValueType::Number]
        );
    }

    #[test]
    fn accepts_closed_empty_objects_empty_required_sets_and_typed_unions() {
        for schema in [
            r#"{"type":"object"}"#,
            r#"{"type":"object","required":[]}"#,
            r#"{"type":["string","null"],"pattern":"^[a-z]*$"}"#,
            r#"{"type":["object","string"],"properties":{"a":{"type":"integer"}}}"#,
            r#"{"type":"array","items":{"enum":[null,true,2,"x",[],{}]}}"#,
        ] {
            compile_json_schema(schema).unwrap_or_else(|error| panic!("{schema}: {error}"));
        }
    }
}
