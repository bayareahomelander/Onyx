//! JSON schema parsing and blueprint extraction
//!
//! this module parses raw JSON schema definitions and extracts
//! the structural information needed for constraint enforcement.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::constraint::ConstraintError;

fn invalid(path: &str, message: impl std::fmt::Display) -> ConstraintError {
    ConstraintError::CompilationError(format!("JSON schema {path}: {message}"))
}

/// Restrict patterns to a portable subset, with JSON Schema's search semantics.
pub fn schema_pattern(pattern: &str) -> Result<String, ConstraintError> {
    let mut normalized = String::new();
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                let escaped = chars
                    .next()
                    .ok_or_else(|| invalid("pattern", "unfinished escape"))?;
                match escaped {
                    'd' => normalized.push_str("[0-9]"),
                    'D' => normalized.push_str("[^0-9]"),
                    'w' => normalized.push_str("[A-Za-z0-9_]"),
                    'W' => normalized.push_str("[^A-Za-z0-9_]"),
                    'n' | 'r' | 't' | 'f' => {
                        normalized.push('\\');
                        normalized.push(escaped);
                    }
                    c if "\\/.*+?()[]{}^$|-".contains(c) => {
                        if c != '/' {
                            normalized.push('\\');
                        }
                        normalized.push(c);
                    }
                    _ => {
                        return Err(invalid(
                            "pattern",
                            format!("unsupported escape \\{escaped}"),
                        ))
                    }
                }
            }
            '[' if in_class => {
                return Err(invalid(
                    "pattern",
                    "nested character classes are unsupported",
                ))
            }
            '[' => {
                in_class = true;
                normalized.push(ch);
            }
            ']' => {
                in_class = false;
                normalized.push(ch);
            }
            '&' | '-' | '~' if in_class && chars.peek() == Some(&ch) => {
                return Err(invalid(
                    "pattern",
                    "character class set operations are unsupported",
                ));
            }
            '(' if chars.peek() == Some(&'?') => {
                chars.next();
                if chars.next() != Some(':') {
                    return Err(invalid(
                        "pattern",
                        "only ordinary and noncapturing groups are supported",
                    ));
                }
                normalized.push_str("(?:");
            }
            '.' if !in_class => normalized.push_str("[^\\n\\r\\u{2028}\\u{2029}]"),
            _ => normalized.push(ch),
        }
    }
    regex::Regex::new(&normalized).map_err(|e| invalid("pattern", e))?;
    Ok(normalized)
}

/// Fail closed: accepting a schema must never silently discard a validation keyword.
pub fn validate_schema(schema: &Value, path: &str) -> Result<(), ConstraintError> {
    let object = schema
        .as_object()
        .ok_or_else(|| invalid(path, "expected a schema object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "items"
                | "enum"
                | "pattern"
                | "minLength"
                | "maxLength"
                | "minItems"
                | "maxItems"
                | "title"
                | "description"
                | "default"
                | "examples"
                | "$comment"
        ) {
            return Err(invalid(path, format!("unsupported keyword '{key}'")));
        }
    }
    if let Some(value) = schema.get("type") {
        let types: Vec<&Value> = match value {
            Value::String(_) => vec![value],
            Value::Array(values) if !values.is_empty() => values.iter().collect(),
            _ => {
                return Err(invalid(
                    path,
                    "type must be a name or a nonempty array of names",
                ))
            }
        };
        let mut seen = HashSet::new();
        for kind in types {
            let name = kind
                .as_str()
                .ok_or_else(|| invalid(path, "type names must be strings"))?;
            if !matches!(
                name,
                "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
            ) || !seen.insert(name)
            {
                return Err(invalid(path, format!("invalid or duplicate type '{name}'")));
            }
        }
    }
    let types = SchemaType::types_from_value(schema);
    for (kind, keys) in [
        (
            SchemaType::Object,
            &["properties", "required", "additionalProperties"][..],
        ),
        (SchemaType::Array, &["items", "minItems", "maxItems"][..]),
        (
            SchemaType::String,
            &["pattern", "minLength", "maxLength"][..],
        ),
    ] {
        if keys.iter().any(|key| object.contains_key(*key)) && !types.contains(&kind) {
            return Err(invalid(
                path,
                format!("{keys:?} require an explicit {kind:?} type"),
            ));
        }
    }
    if let Some(props) = schema.get("properties") {
        let props = props
            .as_object()
            .ok_or_else(|| invalid(path, "properties must be an object"))?;
        for (name, child) in props {
            validate_schema(child, &format!("{path}.properties[{name:?}]"))?;
        }
    }
    if let Some(required) = schema.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| invalid(path, "required must be an array"))?;
        let mut seen = HashSet::new();
        for name in required {
            let name = name
                .as_str()
                .ok_or_else(|| invalid(path, "required names must be strings"))?;
            if !seen.insert(name) || schema.get("properties").and_then(|p| p.get(name)).is_none() {
                return Err(invalid(
                    path,
                    "required names must be unique and declared in properties",
                ));
            }
        }
    }
    if schema
        .get("additionalProperties")
        .is_some_and(|v| !v.is_boolean())
    {
        return Err(invalid(
            path,
            "only boolean additionalProperties is supported",
        ));
    }
    if let Some(items) = schema.get("items") {
        validate_schema(items, &format!("{path}.items"))?;
    }
    for (min_key, max_key) in [("minLength", "maxLength"), ("minItems", "maxItems")] {
        for key in [min_key, max_key] {
            if schema.get(key).is_some_and(|v| v.as_u64().is_none()) {
                return Err(invalid(
                    path,
                    format!("{key} must be a nonnegative integer"),
                ));
            }
        }
        if let (Some(min), Some(max)) = (
            schema.get(min_key).and_then(Value::as_u64),
            schema.get(max_key).and_then(Value::as_u64),
        ) {
            if min > max {
                return Err(invalid(path, format!("{min_key} exceeds {max_key}")));
            }
        }
    }
    if let Some(pattern) = schema.get("pattern") {
        let pattern = pattern
            .as_str()
            .ok_or_else(|| invalid(path, "pattern must be a string"))?;
        let normalized = schema_pattern(pattern).map_err(|e| invalid(path, e))?;
        crate::regex_engine::compile_pattern_dfa(&format!("(?s:.*(?:{normalized}).*)"))
            .map_err(|e| invalid(path, e))?;
    }
    if let Some(values) = schema.get("enum") {
        let values = values
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid(path, "enum must be a nonempty array"))?;
        if values
            .iter()
            .enumerate()
            .any(|(i, value)| values[..i].contains(value))
        {
            return Err(invalid(path, "enum values must be unique"));
        }
        if !values
            .iter()
            .any(|value| value_matches_schema(value, schema))
        {
            return Err(invalid(
                path,
                "enum has no value satisfying the other constraints",
            ));
        }
    }
    Ok(())
}

fn is_integer(value: &Value) -> bool {
    let Value::Number(number) = value else {
        return false;
    };
    // Decimal placement, not a floating-point conversion, determines integrality.
    let text = number.to_string();
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((&text, "0"));
    if mantissa.chars().all(|c| matches!(c, '-' | '.' | '0')) {
        return true;
    }
    let exponent = exponent.parse::<i64>().unwrap_or_else(|_| {
        if exponent.starts_with('-') {
            i64::MIN
        } else {
            i64::MAX
        }
    });
    let decimals = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let trailing_zeros = mantissa
        .chars()
        .rev()
        .filter(|&c| c != '.')
        .take_while(|&c| c == '0')
        .count();
    exponent >= decimals as i64 - trailing_zeros as i64
}

/// Used for intersecting enums with their sibling constraints and checking completion.
pub fn value_matches_schema(value: &Value, schema: &Value) -> bool {
    let types = SchemaType::types_from_value(schema);
    if !types.iter().any(|kind| match kind {
        SchemaType::Any => true,
        SchemaType::Object => value.is_object(),
        SchemaType::Array => value.is_array(),
        SchemaType::String => value.is_string(),
        SchemaType::Number => value.is_number(),
        SchemaType::Integer => is_integer(value),
        SchemaType::Boolean => value.is_boolean(),
        SchemaType::Null => value.is_null(),
    }) {
        return false;
    }
    if schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| !values.contains(value))
    {
        return false;
    }
    if let Some(text) = value.as_str() {
        let length = text.chars().count() as u64;
        if schema
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|min| length < min)
            || schema
                .get("maxLength")
                .and_then(Value::as_u64)
                .is_some_and(|max| length > max)
        {
            return false;
        }
        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            if !schema_pattern(pattern)
                .ok()
                .and_then(|p| regex::Regex::new(&p).ok())
                .is_some_and(|p| p.is_match(text))
            {
                return false;
            }
        }
    }
    if let Some(items) = value.as_array() {
        let length = items.len() as u64;
        if schema
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|min| length < min)
            || schema
                .get("maxItems")
                .and_then(Value::as_u64)
                .is_some_and(|max| length > max)
        {
            return false;
        }
        if let Some(child) = schema.get("items") {
            if !items.iter().all(|item| value_matches_schema(item, child)) {
                return false;
            }
        }
    }
    if let Some(object) = value.as_object() {
        if schema
            .get("required")
            .and_then(Value::as_array)
            .is_some_and(|keys| {
                keys.iter()
                    .any(|k| !object.contains_key(k.as_str().unwrap()))
            })
        {
            return false;
        }
        for (key, item) in object {
            if let Some(child) = schema.get("properties").and_then(|p| p.get(key)) {
                if !value_matches_schema(item, child) {
                    return false;
                }
            } else if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                return false;
            }
        }
    }
    true
}

/// the type of a JSON schema node
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
    Any, // when type is not specified
}

impl SchemaType {
    /// parse a schema type from a type string
    pub fn from_str(s: &str) -> Self {
        match s {
            "object" => SchemaType::Object,
            "array" => SchemaType::Array,
            "string" => SchemaType::String,
            "number" => SchemaType::Number,
            "integer" => SchemaType::Integer,
            "boolean" => SchemaType::Boolean,
            "null" => SchemaType::Null,
            _ => SchemaType::Any,
        }
    }

    /// parse a single schema type from a JSON value (legacy)
    pub fn from_value(value: &Value) -> Self {
        match value.get("type").and_then(|v| v.as_str()) {
            Some(s) => Self::from_str(s),
            _ => SchemaType::Any,
        }
    }

    /// parse potentially multiple types from a JSON value
    /// handles both "type": "string" and "type": ["string", "null"]
    pub fn types_from_value(value: &Value) -> Vec<Self> {
        match value.get("type") {
            Some(Value::String(s)) => vec![Self::from_str(s)],
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(Self::from_str)
                .collect(),
            _ => vec![SchemaType::Any],
        }
    }
}

/// a parsed property definition from a JSON schema
#[derive(Debug, Clone)]
pub struct PropertyBlueprint {
    /// allowed types (supports union types)
    pub schema_types: Vec<SchemaType>,
    pub properties: HashMap<String, PropertyBlueprint>,
    /// for arrays: item schema
    pub items: Option<Arc<PropertyBlueprint>>,
    pub required: bool,
    /// for enums: allowed values (serialized json bytes)
    pub enum_values: Option<Vec<Vec<u8>>>,
    /// for strings: regex pattern
    pub pattern: Option<String>,
    /// for strings: min char count
    pub min_length: Option<usize>,
    /// for strings: max char count
    pub max_length: Option<usize>,
    /// for arrays: min item count
    pub min_items: Option<usize>,
    /// for arrays: max item count
    pub max_items: Option<usize>,
    pub object_blueprint: Option<Arc<SchemaBlueprint>>,
}

impl PropertyBlueprint {
    /// create a property blueprint from a JSON schema value
    pub fn from_value(_name: &str, value: &Value) -> Self {
        let schema_types = SchemaType::types_from_value(value);

        let mut properties = HashMap::new();
        // parse nested properties if any type is Object
        if schema_types.contains(&SchemaType::Object) {
            if let Some(props) = value.get("properties").and_then(|v| v.as_object()) {
                for (prop_name, prop_value) in props {
                    properties.insert(
                        prop_name.clone(),
                        PropertyBlueprint::from_value(prop_name, prop_value),
                    );
                }
            }
        }

        let mut required_keys = HashSet::new();
        if schema_types.contains(&SchemaType::Object) {
            if let Some(req_array) = value.get("required").and_then(|v| v.as_array()) {
                for item in req_array {
                    if let Some(name) = item.as_str() {
                        required_keys.insert(name.to_string());
                        if let Some(prop) = properties.get_mut(name) {
                            prop.required = true;
                        }
                    }
                }
            }
        }

        // parse items for arrays
        let items = if schema_types.contains(&SchemaType::Array) {
            value
                .get("items")
                .map(|item_schema| Arc::new(PropertyBlueprint::from_value("_item", item_schema)))
        } else {
            None
        };

        // parse enum values if present
        let enum_values = value.get("enum").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter(|v| value_matches_schema(v, value))
                .filter_map(|v| serde_json::to_vec(v).ok())
                .collect()
        });

        // parse pattern for string regex constraints
        let pattern = value
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // parse string length constraints
        let min_length = value
            .get("minLength")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let max_length = value
            .get("maxLength")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        // parse array length constraints
        let min_items = value
            .get("minItems")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let max_items = value
            .get("maxItems")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        let object_blueprint = if schema_types.contains(&SchemaType::Object) {
            let mut allowed_keys: Vec<String> = properties.keys().cloned().collect();
            allowed_keys.sort();
            Some(Arc::new(SchemaBlueprint {
                root_type: SchemaType::Object,
                properties: properties.clone(),
                required: required_keys.clone(),
                allowed_keys,
            }))
        } else {
            None
        };

        PropertyBlueprint {
            schema_types,
            properties,
            items,
            required: false,
            enum_values,
            pattern,
            min_length,
            max_length,
            min_items,
            max_items,
            object_blueprint,
        }
    }
}

/// a parsed JSON schema blueprint
///
/// this struct extracts and stores the structural information
/// from a JSON schema that is needed for constraint enforcement
#[derive(Debug, Clone)]
pub struct SchemaBlueprint {
    pub root_type: SchemaType,
    pub properties: HashMap<String, PropertyBlueprint>,
    pub required: HashSet<String>,
    pub allowed_keys: Vec<String>,
}

impl SchemaBlueprint {
    /// parse a JSON schema string and extract its blueprint
    pub fn from_json(schema: &str) -> Result<Self, ConstraintError> {
        let value = serde_json::from_str(schema).map_err(|error| {
            ConstraintError::CompilationError(format!("Failed to parse JSON schema: {error}"))
        })?;
        Self::from_value(&value)
    }

    /// parse a JSON schema from a serde_json::Value
    pub fn from_value(schema: &Value) -> Result<Self, ConstraintError> {
        validate_schema(schema, "$")?;
        let root_type = SchemaType::from_value(schema);

        let mut properties = HashMap::new();
        let mut allowed_keys = Vec::new();

        if root_type == SchemaType::Object {
            if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
                for (name, prop_value) in props {
                    properties.insert(
                        name.clone(),
                        PropertyBlueprint::from_value(name, prop_value),
                    );
                    allowed_keys.push(name.clone());
                }
            }
        }

        // Extract required fields
        let mut required = HashSet::new();
        if let Some(req_array) = schema.get("required").and_then(|v| v.as_array()) {
            for item in req_array {
                if let Some(name) = item.as_str() {
                    required.insert(name.to_string());
                    if let Some(prop) = properties.get_mut(name) {
                        prop.required = true;
                    }
                }
            }
        }

        // Sort keys for consistent ordering
        allowed_keys.sort();

        Ok(SchemaBlueprint {
            root_type,
            properties,
            required,
            allowed_keys,
        })
    }

    /// get the blueprint for a specific property
    pub fn get_property(&self, name: &str) -> Option<&PropertyBlueprint> {
        self.properties.get(name)
    }

    /// check if a key is allowed
    pub fn is_key_allowed(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }

    /// check if a string is a valid prefix of any allowed key
    pub fn is_valid_key_prefix(&self, prefix: &str) -> bool {
        self.allowed_keys.iter().any(|key| key.starts_with(prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_schema_blueprint_simple_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });

        let blueprint = SchemaBlueprint::from_value(&schema).unwrap();

        assert_eq!(blueprint.root_type, SchemaType::Object);
        assert_eq!(blueprint.allowed_keys.len(), 2);
        assert!(blueprint.is_key_allowed("name"));
        assert!(blueprint.is_key_allowed("age"));
        assert!(!blueprint.is_key_allowed("unknown"));
    }

    #[test]
    fn test_schema_blueprint_prefix_matching() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
                "age": {"type": "integer"}
            }
        });

        let blueprint = SchemaBlueprint::from_value(&schema).unwrap();

        assert!(blueprint.is_valid_key_prefix("n"));
        assert!(blueprint.is_valid_key_prefix("na"));
        assert!(blueprint.is_valid_key_prefix("name"));
        assert!(blueprint.is_valid_key_prefix("nick"));
        assert!(!blueprint.is_valid_key_prefix("namex"));
        assert!(!blueprint.is_valid_key_prefix("z"));
    }

    #[test]
    fn test_schema_blueprint_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        let blueprint = SchemaBlueprint::from_value(&schema).unwrap();

        assert!(blueprint.required.contains("name"));
        assert!(!blueprint.required.contains("age"));
        assert!(blueprint.properties.get("name").unwrap().required);
        assert!(!blueprint.properties.get("age").unwrap().required);
    }

    #[test]
    fn test_schema_blueprint_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        });

        let blueprint = SchemaBlueprint::from_value(&schema).unwrap();

        let person_prop = blueprint.get_property("person").unwrap();
        assert!(person_prop.schema_types.contains(&SchemaType::Object));
        assert!(person_prop.properties.contains_key("name"));
        assert!(person_prop.properties.contains_key("age"));
    }

    #[test]
    fn test_schema_blueprint_rejects_malformed_json() {
        let error = SchemaBlueprint::from_json(r#"{"type":"object""#).unwrap_err();
        assert!(matches!(
            error,
            ConstraintError::CompilationError(message)
                if message.starts_with("Failed to parse JSON schema:")
        ));
    }
}
