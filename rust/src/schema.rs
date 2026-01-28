//! JSON schema parsing and blueprint extraction
//!
//! this module parses raw JSON schema definitions and extracts
//! the structural information needed for constraint enforcement.

use std::collections::{HashMap, HashSet};
use serde_json::Value;

use crate::constraint::ConstraintError;

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
            Some(Value::Array(arr)) => {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(Self::from_str)
                    .collect()
            }
            _ => vec![SchemaType::Any],
        }
    }
}

/// a parsed property definition from a JSON schema
#[derive(Debug, Clone)]
pub struct PropertyBlueprint {
    pub name: String,
    /// allowed types (supports union types)
    pub schema_types: Vec<SchemaType>,
    pub properties: HashMap<String, PropertyBlueprint>,
    /// for arrays: item schema
    pub items: Option<Box<PropertyBlueprint>>,
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
}

impl PropertyBlueprint {
    /// create a property blueprint from a JSON schema value
    pub fn from_value(name: &str, value: &Value) -> Self {
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
        
        // parse items for arrays
        let items = if schema_types.contains(&SchemaType::Array) {
            value.get("items").map(|item_schema| {
                Box::new(PropertyBlueprint::from_value("_item", item_schema))
            })
        } else {
            None
        };
        
        // parse enum values if present
        let enum_values = value.get("enum").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::to_vec(v).ok())
                .collect()
        });
        
        // parse pattern for string regex constraints
        let pattern = value.get("pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // parse string length constraints
        let min_length = value.get("minLength").and_then(|v| v.as_u64()).map(|n| n as usize);
        let max_length = value.get("maxLength").and_then(|v| v.as_u64()).map(|n| n as usize);
        
        // parse array length constraints
        let min_items = value.get("minItems").and_then(|v| v.as_u64()).map(|n| n as usize);
        let max_items = value.get("maxItems").and_then(|v| v.as_u64()).map(|n| n as usize);
        
        PropertyBlueprint {
            name: name.to_string(),
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
        }
    }
    
    /// check if this property has multiple allowed types (union type)
    pub fn is_union_type(&self) -> bool {
        self.schema_types.len() > 1
    }
    
    /// get the primary (first) schema type
    pub fn primary_type(&self) -> &SchemaType {
        self.schema_types.first().unwrap_or(&SchemaType::Any)
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
    /// parse a JSON schema from a serde_json::Value
    pub fn from_value(schema: &Value) -> Result<Self, ConstraintError> {
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
    
    /// get all allowed keys that start with the given prefix
    pub fn keys_with_prefix(&self, prefix: &str) -> Vec<&str> {
        self.allowed_keys
            .iter()
            .filter(|key| key.starts_with(prefix))
            .map(|s| s.as_str())
            .collect()
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
}
