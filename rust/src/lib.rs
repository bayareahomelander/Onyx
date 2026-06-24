//! onyx rust backend
//!
//! high-performance grammar constraint engine for structured LLM outputs
//! this module provides the core grammar processing logic that integrates
//! with the python MLX frontend

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

mod constraint;
mod grammar;
mod json_engine;
mod regex_engine;
mod schema;
use grammar::GrammarConstraint;

/// a compiled regex validator exposed to Python
///
/// this struct pre-compiles a regex pattern for efficient repeated validation
/// it demonstrates the FFI bridge between Python and Rust and will be used
/// to benchmark the function call overhead

#[pyclass]
pub struct RegexValidator {
    pattern: Regex,
    pattern_str: String,
}

#[pymethods]
impl RegexValidator {
    /// create a new RegexValidator with the given pattern
    ///
    /// returns a new RegexValidator instance
    /// raises ValueError if the pattern is invalid

    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        let compiled = Regex::new(pattern)
            .map_err(|e| PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;

        Ok(RegexValidator {
            pattern: compiled,
            pattern_str: pattern.to_string(),
        })
    }

    /// validate that the input text matches the compiled pattern
    ///
    /// returns true if the text matches the pattern, false otherwise

    fn validate(&self, text: &str) -> bool {
        self.pattern.is_match(text)
    }

    fn pattern(&self) -> &str {
        &self.pattern_str
    }

    fn __repr__(&self) -> String {
        format!("RegexValidator(pattern='{}')", self.pattern_str)
    }
}

/// returns a greeting message to verify the Rust extension is working
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from Onyx Rust backend! Grammar engine ready.".to_string())
}

/// returns version information about the rust backend
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// lightweight grammar validation helper
#[pyfunction]
fn validate_grammar(input: &str, grammar_type: &str) -> PyResult<bool> {
    match grammar_type.to_ascii_lowercase().as_str() {
        "json" => Ok(serde_json::from_str::<serde_json::Value>(input).is_ok()),
        "sql" => {
            let upper = input.trim_start().to_ascii_uppercase();
            Ok(upper.starts_with("SELECT")
                || upper.starts_with("INSERT")
                || upper.starts_with("UPDATE")
                || upper.starts_with("DELETE"))
        }
        _ => Ok(false),
    }
}

/// validate a string against a regex pattern (one-shot, no pre-compilation)
/// this is useful for comparing against the RegexValidator class
#[pyfunction]
fn validate_regex_oneshot(text: &str, pattern: &str) -> PyResult<bool> {
    let re = Regex::new(pattern)
        .map_err(|e| PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;
    Ok(re.is_match(text))
}

/// the main Python module for the onyx rust extension

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // add functions
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(validate_grammar, m)?)?;
    m.add_function(wrap_pyfunction!(validate_regex_oneshot, m)?)?;

    // Add classes
    m.add_class::<RegexValidator>()?;
    m.add_class::<GrammarConstraint>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_grammar_json() {
        assert!(validate_grammar(r#"{"key":"value"}"#, "json").unwrap());
        assert!(validate_grammar("[1, 2, 3]", "JSON").unwrap());
        assert!(!validate_grammar(r#"{"key":"#, "json").unwrap());
        assert!(!validate_grammar("not json", "json").unwrap());
    }

    #[test]
    fn test_validate_grammar_sql() {
        assert!(validate_grammar("SELECT * FROM users", "sql").unwrap());
        assert!(validate_grammar("  select * from users", "SQL").unwrap());
        assert!(!validate_grammar("DROP TABLE users", "sql").unwrap());
        assert!(!validate_grammar("not sql", "sql").unwrap());
    }

    #[test]
    fn test_validate_grammar_unknown_type() {
        assert!(!validate_grammar("anything", "unknown").unwrap());
    }
}
