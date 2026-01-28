//! onyx rust backend
//! 
//! high-performance grammar constraint engine for structured LLM outputs
//! this module provides the core grammar processing logic that integrates
//! with the python MLX frontend

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use regex::Regex;

mod constraint;
mod regex_engine;
mod json_engine;
mod schema;
mod grammar;
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

/// placeholder for grammar constraint validation
/// this will be expanded to handle json schema, sql grammar, etc
#[pyfunction]
fn validate_grammar(input: &str, grammar_type: &str) -> PyResult<bool> {
    match grammar_type {
        "json" => Ok(input.starts_with('{') || input.starts_with('[')),
        "sql" => Ok(input.to_uppercase().starts_with("SELECT") 
                   || input.to_uppercase().starts_with("INSERT")
                   || input.to_uppercase().starts_with("UPDATE")
                   || input.to_uppercase().starts_with("DELETE")),
        _ => Ok(true),
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
fn onyx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
