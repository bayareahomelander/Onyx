//! Native extension entry point for Onyx CUDA.

use pyo3::prelude::*;

mod constraint;
mod grammar;
mod json_engine;
mod regex_engine;
mod schema;
use grammar::GrammarConstraint;

#[pyfunction]
fn validate_json_schema(schema: &str) -> PyResult<()> {
    let value = serde_json::from_str(schema)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    schema::validate_schema(&value, "$")
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_json_output(schema: &str, output: &str) -> PyResult<String> {
    validate_json_schema(schema)?;
    let schema = serde_json::from_str(schema).expect("validated schema JSON");
    let value = serde_json::from_str(output).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "JSON generation did not produce a complete valid document: {error}"
        ))
    })?;
    if !schema::value_matches_schema(&value, &schema) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Generated JSON does not satisfy the requested schema",
        ));
    }
    // Preserve exact numeric values when the API requests compact JSON.
    Ok(value.to_string())
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GrammarConstraint>()?;
    module.add_function(wrap_pyfunction!(validate_json_schema, module)?)?;
    module.add_function(wrap_pyfunction!(validate_json_output, module)?)?;
    Ok(())
}
