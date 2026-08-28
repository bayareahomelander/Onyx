//! Native extension entry point for Onyx CUDA.

use pyo3::prelude::*;

mod constraint;
mod grammar;
mod json_engine;
mod regex_engine;
mod schema;

#[pymodule]
fn _rust(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
