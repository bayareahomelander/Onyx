//! Native extension entry point for Onyx CUDA.

use pyo3::prelude::*;

mod constraint;
mod grammar;
mod json_engine;
mod regex_engine;
mod schema;
use grammar::GrammarConstraint;

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GrammarConstraint>()?;
    Ok(())
}
