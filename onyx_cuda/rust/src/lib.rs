//! Native extension entry point for Onyx CUDA.

use pyo3::prelude::*;

mod constraint;

#[pymodule]
fn _rust(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
