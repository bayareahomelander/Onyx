//! Native extension entry point for Onyx CUDA.

use pyo3::prelude::*;

fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn version() -> &'static str {
    crate_version()
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_crate_version() {
        assert_eq!(crate_version(), "0.1.0");
    }
}
