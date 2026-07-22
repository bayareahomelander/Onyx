//! Independent Windows native grammar runtime for Onyx CUDA.

// PyO3 0.22 generates same-type PyErr conversions around fallible binding methods.
#![allow(clippy::useless_conversion)]

mod json_parser;
mod json_runtime;
mod json_schema;
mod regex_runtime;

use json_runtime::{JsonConstraintCore, JsonRuntimeError, JsonStateHandle};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use regex_runtime::{RegexConstraintCore, RegexRuntimeError, RegexStateHandle};
use std::sync::{Mutex, MutexGuard};

const GRAMMAR_ABI_VERSION: u32 = 3;

create_exception!(_grammar_native, NativeRegexCompilationError, PyException);
create_exception!(_grammar_native, NativeRegexStateError, PyException);
create_exception!(_grammar_native, NativeJsonCompilationError, PyException);
create_exception!(_grammar_native, NativeJsonStateError, PyException);

#[pyclass(
    name = "_NativeRegexState",
    module = "onyx_cuda._grammar_native",
    frozen
)]
#[derive(Clone, Copy)]
struct NativeRegexState {
    handle: RegexStateHandle,
}

#[pyclass(name = "_NativeRegexConstraint", module = "onyx_cuda._grammar_native")]
struct NativeRegexConstraint {
    core: Mutex<RegexConstraintCore>,
}

impl NativeRegexConstraint {
    fn lock(&self) -> PyResult<MutexGuard<'_, RegexConstraintCore>> {
        self.core.lock().map_err(|_| {
            NativeRegexStateError::new_err("native regex constraint lifecycle lock is poisoned")
        })
    }
}

#[pymethods]
impl NativeRegexConstraint {
    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.lock()?.vocab_size())
    }

    fn init_state(&self) -> PyResult<NativeRegexState> {
        let handle = self
            .lock()?
            .init_state()
            .map_err(regex_runtime_error_to_py)?;
        Ok(NativeRegexState { handle })
    }

    fn advance_state(
        &self,
        state: PyRef<'_, NativeRegexState>,
        token_id: usize,
    ) -> PyResult<NativeRegexState> {
        let handle = self
            .lock()?
            .advance_state(state.handle, token_id)
            .map_err(regex_runtime_error_to_py)?;
        Ok(NativeRegexState { handle })
    }

    fn get_valid_token_ids(&self, state: PyRef<'_, NativeRegexState>) -> PyResult<Vec<usize>> {
        self.lock()?
            .get_valid_token_ids(state.handle)
            .map_err(regex_runtime_error_to_py)
    }

    fn is_match_state(&self, state: PyRef<'_, NativeRegexState>) -> PyResult<bool> {
        self.lock()?
            .is_match_state(state.handle)
            .map_err(regex_runtime_error_to_py)
    }

    fn is_dead_state(&self, state: PyRef<'_, NativeRegexState>) -> PyResult<bool> {
        self.lock()?
            .is_dead_state(state.handle)
            .map_err(regex_runtime_error_to_py)
    }

    fn release_state(&self, state: PyRef<'_, NativeRegexState>) -> PyResult<()> {
        self.lock()?
            .release_state(state.handle)
            .map_err(regex_runtime_error_to_py)
    }

    fn release_states(&self, py: Python<'_>, states: Vec<Py<NativeRegexState>>) -> PyResult<()> {
        let handles = states
            .iter()
            .map(|state| state.borrow(py).handle)
            .collect::<Vec<_>>();
        self.lock()?
            .release_states(&handles)
            .map_err(regex_runtime_error_to_py)
    }

    fn reset(&self) -> PyResult<()> {
        self.lock()?.reset().map_err(regex_runtime_error_to_py)
    }
}

#[pyfunction]
fn compile_regex(vocabulary: Vec<Vec<u8>>, pattern: &str) -> PyResult<NativeRegexConstraint> {
    let core =
        RegexConstraintCore::compile(vocabulary, pattern).map_err(regex_runtime_error_to_py)?;
    Ok(NativeRegexConstraint {
        core: Mutex::new(core),
    })
}

#[pyclass(
    name = "_NativeJsonState",
    module = "onyx_cuda._grammar_native",
    frozen
)]
#[derive(Clone, Copy)]
struct NativeJsonState {
    handle: JsonStateHandle,
}

#[pyclass(name = "_NativeJsonConstraint", module = "onyx_cuda._grammar_native")]
struct NativeJsonConstraint {
    core: Mutex<JsonConstraintCore>,
}

impl NativeJsonConstraint {
    fn lock(&self) -> PyResult<MutexGuard<'_, JsonConstraintCore>> {
        self.core.lock().map_err(|_| {
            NativeJsonStateError::new_err("native JSON constraint lifecycle lock is poisoned")
        })
    }
}

#[pymethods]
impl NativeJsonConstraint {
    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.lock()?.vocab_size())
    }

    fn init_state(&self) -> PyResult<NativeJsonState> {
        let handle = self
            .lock()?
            .init_state()
            .map_err(json_runtime_error_to_py)?;
        Ok(NativeJsonState { handle })
    }

    fn advance_state(
        &self,
        state: PyRef<'_, NativeJsonState>,
        token_id: usize,
    ) -> PyResult<NativeJsonState> {
        let handle = self
            .lock()?
            .advance_state(state.handle, token_id)
            .map_err(json_runtime_error_to_py)?;
        Ok(NativeJsonState { handle })
    }

    fn get_valid_token_ids(&self, state: PyRef<'_, NativeJsonState>) -> PyResult<Vec<usize>> {
        self.lock()?
            .get_valid_token_ids(state.handle)
            .map_err(json_runtime_error_to_py)
    }

    fn is_match_state(&self, state: PyRef<'_, NativeJsonState>) -> PyResult<bool> {
        self.lock()?
            .is_match_state(state.handle)
            .map_err(json_runtime_error_to_py)
    }

    fn is_dead_state(&self, state: PyRef<'_, NativeJsonState>) -> PyResult<bool> {
        self.lock()?
            .is_dead_state(state.handle)
            .map_err(json_runtime_error_to_py)
    }

    fn release_state(&self, state: PyRef<'_, NativeJsonState>) -> PyResult<()> {
        self.lock()?
            .release_state(state.handle)
            .map_err(json_runtime_error_to_py)
    }

    fn release_states(&self, py: Python<'_>, states: Vec<Py<NativeJsonState>>) -> PyResult<()> {
        let handles = states
            .iter()
            .map(|state| state.borrow(py).handle)
            .collect::<Vec<_>>();
        self.lock()?
            .release_states(&handles)
            .map_err(json_runtime_error_to_py)
    }

    fn reset(&self) -> PyResult<()> {
        self.lock()?.reset().map_err(json_runtime_error_to_py)
    }
}

#[pyfunction]
fn compile_json_schema(vocabulary: Vec<Vec<u8>>, schema: &str) -> PyResult<NativeJsonConstraint> {
    let core = JsonConstraintCore::compile(vocabulary, schema).map_err(json_runtime_error_to_py)?;
    Ok(NativeJsonConstraint {
        core: Mutex::new(core),
    })
}

#[pyfunction]
fn runtime_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn grammar_abi_version() -> u32 {
    GRAMMAR_ABI_VERSION
}

fn regex_runtime_error_to_py(error: RegexRuntimeError) -> PyErr {
    match error {
        RegexRuntimeError::Compilation(message) => NativeRegexCompilationError::new_err(message),
        RegexRuntimeError::InvalidInput(message) => PyValueError::new_err(message),
        RegexRuntimeError::InvalidTokenId { .. } => PyValueError::new_err(error.to_string()),
        RegexRuntimeError::State(message) => NativeRegexStateError::new_err(message),
    }
}

fn json_runtime_error_to_py(error: JsonRuntimeError) -> PyErr {
    match error {
        JsonRuntimeError::Compilation(message) => NativeJsonCompilationError::new_err(message),
        JsonRuntimeError::InvalidInput(message) => PyValueError::new_err(message),
        JsonRuntimeError::InvalidTokenId { .. } => PyValueError::new_err(error.to_string()),
        JsonRuntimeError::InvalidTransition { .. } => {
            NativeJsonStateError::new_err(error.to_string())
        }
        JsonRuntimeError::State(message) => NativeJsonStateError::new_err(message),
    }
}

#[pymodule]
fn _grammar_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(runtime_version, module)?)?;
    module.add_function(wrap_pyfunction!(grammar_abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(compile_regex, module)?)?;
    module.add_function(wrap_pyfunction!(compile_json_schema, module)?)?;
    module.add_class::<NativeRegexConstraint>()?;
    module.add_class::<NativeRegexState>()?;
    module.add_class::<NativeJsonConstraint>()?;
    module.add_class::<NativeJsonState>()?;
    module.add(
        "NativeRegexCompilationError",
        module.py().get_type_bound::<NativeRegexCompilationError>(),
    )?;
    module.add(
        "NativeRegexStateError",
        module.py().get_type_bound::<NativeRegexStateError>(),
    )?;
    module.add(
        "NativeJsonCompilationError",
        module.py().get_type_bound::<NativeJsonCompilationError>(),
    )?;
    module.add(
        "NativeJsonStateError",
        module.py().get_type_bound::<NativeJsonStateError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_identity_is_stable() {
        assert_eq!(runtime_version(), "0.1.0");
        assert_eq!(grammar_abi_version(), 3);
    }
}
