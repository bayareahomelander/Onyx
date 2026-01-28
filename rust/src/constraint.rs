//! constraint engine trait
//!
//! this module defines the core trait that all constraint engines must implement.
//! this abstraction allows us to support multiple constraint types (regex, JSON schema, etc.)
//! behind a unified interface for the sampling loop.

use std::error::Error;
use std::fmt;

/// error type for constraint engine operations
#[derive(Debug)]
pub enum ConstraintError {
    /// token ID is out of range for the vocabulary
    InvalidTokenId { token_id: usize, vocab_size: usize },
    /// constraint engine is in an invalid state
    InvalidState(String),
    /// pattern compilation failed
    CompilationError(String),
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintError::InvalidTokenId { token_id, vocab_size } => {
                write!(f, "Token ID {} out of range (vocab size: {})", token_id, vocab_size)
            }
            ConstraintError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            ConstraintError::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
        }
    }
}

impl Error for ConstraintError {}

/// the core trait that all constraint engines must implement
///
/// this trait defines the interface used by the sampling loop to:
/// - get valid tokens at each generation step
/// - advance state after token selection
/// - check completion and dead states
///
/// implementations include:
/// - `RegexEngine`: DFA-based regex pattern matching
/// - (future) `JsonSchemaEngine`: pushdown automaton for JSON schema validation
pub trait ConstraintEngine: Send + Sync {
    /// reset the engine to its initial state
    ///
    /// this should be called before starting a new generation sequence
    fn reset(&mut self);

    /// get all valid token IDs from the current state
    ///
    /// returns a vector of token IDs that are valid continuations from
    /// the current state. the sampling loop uses this to mask invalid tokens.
    fn get_valid_tokens(&self) -> Vec<usize>;

    /// advance the state by consuming a token
    ///
    /// this updates the internal state machine after a token has been selected.
    /// returns an error if the token ID is invalid or the state transition fails.
    fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError>;

    /// check if the constraint is satisfied (pattern complete)
    ///
    /// returns true if the current state represents a valid completion
    /// of the constraint (e.g., regex pattern fully matched)
    fn is_finished(&self) -> bool;

    /// check if the constraint is in a dead state
    ///
    /// returns true if no valid continuation exists from the current state
    /// (i.e., the pattern can never be completed from here)
    fn is_dead(&self) -> bool;

    /// get the current state as an opaque u32 value
    ///
    /// this is useful for debugging and for the Python bindings
    fn current_state_id(&self) -> u32;
}
