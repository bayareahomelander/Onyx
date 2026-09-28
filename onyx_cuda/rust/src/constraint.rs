//! Core contract shared by grammar constraint engines.

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ConstraintError {
    InvalidTokenId { token_id: usize, vocab_size: usize },
    InvalidState(String),
    CompilationError(String),
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTokenId {
                token_id,
                vocab_size,
            } => write!(
                f,
                "Token ID {token_id} out of range (vocab size: {vocab_size})"
            ),
            Self::InvalidState(message) => write!(f, "Invalid state: {message}"),
            Self::CompilationError(message) => {
                write!(f, "Compilation error: {message}")
            }
        }
    }
}

impl Error for ConstraintError {}

/// Identifies engine states with the same valid-token scan. Only tokens that
/// `scan` reports as output-dependent may differ between states with equal keys.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ScanKey {
    /// A state that permits no token, such as a finished document.
    Empty,
    Regex(u32),
    Json(crate::json_engine::JsonScanKey),
}

pub trait ConstraintEngine: Send + Sync {
    fn get_valid_tokens(&self) -> Vec<usize>;
    fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError>;
    fn is_finished(&self) -> bool;
    fn is_dead(&self) -> bool;
    fn clone_box(&self) -> Box<dyn ConstraintEngine>;

    /// Cache key for this state's scan; None disables caching.
    fn scan_key(&self) -> Option<ScanKey> {
        None
    }

    /// Valid tokens, and the scanned tokens whose validity also depends on
    /// state outside the scan key. Cache hits recheck the latter.
    fn scan(&self) -> (Vec<usize>, Vec<usize>) {
        (self.get_valid_tokens(), Vec::new())
    }

    fn is_valid_token(&self, token_id: usize) -> bool {
        self.get_valid_tokens().contains(&token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_descriptive() {
        assert_eq!(
            ConstraintError::InvalidTokenId {
                token_id: 3,
                vocab_size: 3,
            }
            .to_string(),
            "Token ID 3 out of range (vocab size: 3)"
        );
        assert_eq!(
            ConstraintError::InvalidState("missing state".into()).to_string(),
            "Invalid state: missing state"
        );
        assert_eq!(
            ConstraintError::CompilationError("bad pattern".into()).to_string(),
            "Compilation error: bad pattern"
        );
    }
}
