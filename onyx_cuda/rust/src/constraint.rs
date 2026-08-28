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

pub trait ConstraintEngine: Send + Sync {
    fn reset(&mut self);
    fn get_valid_tokens(&self) -> Vec<usize>;
    fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError>;
    fn is_finished(&self) -> bool;
    fn is_dead(&self) -> bool;
    fn current_state_id(&self) -> u32;
    fn clone_box(&self) -> Box<dyn ConstraintEngine>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct TestEngine {
        state: u32,
    }

    impl ConstraintEngine for TestEngine {
        fn reset(&mut self) {
            self.state = 0;
        }

        fn get_valid_tokens(&self) -> Vec<usize> {
            if self.is_finished() || self.is_dead() {
                Vec::new()
            } else {
                vec![0, 1, 2]
            }
        }

        fn advance(&mut self, token_id: usize) -> Result<(), ConstraintError> {
            if token_id >= 3 {
                return Err(ConstraintError::InvalidTokenId {
                    token_id,
                    vocab_size: 3,
                });
            }
            self.state = token_id as u32 + 1;
            Ok(())
        }

        fn is_finished(&self) -> bool {
            self.state == 2
        }

        fn is_dead(&self) -> bool {
            self.state == 3
        }

        fn current_state_id(&self) -> u32 {
            self.state
        }

        fn clone_box(&self) -> Box<dyn ConstraintEngine> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn state_and_clone_contract() {
        let mut engine = TestEngine::default();
        assert_eq!(engine.get_valid_tokens(), vec![0, 1, 2]);

        engine.advance(0).unwrap();
        let mut branch = engine.clone_box();
        branch.advance(1).unwrap();
        assert!(branch.is_finished());
        assert_eq!(engine.current_state_id(), 1);

        engine.advance(2).unwrap();
        assert!(engine.is_dead());
        engine.reset();
        assert_eq!(engine.current_state_id(), 0);
        assert!(branch.is_finished());

        assert!(matches!(
            engine.advance(3),
            Err(ConstraintError::InvalidTokenId {
                token_id: 3,
                vocab_size: 3
            })
        ));
    }

    #[test]
    fn errors_are_descriptive() {
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
