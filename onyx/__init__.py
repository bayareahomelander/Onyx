"""
Onyx - High-Performance Inference Engine for Apple Silicon

A hybrid Python/Rust inference engine designed to solve the latency vs. reliability
trade-off in agentic AI by enabling local LLMs to generate structured outputs
(like JSON or SQL) at speeds exceeding human reading capabilities.
"""

__version__ = "0.1.0"

# Import Rust backend
try:
    from onyx._rust import (
        hello,
        version,
        validate_grammar,
        validate_regex_oneshot,
        RegexValidator,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    
    def hello():
        return "Onyx Python fallback (Rust extension not built)"
    
    def version():
        return __version__
    
    def validate_grammar(input: str, grammar_type: str) -> bool:
        return True
    
    def validate_regex_oneshot(text: str, pattern: str) -> bool:
        import re
        return bool(re.match(pattern, text))
    
    class RegexValidator:
        """Fallback Python implementation of RegexValidator."""
        def __init__(self, pattern: str):
            import re
            self._pattern = re.compile(pattern)
            self._pattern_str = pattern
        
        def validate(self, text: str) -> bool:
            return bool(self._pattern.match(text))
        
        def pattern(self) -> str:
            return self._pattern_str
        
        def __repr__(self) -> str:
            return f"RegexValidator(pattern='{self._pattern_str}')"

# Public API
__all__ = [
    "hello",
    "version", 
    "validate_grammar",
    "validate_regex_oneshot",
    "RegexValidator",
    "RUST_AVAILABLE",
    "__version__",
]
