"""basic tests for Onyx."""

import pytest


def test_import():
    """test that onyx can be imported."""
    import onyx
    assert onyx.__version__ == "0.1.0"


def test_hello():
    """test the hello function."""
    import onyx
    result = onyx.hello()
    assert "Onyx" in result


def test_version():
    """test the version function."""
    import onyx
    result = onyx.version()
    assert result == "0.1.0"


def test_validate_json():
    """test JSON grammar validation."""
    import onyx
    assert onyx.validate_grammar('{"key": "value"}', "json") == True
    assert onyx.validate_grammar('[1, 2, 3]', "json") == True


def test_validate_sql():
    """test SQL grammar validation."""
    import onyx
    assert onyx.validate_grammar('SELECT * FROM users', "sql") == True
    assert onyx.validate_grammar('INSERT INTO users VALUES (1)', "sql") == True


def test_engine_creation():
    """test engine can be created."""
    from onyx.engine import OnyxEngine
    engine = OnyxEngine()
    assert engine is not None


def test_device_info():
    """test device info retrieval."""
    from onyx.engine import get_device_info
    info = get_device_info()
    assert "device" in info
    assert "mlx_available" in info
