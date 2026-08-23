"""basic tests for Onyx."""


def test_import():
    """test that onyx can be imported."""
    import onyx
    assert onyx.__version__ == "0.2.0"


def test_hello():
    """test the hello function."""
    import onyx
    result = onyx.hello()
    assert "Onyx" in result


def test_version():
    """test the version function."""
    import onyx
    result = onyx.version()
    assert result == onyx.__version__


def test_validate_json():
    """test JSON grammar validation."""
    import onyx
    assert onyx.validate_grammar('{"key": "value"}', "json") == True
    assert onyx.validate_grammar('[1, 2, 3]', "json") == True
    assert onyx.validate_grammar('{"key":', "json") == False
    assert onyx.validate_grammar('not json', "json") == False


def test_validate_sql():
    """test SQL grammar validation."""
    import onyx
    assert onyx.validate_grammar('SELECT * FROM users', "sql") == True
    assert onyx.validate_grammar('INSERT INTO users VALUES (1)', "sql") == True
    assert onyx.validate_grammar('  select * from users', "sql") == True
    assert onyx.validate_grammar('DROP TABLE users', "sql") == False
    assert onyx.validate_grammar('not sql', "sql") == False


def test_validate_unknown_grammar_type():
    """test unsupported grammar type validation."""
    import onyx
    assert onyx.validate_grammar('anything', "unknown") == False


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
