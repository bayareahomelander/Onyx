from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_windows_native_build_is_independent_and_namespaced():
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (PACKAGE_ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    rust_source = (PACKAGE_ROOT / "rust" / "src" / "lib.rs").read_text(encoding="utf-8")
    regex_source = (PACKAGE_ROOT / "rust" / "src" / "regex_runtime.rs").read_text(
        encoding="utf-8"
    )
    json_schema_source = (
        PACKAGE_ROOT / "rust" / "src" / "json_schema.rs"
    ).read_text(encoding="utf-8")
    json_parser_source = (
        PACKAGE_ROOT / "rust" / "src" / "json_parser.rs"
    ).read_text(encoding="utf-8")
    json_runtime_source = (
        PACKAGE_ROOT / "rust" / "src" / "json_runtime.rs"
    ).read_text(encoding="utf-8")
    native_json_source = (
        PACKAGE_ROOT / "src" / "onyx_cuda" / "native_json.py"
    ).read_text(encoding="utf-8")

    assert 'build-backend = "maturin"' in pyproject
    assert 'module-name = "onyx_cuda._grammar_native"' in pyproject
    assert 'manifest-path = "rust/Cargo.toml"' in pyproject
    assert 'python-source = "src"' in pyproject
    assert '"src/**/__pycache__/**"' in pyproject
    assert '"src/**/*.pyc"' in pyproject
    assert '"cuda_support_roadmap.md"' in pyproject
    assert '"codebase_review_notes.md"' in pyproject
    assert '"d22_implementation_plan.md"' in pyproject
    assert '"d23_implementation_plan.md"' in pyproject
    assert '"local markdowns/**"' in pyproject
    assert '"local_markdowns/**"' in pyproject
    assert '"**/d*_implementation_plan.md"' in pyproject
    assert '"rust/.gitignore"' in pyproject
    assert '"src/**/*.pyo"' in pyproject
    assert '"src/**/*.pyd"' in pyproject
    assert "[tool.maturin.sbom]" in pyproject
    assert "rust = false" in pyproject

    assert 'name = "onyx_cuda_grammar_native"' in cargo
    assert 'name = "_grammar_native"' in cargo
    assert "path =" not in cargo
    assert "onyx_rust" not in cargo
    assert 'regex-automata = { version = "0.4"' in cargo
    assert "default-features = false" in cargo
    assert 'features = ["dfa", "dfa-build", "dfa-search", "syntax"]' in cargo
    assert "\nregex =" not in cargo
    assert 'serde_json = "1"' in cargo
    assert "gil-refs = []" not in cargo
    assert "unexpected_cfgs" in cargo
    assert 'cfg(feature, values("gil-refs"))' in cargo
    assert "fn _grammar_native" in rust_source
    assert "runtime_version" in rust_source
    assert "grammar_abi_version" in rust_source
    assert "GRAMMAR_ABI_VERSION: u32 = 3" in rust_source
    assert "mod json_schema;" in rust_source
    assert "mod json_parser;" in rust_source
    assert "mod json_runtime;" in rust_source
    assert "fn compile_json_schema" in rust_source
    assert "NativeJsonCompilationError" in rust_source
    assert "NativeJsonStateError" in rust_source
    assert "NativeJsonConstraint" in rust_source
    assert "NativeJsonState" in rust_source
    assert "dense::DFA" in regex_source
    assert "next_eoi_state" in regex_source
    assert "compile_json_schema" in json_schema_source
    assert "serde_json" in json_schema_source
    assert "JsonParserState" in json_parser_source
    assert "JsonConstraintCore" in json_runtime_source
    assert "InvalidTransition" in json_runtime_source
    for json_source in (json_schema_source, json_parser_source, json_runtime_source):
        assert "pyo3" not in json_source
        assert "onyx::" not in json_source
        assert "mlx" not in json_source.lower()
    assert "compile_native_json_schema" in native_json_source
    assert "_load_native_grammar_runtime" in native_json_source
    assert "import onyx" not in native_json_source
    assert "from onyx" not in native_json_source
    assert "import mlx" not in native_json_source


def test_windows_native_build_keeps_its_exact_dependency_lock():
    cargo_lock = (PACKAGE_ROOT / "rust" / "Cargo.lock").read_text(encoding="utf-8")
    crate_ignore = (PACKAGE_ROOT / "rust" / ".gitignore").read_text(encoding="utf-8")

    assert "version = 4" in cargo_lock
    assert 'name = "onyx_cuda_grammar_native"' in cargo_lock
    assert 'name = "serde_json"' in cargo_lock
    assert '"serde_json"' in cargo_lock
    assert "!Cargo.lock" in crate_ignore.splitlines()


def test_native_distribution_qualification_is_committed_and_uses_frozen_parity():
    regex_wheel_smoke = (
        PACKAGE_ROOT / "tests" / "installed_wheel_regex_smoke.py"
    ).read_text(
        encoding="utf-8"
    )
    json_wheel_smoke = (
        PACKAGE_ROOT / "tests" / "installed_wheel_json_smoke.py"
    ).read_text(
        encoding="utf-8"
    )
    distribution_inspection = (
        PACKAGE_ROOT / "tests" / "inspect_native_distribution.py"
    ).read_text(encoding="utf-8")
    vocabulary_qualification = (
        PACKAGE_ROOT / "tests" / "qualify_qwen_grammar_vocabulary.py"
    ).read_text(encoding="utf-8")
    mask_qualification = (
        PACKAGE_ROOT / "tests" / "qualify_cuda_grammar_mask.py"
    ).read_text(encoding="utf-8")
    constrained_smoke = (
        PACKAGE_ROOT / "tests" / "installed_wheel_constrained_smoke.py"
    ).read_text(encoding="utf-8")
    d24_qualification = (
        PACKAGE_ROOT / "tests" / "qualify_d24_constrained_generation.py"
    ).read_text(encoding="utf-8")
    d25_qualification = (
        PACKAGE_ROOT / "tests" / "qualify_d25_constrained_streaming.py"
    ).read_text(encoding="utf-8")
    d26_qualification = (
        PACKAGE_ROOT / "tests" / "qualify_d26_grammar_timing.py"
    ).read_text(encoding="utf-8")

    assert "REGEX_PARITY_CASES" in regex_wheel_smoke
    assert "INVALID_REGEX_PATTERNS" in regex_wheel_smoke
    assert "LIFECYCLE_ITERATIONS = 1_000" in regex_wheel_smoke
    assert "_require_json_surface_present" in regex_wheel_smoke
    assert "JSON_PARITY_CASES" in json_wheel_smoke
    assert "JSON_SCHEMA_OBSERVATIONS" in json_wheel_smoke
    assert "compile_native_json_schema" in json_wheel_smoke
    assert "LIFECYCLE_ITERATIONS = 1_000" in json_wheel_smoke
    assert "__pycache__" in distribution_inspection
    assert '".pyc"' in distribution_inspection
    assert "cuda_support_roadmap.md" in distribution_inspection
    assert "codebase_review_notes.md" in distribution_inspection
    assert "d22_implementation_plan.md" in distribution_inspection
    assert "d23_implementation_plan.md" in distribution_inspection
    assert "local markdowns" in distribution_inspection
    assert "local_markdowns" in distribution_inspection
    assert "_implementation_plan.md" in distribution_inspection
    assert "EXPECTED_BASE_VOCAB_SIZE = 151_643" in vocabulary_qualification
    assert "EXPECTED_VOCAB_SIZE = 151_665" in vocabulary_qualification
    assert "EXPECTED_FINGERPRINT" in vocabulary_qualification
    assert "local_files_only=True" in vocabulary_qualification
    assert "build_qwen_grammar_vocabulary" in vocabulary_qualification
    assert "onyx_cuda._grammar_native" in vocabulary_qualification
    assert "VOCAB_SIZE = 151_665" in mask_qualification
    assert 'EXPECTED_TRANSPORT = "sparse_valid_indices"' in mask_qualification
    assert "create_cuda_grammar_logit_mask" in mask_qualification
    assert "select_cuda_argmax" in mask_qualification
    assert "create_cuda_sampler" in mask_qualification
    assert "block_snapshots" in mask_qualification
    assert "onyx_cuda._grammar_native" in mask_qualification
    assert "generate_constrained_target" in constrained_smoke
    assert "compile_native_regex" in constrained_smoke
    assert "compile_native_json_schema" in constrained_smoke
    assert "stream_constrained" in constrained_smoke
    assert "TextGenerationDelta" in constrained_smoke
    assert "partial.close()" in constrained_smoke
    assert "LIFECYCLE_ITERATIONS = 100" in constrained_smoke
    assert "GrammarTimingMetrics" in constrained_smoke
    assert "create_deterministic_grammar_timing_session" in constrained_smoke
    assert "load_production_target_engine" in d24_qualification
    assert "RegexGrammar" in d24_qualification
    assert "JsonSchemaGrammar" in d24_qualification
    assert "VRAM_LIMIT_BYTES = 6_141" in d24_qualification
    assert "POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024" in d24_qualification
    assert "local_files_only=True" in d24_qualification
    assert "load_production_target_engine" in d25_qualification
    assert "stream_constrained" in d25_qualification
    assert "RegexGrammar" in d25_qualification
    assert "JsonSchemaGrammar" in d25_qualification
    assert "SAMPLED_POLICY" in d25_qualification
    assert "_cancel_and_reuse" in d25_qualification
    assert "engine.stream(" in d25_qualification
    assert "VRAM_LIMIT_BYTES = 6_141" in d25_qualification
    assert "POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024" in d25_qualification
    assert "local_files_only=True" in d25_qualification
    assert "load_production_target_engine" in d26_qualification
    assert "GrammarTimingMetrics" in d26_qualification
    assert "valid_index_transfer_time" in d26_qualification
    assert "mask_application_time" in d26_qualification
    assert "_measure_standalone_mask_overhead" in d26_qualification
    assert "for lifecycle in (1, 2)" in d26_qualification
    assert "VRAM_LIMIT_BYTES = 6_141" in d26_qualification
    assert "POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024" in d26_qualification
    assert "local_files_only=True" in d26_qualification


def test_native_build_does_not_modify_or_depend_on_mac_paths():
    cargo = (PACKAGE_ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    forbidden = ("../rust", "../onyx", "onyx._rust", "mlx")
    assert all(value not in cargo for value in forbidden)
    assert all(value not in pyproject for value in forbidden)
