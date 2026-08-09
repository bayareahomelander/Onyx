import builtins
import importlib
import sys
from pathlib import Path

from onyx_cuda._torch_install import (
    PYTORCH_CUDA_INSTALL_COMMAND,
    PYTORCH_CUDA_REQUIREMENT,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VALIDATED_CUDA_REQUIREMENTS = (
    "accelerate==1.14.0",
    "bitsandbytes==0.49.2",
    "huggingface-hub==0.36.2",
    "psutil==7.2.2",
    "tokenizers==0.22.2",
    PYTORCH_CUDA_REQUIREMENT,
    "transformers==4.57.6",
)


def test_package_identifies_the_windows_cuda_variant():
    import onyx_cuda

    assert onyx_cuda.ENGINE_NAME == "Onyx for Windows"
    assert onyx_cuda.SUPPORTED_PLATFORM == "windows"
    assert onyx_cuda.ACCELERATOR_BACKEND == "cuda"
    assert onyx_cuda.CacheCheckpoint.__module__ == "onyx_cuda.cache"
    assert onyx_cuda.CheckpointableAutoregressiveBackend.__module__ == "onyx_cuda.cache"
    assert issubclass(onyx_cuda.CacheCheckpointStateError, onyx_cuda.BackendStateError)
    for checkpoint_symbol in (
        "CacheCheckpoint",
        "CacheCheckpointStateError",
        "CheckpointableAutoregressiveBackend",
    ):
        assert checkpoint_symbol in onyx_cuda.__all__
    assert onyx_cuda.BatchedTargetVerificationResult.__module__ == "onyx_cuda.verification"
    assert onyx_cuda.BatchedTargetVerificationBackend.__module__ == "onyx_cuda.verification"
    for verification_symbol in (
        "BatchedTargetVerificationResult",
        "BatchedTargetVerificationBackend",
    ):
        assert verification_symbol in onyx_cuda.__all__
    for draft_symbol in (
        "DraftProposalCleanupError",
        "DraftProposalError",
        "DraftProposalInvariantError",
        "DraftProposalResult",
        "generate_draft_proposal",
    ):
        exported = getattr(onyx_cuda, draft_symbol)
        assert exported.__module__ == "onyx_cuda.draft"
        assert draft_symbol in onyx_cuda.__all__
    for grammar_draft_symbol in (
        "GrammarMaskedDraftProposalCleanupError",
        "GrammarMaskedDraftProposalError",
        "GrammarMaskedDraftProposalInvariantError",
        "GrammarMaskedDraftProposalResult",
        "generate_grammar_masked_draft_proposal",
    ):
        exported = getattr(onyx_cuda, grammar_draft_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_draft"
        assert grammar_draft_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedDraftProposalError,
        onyx_cuda.DraftProposalError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedDraftProposalInvariantError,
        onyx_cuda.GrammarMaskedDraftProposalError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedDraftProposalCleanupError,
        onyx_cuda.GrammarMaskedDraftProposalError,
    )
    for acceptance_symbol in (
        "MatchReplaceAcceptanceError",
        "MatchReplaceAcceptanceInvariantError",
        "MatchReplaceAcceptanceResult",
        "decide_match_replace_acceptance",
    ):
        exported = getattr(onyx_cuda, acceptance_symbol)
        assert exported.__module__ == "onyx_cuda.acceptance"
        assert acceptance_symbol in onyx_cuda.__all__
    for grammar_acceptance_symbol in (
        "GrammarMaskedTargetAcceptanceCleanupError",
        "GrammarMaskedTargetAcceptanceError",
        "GrammarMaskedTargetAcceptanceInvariantError",
        "GrammarMaskedTargetAcceptanceResult",
        "decide_grammar_masked_target_acceptance",
    ):
        exported = getattr(onyx_cuda, grammar_acceptance_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_acceptance"
        assert grammar_acceptance_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedTargetAcceptanceError,
        onyx_cuda.MatchReplaceAcceptanceError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedTargetAcceptanceInvariantError,
        onyx_cuda.GrammarMaskedTargetAcceptanceError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedTargetAcceptanceCleanupError,
        onyx_cuda.GrammarMaskedTargetAcceptanceError,
    )
    for grammar_continuation_symbol in (
        "GrammarMaskedPostAcceptanceContinuationCleanupError",
        "GrammarMaskedPostAcceptanceContinuationError",
        "GrammarMaskedPostAcceptanceContinuationInvariantError",
        "GrammarMaskedPostAcceptanceContinuationResult",
        "decide_grammar_masked_post_acceptance_continuation",
    ):
        exported = getattr(onyx_cuda, grammar_continuation_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_continuation"
        assert grammar_continuation_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedPostAcceptanceContinuationError,
        onyx_cuda.PostIterationContinuationError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedPostAcceptanceContinuationInvariantError,
        onyx_cuda.GrammarMaskedPostAcceptanceContinuationError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedPostAcceptanceContinuationCleanupError,
        onyx_cuda.GrammarMaskedPostAcceptanceContinuationError,
    )
    for grammar_iteration_symbol in (
        "GrammarMaskedSpeculativeIterationCleanupError",
        "GrammarMaskedSpeculativeIterationError",
        "GrammarMaskedSpeculativeIterationInvariantError",
        "GrammarMaskedSpeculativeIterationResult",
        "coordinate_grammar_masked_speculative_iteration",
    ):
        exported = getattr(onyx_cuda, grammar_iteration_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_speculative_iteration"
        assert grammar_iteration_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeIterationError,
        onyx_cuda.SpeculativeIterationError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeIterationInvariantError,
        onyx_cuda.GrammarMaskedSpeculativeIterationError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeIterationCleanupError,
        onyx_cuda.GrammarMaskedSpeculativeIterationError,
    )
    for grammar_outcome_symbol in (
        "GrammarMaskedSpeculativeOutcomeError",
        "GrammarMaskedSpeculativeOutcomeInvariantError",
        "GrammarMaskedSpeculativeOutcomeResult",
        "classify_grammar_masked_speculative_outcome",
    ):
        exported = getattr(onyx_cuda, grammar_outcome_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_speculative_outcome"
        assert grammar_outcome_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeOutcomeError,
        onyx_cuda.GrammarMaskedSpeculativeIterationError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeOutcomeInvariantError,
        onyx_cuda.GrammarMaskedSpeculativeOutcomeError,
    )
    for grammar_handoff_symbol in (
        "GrammarMaskedSpeculativeHandoffCleanupError",
        "GrammarMaskedSpeculativeHandoffError",
        "GrammarMaskedSpeculativeHandoffInvariantError",
        "GrammarMaskedSpeculativeHandoffResult",
        "coordinate_grammar_masked_speculative_handoff",
        "coordinate_multi_iteration_grammar_masked_speculative_handoff",
    ):
        exported = getattr(onyx_cuda, grammar_handoff_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_speculative_handoff"
        assert grammar_handoff_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeHandoffError,
        onyx_cuda.GrammarMaskedSpeculativeOutcomeError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeHandoffInvariantError,
        onyx_cuda.GrammarMaskedSpeculativeHandoffError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedSpeculativeHandoffCleanupError,
        onyx_cuda.GrammarMaskedSpeculativeHandoffError,
    )
    for continuation_symbol in (
        "PostIterationContinuationError",
        "PostIterationContinuationInvariantError",
        "PostIterationContinuationResult",
        "decide_post_iteration_continuation",
    ):
        exported = getattr(onyx_cuda, continuation_symbol)
        assert exported.__module__ == "onyx_cuda.continuation"
        assert continuation_symbol in onyx_cuda.__all__
    for iteration_symbol in (
        "ContinuationAwareSpeculativeIterationResult",
        "SpeculativeIterationCleanupError",
        "SpeculativeIterationError",
        "SpeculativeIterationInvariantError",
        "SpeculativeIterationResult",
        "coordinate_continuation_aware_speculative_iteration",
        "coordinate_speculative_iteration",
    ):
        exported = getattr(onyx_cuda, iteration_symbol)
        assert exported.__module__ == "onyx_cuda.speculative_iteration"
        assert iteration_symbol in onyx_cuda.__all__
    for handoff_symbol in (
        "SpeculativeHandoffCleanupError",
        "SpeculativeHandoffError",
        "SpeculativeHandoffInvariantError",
        "TwoIterationSpeculativeHandoffResult",
        "coordinate_two_iteration_speculative_handoff",
        "MultiIterationSpeculativeHandoffResult",
        "coordinate_multi_iteration_speculative_handoff",
    ):
        exported = getattr(onyx_cuda, handoff_symbol)
        assert exported.__module__ == "onyx_cuda.speculative_handoff"
        assert handoff_symbol in onyx_cuda.__all__
    for grammar_reconciliation_symbol in (
        "SpeculativeGrammarReconciliationCleanupError",
        "SpeculativeGrammarReconciliationError",
        "SpeculativeGrammarReconciliationInvariantError",
        "SpeculativeGrammarReconciliationResult",
        "reconcile_speculative_grammar_state",
    ):
        exported = getattr(onyx_cuda, grammar_reconciliation_symbol)
        assert exported.__module__ == "onyx_cuda.speculative_grammar"
        assert grammar_reconciliation_symbol in onyx_cuda.__all__
    for grammar_selection_symbol in (
        "GrammarMaskedSelectionError",
        "GrammarMaskedSelectionInvariantError",
        "GrammarMaskedSelectionResult",
        "select_grammar_masked_token",
    ):
        exported = getattr(onyx_cuda, grammar_selection_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_selection"
        assert grammar_selection_symbol in onyx_cuda.__all__
    assert issubclass(onyx_cuda.GrammarMaskedSelectionError, onyx_cuda.GrammarError)
    assert issubclass(
        onyx_cuda.GrammarMaskedSelectionInvariantError,
        onyx_cuda.GrammarMaskedSelectionError,
    )
    for grammar_transition_symbol in (
        "GrammarMaskedTransitionCleanupError",
        "GrammarMaskedTransitionError",
        "GrammarMaskedTransitionInvariantError",
        "GrammarMaskedTransitionResult",
        "select_and_advance_grammar_state",
    ):
        exported = getattr(onyx_cuda, grammar_transition_symbol)
        assert exported.__module__ == "onyx_cuda.grammar_transition"
        assert grammar_transition_symbol in onyx_cuda.__all__
    assert issubclass(
        onyx_cuda.GrammarMaskedTransitionError,
        onyx_cuda.GrammarError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedTransitionInvariantError,
        onyx_cuda.GrammarMaskedTransitionError,
    )
    assert issubclass(
        onyx_cuda.GrammarMaskedTransitionCleanupError,
        onyx_cuda.GrammarMaskedTransitionError,
    )
    assert callable(onyx_cuda.build_qwen_grammar_vocabulary)
    assert callable(onyx_cuda.create_cuda_grammar_logit_mask)
    assert callable(onyx_cuda.generate_constrained_target)
    assert callable(onyx_cuda.TargetTextEngine.stream_constrained)
    assert callable(onyx_cuda.ProductionTargetTextEngine.stream_constrained)
    assert onyx_cuda.RegexGrammar("a").pattern == "a"
    assert onyx_cuda.JsonSchemaGrammar("{}").schema == "{}"
    assert issubclass(onyx_cuda.ConstrainedGenerationError, RuntimeError)
    assert onyx_cuda.TorchCUDAGrammarLogitMask.__module__ == "onyx_cuda.torch_grammar_mask"
    for error_name in (
        "TorchGrammarMaskError",
        "TorchGrammarMaskImportError",
        "TorchGrammarMaskUnavailableError",
        "TorchGrammarMaskInvariantError",
        "TorchGrammarMaskExecutionError",
    ):
        assert issubclass(getattr(onyx_cuda, error_name), RuntimeError)


def test_cuda_extra_and_installation_guide_pin_the_validated_stack():
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    for requirement in VALIDATED_CUDA_REQUIREMENTS:
        assert f'"{requirement}"' in pyproject
    assert PYTORCH_CUDA_INSTALL_COMMAND in readme


def test_import_does_not_load_mac_or_gpu_runtimes(monkeypatch):
    forbidden_imports = (
        "onyx",
        "mlx",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
        "onnxruntime",
        "psutil",
    )
    requested_imports = []
    original_import = builtins.__import__

    def record_import(name, *args, **kwargs):
        requested_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)

    for module_name in tuple(sys.modules):
        if module_name == "onyx_cuda" or module_name.startswith("onyx_cuda."):
            sys.modules.pop(module_name)
    imported = importlib.import_module("onyx_cuda")

    assert imported.ENGINE_NAME == "Onyx for Windows"
    assert "onyx_cuda._grammar_native" not in requested_imports
    assert not any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for module_name in requested_imports
        for prefix in forbidden_imports
    )
