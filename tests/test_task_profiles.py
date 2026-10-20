import pytest

from evalom.task_profiles import (
    ProfileRequirement,
    TaskProfile,
    builtin_profile,
    evaluate_profile,
    resolve_profile,
)


def _profile(requirements=None):
    return TaskProfile.from_dict({
        "profile_id": "demo.feature",
        "profile_version": "1.0",
        "task_type": "feature_implementation",
        "requirements": requirements or [
            {"requirement_id": "eval", "kind": "evaluator", "evaluator": "exact_match"},
            {"requirement_id": "tests", "kind": "evidence", "evidence_type": "test_result"},
        ],
    })


@pytest.mark.parametrize("task_type", ["feature_implementation", "bug_fix", "test_generation", "code_review"])
def test_builtin_profiles_are_declarative_and_versioned(task_type):
    profile = builtin_profile(task_type)
    resolved = resolve_profile(profile)
    assert resolved.profile_id == f"builtin.{task_type}"
    assert resolved.profile_version == "1.0"
    assert len(resolved.required) == 2
    assert len(profile.definition_hash) == 64


def test_required_checks_and_evidence_are_not_converted_to_success():
    result = evaluate_profile(_profile(), checks={}, evidence=[])
    assert result["status"] == "insufficient_evidence"
    assert {item["status"] for item in result["checks"]} == {"not_run"}

    result = evaluate_profile(
        _profile(),
        checks={"exact_match": {"passed": True, "status": "valid"}},
        evidence=[{"evidence_type": "test_result", "lifecycle_status": "active", "digest_status": "unavailable"}],
    )
    assert result["status"] == "insufficient_evidence"


def test_failed_required_check_cannot_be_overridden_by_optional_success():
    profile = _profile([
        {"requirement_id": "required", "kind": "evaluator", "evaluator": "exact_match"},
        {"requirement_id": "optional", "kind": "evaluator", "evaluator": "contains", "required": False},
    ])
    result = evaluate_profile(
        profile,
        checks={
            "exact_match": {"passed": False, "status": "failed"},
            "contains": {"passed": True, "status": "valid"},
        },
    )
    assert result["status"] == "failed"
    assert result["optional_checks"] == ["optional"]


def test_profile_validation_rejects_unknown_fields_versions_duplicates_and_unsafe_content():
    with pytest.raises(ValueError, match="Unknown profile field"):
        TaskProfile.from_dict({**_profile().to_dict(), "execute": "rm -rf /"})
    with pytest.raises(ValueError, match="Unsupported profile_version"):
        TaskProfile.from_dict({**_profile().to_dict(), "profile_version": "9.0"})
    with pytest.raises(ValueError, match="unique"):
        _profile([
            {"requirement_id": "same", "kind": "evaluator", "evaluator": "exact_match"},
            {"requirement_id": "same", "kind": "evaluator", "evaluator": "contains"},
        ])


def test_profile_rejects_unknown_evaluators_and_arbitrary_paths():
    with pytest.raises(ValueError, match="Unsupported evaluator"):
        ProfileRequirement.from_dict({
            "requirement_id": "evil",
            "kind": "evaluator",
            "evaluator": "os.system('id')",
        })
    with pytest.raises(ValueError, match="Unsupported evidence type"):
        ProfileRequirement.from_dict({
            "requirement_id": "path",
            "kind": "evidence",
            "evidence_type": "/tmp/secret",
        })


def test_resolution_is_deterministic_and_serializable():
    first = resolve_profile(_profile()).to_dict()
    second = resolve_profile(_profile()).to_dict()
    assert first == second
    assert first["required"][0]["requirement_id"] == "eval"
