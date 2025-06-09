import json
import pytest
from app.data_structures import FunctionCallIntent, OpenaiFunction


# Define a dummy OpenaiFunction for testing.
class DummyOpenaiFunction:
    def __init__(self, arguments, name):
        self.arguments = arguments
        self.name = name

    def __eq__(self, other):
        return (
            isinstance(other, DummyOpenaiFunction)
            and self.arguments == other.arguments
            and self.name == other.name
        )

    def __str__(self):
        return f"DummyOpenaiFunction(name={self.name}, arguments={self.arguments})"


# Automatically replace OpenaiFunction in the module with DummyOpenaiFunction for tests.
@pytest.fixture(autouse=True)
def use_dummy_openai_function(monkeypatch):
    monkeypatch.setattr(
        "app.data_structures.OpenaiFunction",
        DummyOpenaiFunction,
    )


def test_function_call_intent_default():
    func_name = "test_func"
    arguments = {"arg1": "value1", "arg2": "value2"}
    # When no openai_func is provided, it should create one using DummyOpenaiFunction.
    intent = FunctionCallIntent(func_name, arguments, None)
    # Check that func_name is set.
    assert intent.func_name == func_name
    # Check that arg_values equals the provided arguments.
    assert intent.arg_values == arguments
    # Check that openai_func is created and has the proper values.
    assert isinstance(intent.openai_func, DummyOpenaiFunction)
    expected_args = json.dumps(arguments)
    assert intent.openai_func.arguments == expected_args
    assert intent.openai_func.name == func_name


def test_function_call_intent_with_openai_func():
    func_name = "another_func"
    arguments = {"x": "1"}
    # Create a dummy openai function.
    dummy_func = DummyOpenaiFunction(
        arguments=json.dumps({"x": "override"}), name="dummy"
    )
    intent = FunctionCallIntent(func_name, arguments, dummy_func)
    # The provided openai_func should be used.
    assert intent.openai_func == dummy_func
    # And arg_values should still reflect the passed arguments.
    assert intent.arg_values == arguments


def test_to_dict():
    func_name = "func_to_dict"
    arguments = {"a": "b"}
    intent = FunctionCallIntent(func_name, arguments, None)
    result = intent.to_dict()
    expected = {"func_name": func_name, "arguments": arguments}
    assert result == expected


def test_to_dict_with_result():
    func_name = "func_with_result"
    arguments = {"key": "val"}
    intent = FunctionCallIntent(func_name, arguments, None)
    result_true = intent.to_dict_with_result(True)
    result_false = intent.to_dict_with_result(False)
    expected_true = {"func_name": func_name, "arguments": arguments, "call_ok": True}
    expected_false = {"func_name": func_name, "arguments": arguments, "call_ok": False}
    assert result_true == expected_true
    assert result_false == expected_false


def test_str_method():
    func_name = "str_func"
    arguments = {"param": "123"}
    intent = FunctionCallIntent(func_name, arguments, None)
    s = str(intent)
    # The string representation should include the function name and the arguments.
    assert func_name in s
    assert str(arguments) in s

# Tests for EvaluationPayload
from app.data_structures import EvaluationPayload
from app.api.eval_helper import ResolvedStatus # Assuming ResolvedStatus is in eval_helper

def test_evaluation_payload_instantiation():
    """Test basic instantiation of EvaluationPayload."""
    payload = EvaluationPayload(
        status=ResolvedStatus.FULL,
        message="All tests passed.",
        details={"F2P_count": 5, "P2P_count": 10}
    )
    assert payload.status == ResolvedStatus.FULL
    assert payload.message == "All tests passed."
    assert payload.details == {"F2P_count": 5, "P2P_count": 10}

def test_evaluation_payload_to_llm_feedback_string_no_details():
    """Test to_llm_feedback_string with no details."""
    payload = EvaluationPayload(
        status=ResolvedStatus.NO,
        message="Patch failed."
    )
    expected_string = "Patch evaluation result: RESOLVED_NO. Patch failed."
    assert payload.to_llm_feedback_string() == expected_string

def test_evaluation_payload_to_llm_feedback_string_empty_details():
    """Test to_llm_feedback_string with empty details."""
    payload = EvaluationPayload(
        status=ResolvedStatus.PARTIAL,
        message="Partially resolved.",
        details={}
    )
    expected_string = "Patch evaluation result: RESOLVED_PARTIAL. Partially resolved."
    assert payload.to_llm_feedback_string() == expected_string

def test_evaluation_payload_to_llm_feedback_string_with_f2p_p2p_details():
    """Test to_llm_feedback_string with F2P and P2P details."""
    details_data = {
        "FAIL_TO_PASS": {
            "success": ["test_a", "test_b"],
            "failure": ["test_c"],
            "missing": []
        },
        "PASS_TO_PASS": {
            "success": ["test_d"],
            "failure": [],
            "missing": ["test_e"]
        }
    }
    payload = EvaluationPayload(
        status=ResolvedStatus.PARTIAL,
        message="Mixed results.",
        details=details_data
    )
    # Expected string construction can be a bit complex due to potential ordering or specific formatting
    # For now, check for key components
    result_string = payload.to_llm_feedback_string()
    assert "Patch evaluation result: RESOLVED_PARTIAL. Mixed results." in result_string
    assert "Test outcome details:" in result_string
    assert "Fail-to-Pass: 2 succeeded, 1 failed, 0 missing." in result_string
    assert "Pass-to-Pass: 1 succeeded, 0 failed, 1 missing." in result_string

def test_evaluation_payload_to_llm_feedback_string_with_only_one_category_details():
    """Test to_llm_feedback_string with only F2P details."""
    details_data = {
        "FAIL_TO_PASS": {
            "success": ["test_a"],
            "failure": [],
            "missing": []
        }
    }
    payload = EvaluationPayload(
        status=ResolvedStatus.FULL_P2P_NA, # Example status
        message="F2P resolved, no P2P tests.",
        details=details_data
    )
    result_string = payload.to_llm_feedback_string()
    assert "Patch evaluation result: RESOLVED_FULL_P2P_NA. F2P resolved, no P2P tests." in result_string
    assert "Test outcome details:" in result_string
    assert "Fail-to-Pass: 1 succeeded, 0 failed, 0 missing." in result_string
    assert "Pass-to-Pass" not in result_string # P2P data is not present

def test_evaluation_payload_to_llm_feedback_string_with_other_details():
    """Test to_llm_feedback_string with other arbitrary details."""
    details_data = {
        "custom_metric": 0.75,
        "notes": "Some notes here"
    }
    payload = EvaluationPayload(
        status=ResolvedStatus.NO,
        message="Evaluation with custom metrics.",
        details=details_data
    )
    result_string = payload.to_llm_feedback_string()
    assert "Patch evaluation result: RESOLVED_NO. Evaluation with custom metrics." in result_string
    # The current implementation of to_llm_feedback_string only specifically formats F2P/P2P.
    # Other details are json.dumps'd.
    assert f"\nDetails: {json.dumps(details_data)}" in result_string

def test_evaluation_payload_to_llm_feedback_string_f2p_p2p_zero_counts():
    """Test to_llm_feedback_string when F2P/P2P success/failure/missing are all zero."""
    details_data = {
        "FAIL_TO_PASS": {
            "success": [],
            "failure": [],
            "missing": []
        },
        "PASS_TO_PASS": {
            "success": [],
            "failure": [],
            "missing": []
        }
    }
    payload = EvaluationPayload(
        status=ResolvedStatus.FULL,
        message="All good, but no specific F2P/P2P data to show.",
        details=details_data
    )
    result_string = payload.to_llm_feedback_string()
    # According to current to_llm_feedback_string logic, if all counts are zero, the category line is not added.
    assert "Fail-to-Pass" not in result_string
    assert "Pass-to-Pass" not in result_string
    # If there are other details, they would be json.dumps'd. If only these empty F2P/P2P, details section might be minimal.
    # Current implementation: if detail_items is empty, it falls back to json.dumps(self.details)
    assert f"\nDetails: {json.dumps(details_data)}" in result_string
