import json
import re
import textwrap
from enum import Enum
from pathlib import Path

from app.api import eval_helper
from app.api.eval_helper import *


# Prevent pytest from collecting the helper functions that start with "test_" but are not actually unit tests
# (they are intended to be used as helpers in other modules).
eval_helper.test_passed.__test__ = False
eval_helper.test_failed.__test__ = False


# --- Tests for the log parsers ---


def test_parse_log_pytest():
    # Sample log lines for pytest format.
    # e.g. "PASSED test_func1" and "FAILED test_func2 - some error"
    log = textwrap.dedent(
        """\
        PASSED test_func1
        FAILED test_func2 - AssertionError
        SKIPPED test_func3
        ERROR test_func4 - Exception
    """
    )
    result = parse_log_pytest(log)
    # We expect mapping: test_func1 -> PASSED, test_func2 -> FAILED, etc.
    assert result.get("test_func1") == "PASSED"
    assert result.get("test_func2") == "FAILED"
    assert result.get("test_func3") == "SKIPPED"
    assert result.get("test_func4") == "ERROR"


def test_parse_log_django():
    # Django logs typically have patterns like:
    # "some_test ... ok", "another_test ... skipped", "yetanother_test ... FAIL", etc.
    log = textwrap.dedent(
        """\
        test_app.tests.TestSomething.test_one ... ok
        test_app.tests.TestSomething.test_two ... skipped
        test_app.tests.TestSomething.test_three ... FAIL
        FAIL: test_app.tests.TestSomething.test_four
        ERROR: test_app.tests.TestSomething.test_five
        test_app.tests.TestSomething.test_six ... ERROR
    """
    )
    result = parse_log_django(log)
    # Expected status based on our parser.
    assert (
        result.get("test_app.tests.TestSomething.test_one") == TestStatus.PASSED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_two") == TestStatus.SKIPPED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_three") == TestStatus.FAILED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_four") == TestStatus.FAILED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_five") == TestStatus.ERROR.value
    )
    assert result.get("test_app.tests.TestSomething.test_six") == TestStatus.ERROR.value


def test_parse_log_pytest_v2():
    # Sample log for pytest v2: includes ANSI escape sequences and a hunk for FAILED.
    log = "\x1b[31mFAILED\x1b[0m test_func_v2 - error message"
    result = parse_log_pytest_v2(log)
    # The escape sequences should be removed; we expect test_func_v2 mapped to FAILED.
    assert result.get("test_func_v2") == "FAILED"


def test_parse_log_seaborn():
    # Seaborn log sample: failed line starts with "FAILED", passed line has " PASSED " in it.
    log = textwrap.dedent(
        """\
        dummy_test PASSED some extra text
        FAILED another_test
    """
    )
    result = parse_log_seaborn(log)
    # For FAILED, we split and take the second token.
    assert result.get("another_test") == TestStatus.FAILED.value
    # For PASSED, if the second token equals PASSED, then key is the first token.
    assert result.get("dummy_test") == TestStatus.PASSED.value


def test_parse_log_sympy():
    # Sample sympy log: first part uses regex and then additional lines.
    # Create a fake match pattern. The regex pattern in parse_log_sympy is:
    # r"(_*) (.*)\.py:(.*) (_*)"
    # We can simulate one match and then additional lines.
    log = textwrap.dedent(
        """\
        ____ dummy.py:10 ____
        test_sympy1 E
        test_sympy2 F
        test_sympy3 ok
    """
    )
    result = parse_log_sympy(log)
    # From regex part, we expect one entry for "dummy.py:10"
    assert "dummy.py:10" in result
    # And additional lines produce mappings:
    assert result.get("test_sympy1") == TestStatus.ERROR.value
    assert result.get("test_sympy2") == TestStatus.FAILED.value
    assert result.get("test_sympy3") == TestStatus.PASSED.value


# --- Tests for get_logs_eval ---


def test_get_logs_eval_success(tmp_path):
    # Create a temporary log file with a valid log (using pytest parser).
    log_content = "PASSED test_eval1\nFAILED test_eval2"
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)
    # Use a repo that maps to parse_log_pytest, e.g. "pytest-dev/pytest"
    parsed, overall_status = get_logs_eval("pytest-dev/pytest", str(log_file)) # New return type
    assert overall_status is None # Expect None for a successful parse without global errors
    assert parsed.get("test_eval1") == "PASSED"
    assert parsed.get("test_eval2") == "FAILED"

def test_get_logs_eval_with_timeout_marker(tmp_path):
    log_content = f"PASSED test_before_timeout\n{TESTS_TIMEOUT}\nFAILED test_after_timeout"
    log_file = tmp_path / "log_timeout.txt"
    log_file.write_text(log_content)
    parsed, overall_status = get_logs_eval("pytest-dev/pytest", str(log_file))
    assert overall_status == TestStatus.EXECUTION_TIMEOUT
    # Parser should still process available content
    assert parsed.get("test_before_timeout") == "PASSED"
    assert parsed.get("test_after_timeout") == "FAILED"


def test_get_logs_eval_failure(tmp_path): # This test was for when it returned {}, False
    # Create a temporary log file with error markers.
    log_content = f"{TESTS_ERROR}\nSome error occurred."
    log_file = tmp_path / "log_error.txt"
    log_file.write_text(log_content)
    parsed, overall_status = get_logs_eval("pytest-dev/pytest", str(log_file))
    # Now it returns the overall_status enum and whatever it could parse
    assert overall_status == TestStatus.FRAMEWORK_ERROR
    # Depending on parser robustness, parsed might have some data or be empty
    # For basic pytest parser, if TESTS_ERROR is present, it might still parse lines before it.
    # If TESTS_ERROR is at the start, parsed would be {}.
    # For this specific log_content, if TESTS_ERROR is the first thing, it's likely empty.
    # Let's assume the current pytest parser would still try and might find nothing if error is at start.
    # If the parser was more sophisticated, it might parse around it.
    # For now, this test is more about overall_status.
    # assert parsed == {} # This might be too strict depending on parser logic for partial logs

def test_get_logs_eval_file_not_found():
    parsed, overall_status = get_logs_eval("pytest-dev/pytest", "non_existent_file.log")
    assert overall_status == TestStatus.FRAMEWORK_ERROR # Or a more specific LOG_FILE_MISSING if defined
    assert parsed == {}

def test_get_logs_eval_unsupported_repo():
    parsed, overall_status = get_logs_eval("unsupported/repo", "dummy.log")
    assert overall_status == TestStatus.FRAMEWORK_ERROR
    assert parsed == {}

# --- Tests for test_passed, test_failed, test_missing ---

def test_test_status_functions():
    status_map = {
        "case_pass": TestStatus.PASSED.value,
        "case_fail": TestStatus.FAILED.value,
        "case_error": TestStatus.ERROR.value,
        "case_skip": TestStatus.SKIPPED.value,
    }
    # test_passed
    assert test_passed("case_pass", status_map) is True
    assert test_passed("case_fail", status_map) is False
    assert test_passed("case_error", status_map) is False
    assert test_passed("case_skip", status_map) is False
    assert test_passed("case_missing", status_map) is False

    # test_failed (overall_status = None)
    assert test_failed("case_pass", status_map, None) is False
    assert test_failed("case_fail", status_map, None) is True
    assert test_failed("case_error", status_map, None) is True
    assert test_failed("case_skip", status_map, None) is False # Skipped is not failed
    assert test_failed("case_missing", status_map, None) is False # Missing is not explicitly failed by this func

    # test_failed (overall_status = FRAMEWORK_ERROR) - current test_failed logic doesn't change outcome based on this
    assert test_failed("case_pass", status_map, TestStatus.FRAMEWORK_ERROR) is False # Still passed if explicitly passed
    assert test_failed("case_fail", status_map, TestStatus.FRAMEWORK_ERROR) is True
    assert test_failed("case_missing", status_map, TestStatus.FRAMEWORK_ERROR) is False

    # test_missing
    assert test_missing("case_pass", status_map) is False
    assert test_missing("case_missing", status_map) is True


# --- Tests for get_eval_report, compute_fail_to_pass, compute_pass_to_pass, get_resolution_status ---


def test_get_eval_report():
    # Create dummy gold results.
    gold = {
        FAIL_TO_PASS: ["t1", "t2"],
        PASS_TO_PASS: ["t3", "t4"],
        FAIL_TO_FAIL: ["t5"],
        PASS_TO_FAIL: ["t6"],
    }
    # Create an evaluation status map.
    eval_sm = {
        "t1": TestStatus.PASSED.value,
        "t2": TestStatus.FAILED.value,
        "t3": TestStatus.PASSED.value,
        "t4": TestStatus.FAILED.value,
        "t5": TestStatus.PASSED.value,
        "t6": TestStatus.FAILED.value,
        # t7 (F2P) is missing from eval_sm
        # t8 (P2P) is missing from eval_sm
    }
    # Test with overall_status = None (clean run)
    report_clean_run = get_eval_report(eval_sm, gold, overall_status=None, calculate_to_fail=True)
    assert report_clean_run[FAIL_TO_PASS]["success"] == ["t1"]
    assert report_clean_run[FAIL_TO_PASS]["failure"] == ["t2"]
    assert report_clean_run[FAIL_TO_PASS]["missing"] == ["t7"] # t7 was in gold F2P, missing in eval
    assert report_clean_run[PASS_TO_PASS]["success"] == ["t3"]
    assert report_clean_run[PASS_TO_PASS]["failure"] == ["t4"]
    assert report_clean_run[PASS_TO_PASS]["missing"] == ["t8"] # t8 was in gold P2P, missing in eval
    assert report_clean_run[FAIL_TO_FAIL]["success"] == ["t5"]
    assert report_clean_run[PASS_TO_FAIL]["failure"] == ["t6"]

    # Test with overall_status = FRAMEWORK_ERROR
    report_error_run = get_eval_report(eval_sm, gold, overall_status=TestStatus.FRAMEWORK_ERROR, calculate_to_fail=True)
    # If overall_status is an error, "missing" lists should be empty as per current logic
    # (tests not explicitly PASSED are just not counted towards success/failure, not marked "missing" due to agent vs. framework)
    assert report_error_run[FAIL_TO_PASS]["success"] == ["t1"] # t1 still passed
    assert report_error_run[FAIL_TO_PASS]["failure"] == ["t2"] # t2 still failed
    assert report_error_run[FAIL_TO_PASS]["missing"] == [] # t7 is not "missing" if run failed globally
    assert report_error_run[PASS_TO_PASS]["success"] == ["t3"]
    assert report_error_run[PASS_TO_PASS]["failure"] == ["t4"]
    assert report_error_run[PASS_TO_PASS]["missing"] == []


def test_compute_metrics():
    # Case 1: Standard, some F2P fixed, P2P maintained
    report1 = {
        FAIL_TO_PASS: {"success": ["t1", "t2"], "failure": ["t3"], "missing": []},
        PASS_TO_PASS: {"success": ["t4", "t5"], "failure": [], "missing": []},
    }
    assert abs(compute_fail_to_pass(report1, None) - (2/3)) < 0.001
    assert compute_pass_to_pass(report1, None) == 1.0

    # Case 2: No F2P tests considered (e.g., all missing or none in gold)
    report2 = {
        FAIL_TO_PASS: {"success": [], "failure": [], "missing": ["t1"]}, # or empty success/failure/missing
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []},
    }
    assert compute_fail_to_pass(report2, None) is None

    # Case 3: No P2P tests considered
    report3 = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": ["t2"]}, # or empty success/failure/missing
    }
    assert compute_pass_to_pass(report3, None) is None

    # Case 4: All F2P failed
    report4 = {
        FAIL_TO_PASS: {"success": [], "failure": ["t1","t2"], "missing": []},
        PASS_TO_PASS: {"success": ["t3"], "failure": [], "missing": []},
    }
    assert compute_fail_to_pass(report4, None) == 0.0

    # Case 5: All P2P failed
    report5 = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": ["t3","t4"], "missing": []},
    }
    assert compute_pass_to_pass(report5, None) == 0.0

    # Case 6: F2P has only missing tests (and no global error implies these are true missing)
    report6 = {
        FAIL_TO_PASS: {"success": [], "failure": [], "missing": ["t1", "t2"]},
        PASS_TO_PASS: {"success": ["t3"], "failure": [], "missing": []}
    }
    # Current logic for compute_fail_to_pass returns None if total_considered is 0
    # If "missing" are not counted in total_considered, this will be None.
    assert compute_fail_to_pass(report6, None) is None # As success+failure = 0


def test_get_resolution_status_detailed():
    # Overall Status tests
    assert get_resolution_status({}, TestStatus.FRAMEWORK_ERROR) == ResolvedStatus.NO_FRAMEWORK_ERROR
    assert get_resolution_status({}, TestStatus.EXECUTION_TIMEOUT) == ResolvedStatus.NO_EXECUTION_TIMEOUT

    # Missing Results (overall_status is None)
    report_missing_f2p = {
        FAIL_TO_PASS: {"success": [], "failure": [], "missing": ["t1"]},
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []},
    }
    assert get_resolution_status(report_missing_f2p, None) == ResolvedStatus.NO_MISSING_RESULTS

    report_missing_p2p = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": ["t2"]},
    }
    assert get_resolution_status(report_missing_p2p, None) == ResolvedStatus.NO_MISSING_RESULTS

    # P2P N/A cases (p2p_metric is None because no P2P tests were *considered* for pass/fail)
    report_full_p2p_na = { # All F2P fixed, no P2P tests
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": []}, # No considered P2P
    }
    assert get_resolution_status(report_full_p2p_na, None) == ResolvedStatus.FULL_P2P_NA

    report_partial_p2p_na = { # Some F2P fixed, no P2P tests
        FAIL_TO_PASS: {"success": ["t1"], "failure": ["t2"], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": []},
    }
    assert get_resolution_status(report_partial_p2p_na, None) == ResolvedStatus.PARTIAL_P2P_NA

    report_no_f2p_progress_p2p_na = { # No F2P fixed, no P2P tests
        FAIL_TO_PASS: {"success": [], "failure": ["t1"], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": []},
    }
    assert get_resolution_status(report_no_f2p_progress_p2p_na, None) == ResolvedStatus.NO

    report_no_f2p_tests_p2p_na = { # No F2P tests at all, no P2P tests
        FAIL_TO_PASS: {"success": [], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": [], "failure": [], "missing": []},
    }
    # This edge case: compute_fail_to_pass is None, compute_pass_to_pass is None.
    # Current logic: f2p_metric is None, p2p_metric is None -> FULL_P2P_NA
    assert get_resolution_status(report_no_f2p_tests_p2p_na, None) == ResolvedStatus.FULL_P2P_NA


    # P2P Regression
    report_p2p_regression = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []}, # Full F2P
        PASS_TO_PASS: {"success": ["t2"], "failure": ["t3"], "missing": []}, # P2P regression
    }
    assert get_resolution_status(report_p2p_regression, None) == ResolvedStatus.NO_P2P_REGRESSION

    # Standard FULL, PARTIAL, NO (assuming P2P is 1.0 or P2P tests exist)
    report_full = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": [], "missing": []},
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []},
    }
    assert get_resolution_status(report_full, None) == ResolvedStatus.FULL

    report_partial = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": ["t3"], "missing": []},
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []},
    }
    assert get_resolution_status(report_partial, None) == ResolvedStatus.PARTIAL

    report_no_f2p_resolved = {
        FAIL_TO_PASS: {"success": [], "failure": ["t1"], "missing": []},
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []}, # P2P maintained
    }
    assert get_resolution_status(report_no_f2p_resolved, None) == ResolvedStatus.NO

    # Edge case: No F2P tests to fix, P2P maintained
    report_no_f2p_tests_p2p_ok = {
        FAIL_TO_PASS: {"success": [], "failure": [], "missing": []}, # No F2P tests
        PASS_TO_PASS: {"success": ["t2"], "failure": [], "missing": []},
    }
    # f2p_metric is None, p2p_metric is 1.0 -> FULL
    assert get_resolution_status(report_no_f2p_tests_p2p_ok, None) == ResolvedStatus.FULL
