"""Parses test execution logs from various frameworks and evaluates test outcomes.

This module provides functionalities to:
- Define standardized test statuses (`TestStatus`, `ResolvedStatus`).
- Parse logs from different testing frameworks (PyTest, Django, etc.) to extract
  individual test case statuses.
- Compare evaluation run results against a gold standard.
- Compute metrics like Fail-to-Pass (F2P) and Pass-to-Pass (P2P).
- Determine an overall resolution status for an evaluation instance.

Much of the initial log parsing logic was adapted from SWE-bench (metrics/log_parsers.py).
"""

import re
from enum import Enum


class TestStatus(Enum):
    """Enumerates possible outcomes for an individual test case or a test run."""
    FAILED = "FAILED"
    """Test case explicitly failed an assertion."""
    PASSED = "PASSED"
    """Test case passed successfully."""
    SKIPPED = "SKIPPED"
    """Test case was skipped by the test runner."""
    ERROR = "ERROR"
    """Test case encountered an error during its execution (e.g., an unhandled exception in the test code itself)."""
    FRAMEWORK_ERROR = "FRAMEWORK_ERROR"
    """A general error occurred within the test framework or during test execution, not specific to a single test case's logic."""
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    """The entire test suite execution timed out."""
    MISSING_RESULT = "MISSING_RESULT"
    """An expected test case result was not found in the logs, without a global framework error."""


def parse_log_pytest(log: str) -> dict:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")
    for line in lines:
        line = line.strip()
        if line.endswith(" ... ok"):
            test = line.split(" ... ok")[0]
            test_status_map[test] = TestStatus.PASSED.value
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value
    return test_status_map


def parse_log_pytest_v2(log):
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_seaborn(log):
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log):
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_matplotlib = parse_log_pytest
parse_log_pydicom = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pylint = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_requests = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
}


# TODO: this is duplicated from execution.py
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"


# not from metrics/log_parsers.py
def get_logs_eval(repo_name: str, log_file_path: str) -> tuple[dict, TestStatus | None]:
    """
    Parse a log file to get test status for each test case.
    Parses a log file to extract test statuses for each test case.

    It uses a specific parser based on the `repo_name`. If global error indicators
    (like `TESTS_ERROR` or `TESTS_TIMEOUT`) are found in the log, an appropriate
    `overall_status` is returned alongside any partially parsed results.

    Args:
        repo_name (str): The name of the repository, used to select the correct log parser.
        log_file_path (str): The path to the log file to be parsed.

    Returns:
        tuple[dict, TestStatus | None]: A tuple containing:
            - A dictionary mapping test case names (str) to their statuses (str, values from `TestStatus` enum).
            - An overall `TestStatus` for the run if a global error (e.g., framework error, timeout)
              occurred, otherwise None.
    """
    # Ensure repo_name is valid to prevent KeyError
    if repo_name not in MAP_REPO_TO_PARSER:
        # Consider logging this event
        # logger.error(f"No parser found for repo: {repo_name}. Returning empty results and framework error.")
        return {}, TestStatus.FRAMEWORK_ERROR # Or a more specific "UnsupportedRepoError"

    log_parser = MAP_REPO_TO_PARSER[repo_name]
    parsed_results = {}
    overall_status: TestStatus | None = None

    try:
        with open(log_file_path) as f:
            # It's safer to read the whole content for multi-line parsing needs of some parsers
            # and then check for global errors.
            # A more advanced version might feed lines to parsers, but current parsers take full content.
            content = f.read()

            if TESTS_ERROR in content:
                overall_status = TestStatus.FRAMEWORK_ERROR
                # Attempt to parse what we have, error might be at the end
            elif TESTS_TIMEOUT in content:
                overall_status = TestStatus.EXECUTION_TIMEOUT
                # Attempt to parse what we have, timeout might be at the end

            # Run the parser even if a global error is detected,
            # as some results might be available before the error.
            # Parsers should ideally be robust enough not to fail catastrophically with partial/error logs.
            # This is a simplification; true line-by-line parsing before error is more complex
            # and would require rewriting all individual parsers.
            # For now, we parse the whole log, and the overall_status serves as an override/indicator.
            parsed_results = log_parser(content)

            # If a global status was set, it implies the parsed_results might be incomplete or unreliable.
            # The current parsers don't have a way to indicate "stopped early due to global error".
            # So, if overall_status is set, the `parsed_results` should be treated with caution.

    except FileNotFoundError:
        # Handle cases where the log file itself is missing
        return {}, TestStatus.FRAMEWORK_ERROR # Or a new specific status like LOG_FILE_MISSING
    except Exception:
        # Catch other potential errors during parsing (e.g., unexpected log format)
        return parsed_results, TestStatus.FRAMEWORK_ERROR # Or a more specific parsing error status

    return parsed_results, overall_status


def test_passed(case: str, sm: dict) -> bool:
    """Checks if a test case is present in the status map and marked as PASSED.

    Args:
        case (str): The name of the test case.
        sm (dict): The status map (test case name -> status string).

    Returns:
        bool: True if the test case passed, False otherwise.
    """
    return case in sm and sm[case] == TestStatus.PASSED.value


def test_failed(case: str, sm: dict, overall_status: TestStatus | None) -> bool:
    """Determines if a test case explicitly failed or errored.

    This function checks the status of an individual test case in the provided
    status map (`sm`). It does not consider a missing test as failed by this
    function's criteria; `test_missing` or logic in `get_eval_report` handles that.
    The `overall_status` of the test run is not directly used here to override
    explicit test statuses but can be used by calling functions for context.

    Args:
        case (str): The name of the test case.
        sm (dict): The status map (test case name -> status string).
        overall_status (TestStatus | None): The overall status of the test run.
            (Currently not used to alter outcome, but available for future logic).

    Returns:
        bool: True if the test case is present and its status is FAILED or ERROR,
              False otherwise (including if missing or PASSED/SKIPPED).
    """
    # overall_status might be used in future to infer failure if test is missing during a bad run.
    # For now, behavior is based on explicit status in sm.
    status = sm.get(case)
    if status is None:
        return False # A missing test is not considered explicitly FAILED by this function.
    return status in [TestStatus.FAILED.value, TestStatus.ERROR.value]

def test_missing(case: str, sm: dict) -> bool:
    """Checks if a test case is missing from the status map.

    Args:
        case (str): The name of the test case.
        sm (dict): The status map (test case name -> status string).

    Returns:
        bool: True if the test case is not found in the status map, False otherwise.
    """
    return case not in sm


# Result Categories used for evaluation reporting
FAIL_TO_PASS = "FAIL_TO_PASS"
"""Category for tests that were failing in the gold standard and are expected to pass after a patch."""
FAIL_TO_FAIL = "FAIL_TO_FAIL"
"""Category for tests that were failing in the gold standard and are still failing (or expected to fail)."""
PASS_TO_PASS = "PASS_TO_PASS"
"""Category for tests that were passing in the gold standard and are expected to remain passing."""
PASS_TO_FAIL = "PASS_TO_FAIL"
"""Category for tests that were passing in the gold standard but are now failing after a patch (regression)."""


# not from metrics/log_parsers.py
def get_eval_report(
    eval_sm: dict,
    gold_results: dict,
    overall_status: TestStatus | None,
    calculate_to_fail: bool = False,
) -> dict:
    """
    Create a report based on failure/pass change from gold results to eval results.
    Considers the overall status of the test run.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """
    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    f2p_missing = []  # Track missing results separately
    for test_case in gold_results[FAIL_TO_PASS]:
        if test_passed(test_case, eval_sm):
            f2p_success.append(test_case)
        elif test_failed(test_case, eval_sm, overall_status):
            f2p_failure.append(test_case)
        elif test_missing(test_case, eval_sm) and overall_status is None:
            # Only count as missing if there wasn't a global test run failure
            f2p_missing.append(test_case)
        # If overall_status is not None and test is missing, it's implicitly a run failure, not a specific test failure.
        # These won't be added to f2p_success or f2p_failure or f2p_missing.
        # Effectively, they are excluded from these specific F2P stats if a global error occurred and they didn't explicitly pass.

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    p2p_missing = [] # Track missing results separately
    for test_case in gold_results[PASS_TO_PASS]:
        if test_passed(test_case, eval_sm):
            p2p_success.append(test_case)
        elif test_failed(test_case, eval_sm, overall_status):
            p2p_failure.append(test_case)
        elif test_missing(test_case, eval_sm) and overall_status is None:
            p2p_missing.append(test_case)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
            "missing": f2p_missing,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
            "missing": p2p_missing,
        },
    }

    f2f_success = []
    f2f_failure = []
    f2f_missing = []
    p2f_success = []
    p2f_failure = []
    p2f_missing = []
    if calculate_to_fail:
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                f2f_success.append(test_case)
            elif test_failed(test_case, eval_sm, overall_status):
                f2f_failure.append(test_case)
            elif test_missing(test_case, eval_sm) and overall_status is None:
                f2f_missing.append(test_case)

        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                p2f_success.append(test_case)
            elif test_failed(test_case, eval_sm, overall_status):
                p2f_failure.append(test_case)
            elif test_missing(test_case, eval_sm) and overall_status is None:
                p2f_missing.append(test_case)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
                "missing": f2f_missing,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
                "missing": p2f_missing,
            },
        }
    )
    return results


class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"  # No resolution or regression
    PARTIAL = "RESOLVED_PARTIAL"
    """Patch partially resolved failing tests (some F2P passed) and maintained all passing tests (P2P)."""
    FULL = "RESOLVED_FULL"
    """Patch fully resolved all failing tests (all F2P passed) and maintained all passing tests (P2P)."""
    PARTIAL_P2P_NA = "RESOLVED_PARTIAL_P2P_NA"
    """Patch partially resolved failing tests, and there were no Pass-to-Pass (P2P) tests to maintain."""
    FULL_P2P_NA = "RESOLVED_FULL_P2P_NA"
    """Patch fully resolved all failing tests, and there were no Pass-to-Pass (P2P) tests to maintain."""
    NO_P2P_REGRESSION = "RESOLVED_NO_P2P_REGRESSION"
    """Resolution of Fail-to-Pass (F2P) tests might vary, but at least one Pass-to-Pass (P2P) test failed (a regression)."""
    NO_MISSING_RESULTS = "RESOLVED_NO_MISSING_RESULTS"
    """Some expected test results were missing from the evaluation logs, and no global test run error was reported."""
    NO_FRAMEWORK_ERROR = "RESOLVED_NO_FRAMEWORK_ERROR"
    """The test execution failed due to a framework error, making detailed resolution status indeterminable."""
    NO_EXECUTION_TIMEOUT = "RESOLVED_NO_EXECUTION_TIMEOUT"
    """The test execution timed out, making detailed resolution status indeterminable."""


def compute_fail_to_pass(report: dict, overall_status: TestStatus | None) -> float | None:
    """Computes the Fail-to-Pass (F2P) success rate.

    This rate is the number of successfully resolved F2P tests divided by the total
    number of F2P tests considered (those that were either successful or failed,
    excluding missing ones if no global error occurred).

    Args:
        report (dict): The evaluation report from `get_eval_report`.
        overall_status (TestStatus | None): The overall status of the test run.
            (Currently used to decide if missing tests impact reliability).

    Returns:
        float | None: The F2P rate (0.0 to 1.0), or None if the metric cannot be
                      reliably calculated (e.g., no F2P tests to consider).
    """
    f2p_data = report[FAIL_TO_PASS]
    # If overall_status indicates a global problem, F2P calculation might be unreliable
    # if there are missing tests that weren't explicitly PASSED.
    # For now, we only exclude explicitly missing ones if no global error.
    if overall_status is None and f2p_data["missing"]:
        # If results are missing and no global error, we cannot make a definitive statement.
        # Or, alternatively, count missing as not-passed. For now, let's be conservative.
        # This behavior might need refinement based on desired strictness.
        pass # Keep as is for now, missing are not counted in total for percentage.

    total_considered_f2p = len(f2p_data["success"]) + len(f2p_data["failure"])
    if total_considered_f2p == 0:
        # No F2P tests were successfully run and evaluated (e.g. all missing, or no F2P tests in gold)
        # If there were originally F2P tests in gold, but all are now missing without a global error, this is problematic.
        # If there were no F2P tests in gold, this is fine.
        # This function just computes the ratio, get_resolution_status interprets it.
        return None # Or 1.0 if we define "no tests to fix" as 100% fixed. Let's use None.
    return len(f2p_data["success"]) / total_considered_f2p


def compute_pass_to_pass(report: dict, overall_status: TestStatus | None) -> float | None:
    """
    Computes the Pass-to-Pass (P2P) success rate (maintenance rate).

    This rate is the number of P2P tests that remained passing divided by the
    total number of P2P tests considered (those that were either successful or failed,
    excluding missing ones if no global error occurred).

    Args:
        report (dict): The evaluation report from `get_eval_report`.
        overall_status (TestStatus | None): The overall status of the test run.
            (Currently used to decide if missing tests impact reliability).

    Returns:
        float | None: The P2P rate (0.0 to 1.0), or None if the metric cannot be
                      reliably calculated (e.g., no P2P tests to consider).
    """
    p2p_data = report[PASS_TO_PASS]
    if overall_status is None and p2p_data["missing"]:
        # Similar to F2P, if results are missing without a global error, the metric is less reliable.
        pass # Current logic counts missing as not part of the considered total for rate calculation.

    total_considered_p2p = len(p2p_data["success"]) + len(p2p_data["failure"])
    if total_considered_p2p == 0:
        return None # No P2P tests to consider for the ratio.
    return len(p2p_data["success"]) / total_considered_p2p


def get_resolution_status(report: dict, overall_status: TestStatus | None) -> ResolvedStatus:
    """Determines the overall resolution status of an evaluation instance.

    This function synthesizes Fail-to-Pass (F2P) and Pass-to-Pass (P2P)
    metrics, along with any global test run statuses, to categorize the outcome.

    Args:
        report (dict): The evaluation report generated by `get_eval_report`.
        overall_status (TestStatus | None): The overall status of the test run
            (e.g., FRAMEWORK_ERROR, EXECUTION_TIMEOUT) from `get_logs_eval`.

    Returns:
        ResolvedStatus: The determined overall resolution status.
    """
    if overall_status == TestStatus.FRAMEWORK_ERROR:
        return ResolvedStatus.NO_FRAMEWORK_ERROR
    if overall_status == TestStatus.EXECUTION_TIMEOUT:
        return ResolvedStatus.NO_EXECUTION_TIMEOUT

    # Check for missing results not covered by a global error
    if overall_status is None and (report[FAIL_TO_PASS]["missing"] or report[PASS_TO_PASS]["missing"]):
        # This condition means some tests were expected but results are missing,
        # and it wasn't due to a global framework error or timeout.
        return ResolvedStatus.NO_MISSING_RESULTS

    f2p_metric = compute_fail_to_pass(report, overall_status)
    p2p_metric = compute_pass_to_pass(report, overall_status)

    # Handle cases where P2P is not applicable (no P2P tests in gold, or all were missing without global error)
    if p2p_metric is None:
        if f2p_metric is None: # No F2P tests to consider either
             # This case implies no F2P tests in gold or all F2P tests were missing.
             # If there are no gold F2P and no gold P2P tests, it's effectively full resolution by vacuous truth.
             # However, if there were gold F2P/P2P tests but all are missing, it's NO_MISSING_RESULTS (handled above).
             # This path should ideally only be hit if gold standard had NO F2P and NO P2P tests.
            return ResolvedStatus.FULL_P2P_NA # Or a more specific "NO_APPLICABLE_TESTS"
        if f2p_metric == 1.0:
            return ResolvedStatus.FULL_P2P_NA
        elif f2p_metric > 0: # f2p_metric < 1.0 implicitly
            return ResolvedStatus.PARTIAL_P2P_NA
        else: # f2p_metric == 0.0 (or also None if no F2P tests but some P2P were there - though p2p_metric is None here)
            return ResolvedStatus.NO # No F2P progress, and P2P N/A

    # P2P metric is available (i.e., not None)
    if p2p_metric < 1.0:
        # Any failure in P2P tests is a regression or failure to maintain.
        return ResolvedStatus.NO_P2P_REGRESSION

    # At this point, p2p_metric == 1.0
    if f2p_metric is None: # No F2P tests to consider, but P2P is 1.0
        return ResolvedStatus.FULL # All (zero) failing tests fixed, and all passing tests maintained.
    if f2p_metric == 1.0:
        return ResolvedStatus.FULL
    elif f2p_metric > 0: # f2p_metric < 1.0 implicitly
        return ResolvedStatus.PARTIAL
    else: # f2p_metric == 0.0
        return ResolvedStatus.NO
