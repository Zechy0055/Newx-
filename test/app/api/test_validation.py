import ast
import itertools
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
import textwrap

import pytest
from unidiff import PatchSet

from app import config
from app.api import validation
from app.api.validation import *
from app.data_structures import MethodId
from app.task import SweTask, Task
from app.agents.agent_write_patch import PatchHandle


from app.data_structures import EvaluationPayload # For type checking
from app.api.eval_helper import ResolvedStatus, TestStatus, FAIL_TO_PASS, PASS_TO_PASS, FAIL_TO_FAIL, PASS_TO_FAIL # For mocking and asserting

# --- Dummy and Fixture Setup ---

class DummyTask(Task): # Basic Task for non-SWE-bench scenarios
    def __init__(self, task_id="dummy_task", project_path="dummy_project_path", repo_name="dummy/repo"):
        self._task_id = task_id
        self.project_path = project_path
        self.repo_name = repo_name # For get_logs_eval

    def get_instance_id(self) -> str:
        return self._task_id

    def validate(self, patch_content: str) -> tuple[bool, str, str, str]:
        # Ensure log files are actually created for Path().exists() checks
        # These will be moved by validate_and_move_logs in evaluate_patch
        # So, the test needs to expect them in the output_dir
        dummy_eval_log = Path(tempfile.gettempdir()) / f"dummy_eval_{Path(tempfile.NamedTemporaryFile(delete=False).name).name}.log"
        dummy_orig_log = Path(tempfile.gettempdir()) / f"dummy_orig_{Path(tempfile.NamedTemporaryFile(delete=False).name).name}.log"
        dummy_eval_log.write_text("EVAL LOG CONTENT")
        dummy_orig_log.write_text("ORIG LOG CONTENT")
        # Return True (regression check passed), message, and paths to *temporary* files
        return True, "Validation regression check passed", str(dummy_eval_log), str(dummy_orig_log)

class DummySweTask(SweTask): # For SWE-bench specific scenarios (if any in tests)
    def __init__(self, task_id="dummy_swe_task", project_path="dummy_swe_project_path", repo="swe/repo"):
        self.instance = {"repo": repo, "instance_id": task_id}
        self.task_id = task_id # Keep for consistency if used directly
        self.project_path = str(project_path)
        self.repo_name = repo # For get_logs_eval

    def get_instance_id(self) -> str:
        return self.instance["instance_id"]

    def validate(self, patch_content: str) -> tuple[bool, str, str, str]:
        dummy_eval_log = Path(tempfile.gettempdir()) / f"dummy_swe_eval_{Path(tempfile.NamedTemporaryFile(delete=False).name).name}.log"
        dummy_orig_log = Path(tempfile.gettempdir()) / f"dummy_swe_orig_{Path(tempfile.NamedTemporaryFile(delete=False).name).name}.log"
        dummy_eval_log.write_text("SWE EVAL LOG")
        dummy_orig_log.write_text("SWE ORIG LOG")
        return True, "SWE Validation regression check passed", str(dummy_eval_log), str(dummy_orig_log)


class DummyPatchHandle(PatchHandle):
    def __str__(self):
        return "dummy_patch"


@pytest.fixture
def tmp_project(tmp_path):
    # Create a temporary project with one dummy Python file.
    proj = tmp_path / "project"
    proj.mkdir()
    dummy_py = proj / "dummy.py"
    dummy_py.write_text(
        textwrap.dedent(
            """
            def foo():
                pass

            class Bar:
                def baz(self):
                    pass
            """
        )
    )
    return str(proj)


# Monkey-patch apputils functions used in get_changed_methods
@pytest.fixture(autouse=True)
def fake_apputils(monkeypatch):
    # Fake context manager for cd.
    class DummyCD:
        def __init__(self, directory):
            self.directory = directory

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def fake_cd(directory):
        return DummyCD(directory)

    def fake_repo_clean_changes():
        pass

    monkeypatch.setattr("app.api.validation.apputils.cd", fake_cd)
    monkeypatch.setattr(
        "app.api.validation.apputils.repo_clean_changes", fake_repo_clean_changes
    )


# Fake subprocess.run for patch command: simulate a successful patch that changes the file.
@pytest.fixture(autouse=True)
def fake_subprocess_run(monkeypatch):
    def fake_run(cmd, cwd, stdout, stderr, text):
        # In our fake run, we simulate a patch command with returncode 0.
        class DummyCompletedProcess:
            returncode = 0
            stdout = "patch applied"
            stderr = ""

        # Additionally, we modify the file in cwd to simulate a change.
        # Expect that cmd is like: patch -p1 -f -i <diff_file>
        # We assume that in cwd, there is a file "dummy.py".
        target = Path(cwd) / "dummy.py"
        if target.is_file():
            # Replace the content to simulate a change.
            target.write_text("def foo():\n    print('changed')\n")
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)


# --- Tests for validation.py ---


def test_angelic_debugging_message():
    # Test that a non-empty incorrect_locations produces a message.
    incorrect_locations = [("dummy.py", MethodId("Bar", "baz"))]
    msg = angelic_debugging_message(incorrect_locations)
    assert "dummy.py" in msg
    assert "Bar" in msg
    assert "baz" in msg


def test_collect_method_definitions(tmp_path):
    # Create a temporary Python file with a function and a method.
    file_content = textwrap.dedent(
        """
        def alpha():
            pass

        class Beta:
            def gamma(self):
                pass
    """
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(file_content)
    defs = collect_method_definitions(str(file_path))
    method_names = {mid.method_name for mid in defs.keys()}
    assert "alpha" in method_names
    assert "gamma" in method_names


def test_get_method_id(monkeypatch, tmp_path):
    # Monkey-patch method_ranges_in_file to return a fake range.
    def fake_method_ranges(file):
        return {MethodId("Dummy", "foo"): (1, 10)}

    monkeypatch.setattr("app.api.validation.method_ranges_in_file", fake_method_ranges)
    mid = get_method_id("dummy.py", 5)
    assert mid == MethodId("Dummy", "foo")
    mid_none = get_method_id("dummy.py", 20)
    assert mid_none is None


def test_get_changed_methods(tmp_project, tmp_path, monkeypatch):
    # Updated diff content with a valid hunk.
    diff_content = textwrap.dedent(
        """\
        --- a/dummy.py
        +++ b/dummy.py
        @@ -1,2 +1,2 @@
         def foo():
        -    pass
        +    print("changed")
    """
    )
    diff_file = tmp_path / "change.diff"
    diff_file.write_text(diff_content)

    def fake_collect_method_definitions(file):
        if "apply_patch_" in str(file):
            return {MethodId("", "foo"): "def foo():\n    print('changed')"}
        else:
            return {MethodId("", "foo"): "def foo():\n    pass"}

    monkeypatch.setattr(
        validation, "collect_method_definitions", fake_collect_method_definitions
    )

    result = get_changed_methods(str(diff_file), tmp_project)
    assert "dummy.py" in result
    changed_methods = result["dummy.py"]
    assert MethodId("", "foo") in changed_methods


def test_get_developer_patch_file(tmp_path, monkeypatch):
    # Create a temporary file to simulate a developer patch diff.
    task_id = "dummy_task"
    processed_data_lite = tmp_path / "processed_data_lite"
    test_dir = processed_data_lite / "test" / task_id
    test_dir.mkdir(parents=True)
    dev_patch = test_dir / "developer_patch.diff"
    dev_patch.write_text("dummy patch")

    # Monkey-patch Path.__file__ relative resolution in get_developer_patch_file.
    # Instead, we override the parent's with_name to return our tmp directory.
    def fake_with_name(self, name):
        return processed_data_lite

    monkeypatch.setattr(Path, "with_name", fake_with_name)
    path = get_developer_patch_file(task_id)
    assert Path(path).is_file()


def test_perfect_angelic_debug(monkeypatch, tmp_path):
    # Updated diff content with a valid hunk that changes dummy.py.
    diff_content = textwrap.dedent(
        """\
        --- a/dummy.py
        +++ b/dummy.py
        @@ -1,2 +1,2 @@
         def foo():
        -    pass
        +    print("changed")
    """
    )
    diff_file = tmp_path / "change.diff"
    diff_file.write_text(diff_content)

    # Create a dummy developer patch file that is a no-op (empty diff).
    # When the diff is empty, get_changed_methods returns {} for the developer patch.
    task_id = "dummy_task"
    processed_data_lite = tmp_path / "processed_data_lite"
    test_dir = processed_data_lite / "test" / task_id
    test_dir.mkdir(parents=True)
    dev_patch = test_dir / "developer_patch.diff"
    dev_patch.write_text("")  # no changes

    monkeypatch.setattr(
        validation, "get_developer_patch_file", lambda tid: str(dev_patch)
    )

    proj = tmp_path / "project"
    proj.mkdir()
    dummy_py = proj / "dummy.py"
    dummy_py.write_text("def foo():\n    pass\n")

    results = perfect_angelic_debug(task_id, str(diff_file), str(proj))
    diff_only, common, dev_only = results
    # Since the developer patch is empty, we expect the changed method to appear in diff_only.
    assert ("dummy.py", MethodId("", "foo")) in diff_only


def test_evaluate_patch(monkeypatch, tmp_path):
    # Store original config values
    original_enable_validation = config.enable_validation
    original_enable_perfect_angelic = config.enable_perfect_angelic
    original_enable_angelic = config.enable_angelic

    config.enable_validation = True
    config.enable_perfect_angelic = False # Disable angelic by default for this basic test
    config.enable_angelic = False

    task = DummyTask(repo_name="dummy/repo") # Use repo_name consistent with get_logs_eval needs
    patch_handle = DummyPatchHandle()
    patch_content = "dummy patch content"

    output_dir_path = tmp_path / "output"
    output_dir_path.mkdir()

    # Mock dependencies of evaluate_patch
    # 1. task.validate is handled by DummyTask
    # 2. get_logs_eval
    mock_gold_sm = {"test1": TestStatus.FAILED.value, "test2": TestStatus.PASSED.value}
    mock_eval_sm = {"test1": TestStatus.PASSED.value, "test2": TestStatus.PASSED.value} # All pass

    def mock_get_logs_eval_func(repo_name, log_file_path):
        if "EMPTY" in Path(log_file_path).name: # Distinguish orig_log from eval_log
            return mock_gold_sm, None
        return mock_eval_sm, None
    monkeypatch.setattr(validation, "get_logs_eval", mock_get_logs_eval_func)

    # 3. (Optional) perfect_angelic_debug - for this test, assume it's not triggered or returns no issues
    monkeypatch.setattr(validation, "perfect_angelic_debug", lambda tid, df, pp: (set(), set(), set()))

    passed, payload = evaluate_patch(task, patch_handle, patch_content, str(output_dir_path))

    assert passed is True
    assert isinstance(payload, EvaluationPayload)
    assert payload.status == ResolvedStatus.FULL # Because test1 FAILED->PASSED, test2 PASSED->PASSED
    assert "ResolvedStatus: RESOLVED_FULL" in payload.message
    assert payload.details is not None
    if payload.details: # type guard
        assert "test1" in payload.details[FAIL_TO_PASS]["success"]
        assert "test2" in payload.details[PASS_TO_PASS]["success"]

    # Restore original config values
    config.enable_validation = original_enable_validation
    config.enable_perfect_angelic = original_enable_perfect_angelic
    config.enable_angelic = original_enable_angelic


def test_evaluate_patch_p2p_na(monkeypatch, tmp_path):
    config.enable_validation = True
    config.enable_perfect_angelic = False
    config.enable_angelic = False

    task = DummyTask(repo_name="dummy/repo_p2p_na")
    patch_handle = DummyPatchHandle()
    patch_content = "p2p_na_patch"
    output_dir_path = tmp_path / "output_p2p_na"
    output_dir_path.mkdir()

    mock_gold_sm = {"test_f2p": TestStatus.FAILED.value} # Only F2P tests
    mock_eval_sm = {"test_f2p": TestStatus.PASSED.value}

    def mock_get_logs_eval_func(repo_name, log_file_path):
        if "EMPTY" in Path(log_file_path).name:
            return mock_gold_sm, None
        return mock_eval_sm, None
    monkeypatch.setattr(validation, "get_logs_eval", mock_get_logs_eval_func)
    monkeypatch.setattr(validation, "perfect_angelic_debug", lambda tid, df, pp: (set(), set(), set()))

    passed, payload = evaluate_patch(task, patch_handle, patch_content, str(output_dir_path))

    assert passed is True
    assert isinstance(payload, EvaluationPayload)
    assert payload.status == ResolvedStatus.FULL_P2P_NA
    assert "ResolvedStatus: RESOLVED_FULL_P2P_NA" in payload.message


def test_evaluate_patch_no_missing_results(monkeypatch, tmp_path):
    config.enable_validation = True
    task = DummyTask(repo_name="dummy/repo_missing")
    patch_handle = DummyPatchHandle()
    output_dir_path = tmp_path / "output_missing"
    output_dir_path.mkdir()

    # Gold: test_f2p should pass, test_p2p should pass
    mock_gold_sm = {"test_f2p": TestStatus.FAILED.value, "test_p2p": TestStatus.PASSED.value}
    # Eval: test_f2p is missing, test_p2p passes
    mock_eval_sm = {"test_p2p": TestStatus.PASSED.value}

    def mock_get_logs_eval_func(repo_name, log_file_path):
        if "EMPTY" in Path(log_file_path).name: return mock_gold_sm, None
        return mock_eval_sm, None # overall_status is None for eval run
    monkeypatch.setattr(validation, "get_logs_eval", mock_get_logs_eval_func)
    monkeypatch.setattr(validation, "perfect_angelic_debug", lambda tid, df, pp: (set(), set(), set()))

    passed, payload = evaluate_patch(task, patch_handle, "patch_missing_results", str(output_dir_path))

    assert passed is False # Because NO_MISSING_RESULTS is not a "pass" state
    assert isinstance(payload, EvaluationPayload)
    assert payload.status == ResolvedStatus.NO_MISSING_RESULTS
    assert "ResolvedStatus: RESOLVED_NO_MISSING_RESULTS" in payload.message
    assert "test_f2p" in payload.details[FAIL_TO_PASS]["missing"]


def test_evaluate_patch_framework_error_in_eval(monkeypatch, tmp_path):
    config.enable_validation = True
    task = DummyTask(repo_name="dummy/repo_framework_error")
    patch_handle = DummyPatchHandle()
    output_dir_path = tmp_path / "output_framework_error"
    output_dir_path.mkdir()

    mock_gold_sm = {"test_f2p": TestStatus.FAILED.value}
    # Eval: Simulates a framework error during the evaluation run
    mock_eval_sm = {} # No results due to framework error

    def mock_get_logs_eval_func(repo_name, log_file_path):
        if "EMPTY" in Path(log_file_path).name: return mock_gold_sm, None
        return mock_eval_sm, TestStatus.FRAMEWORK_ERROR # Eval run had a framework error
    monkeypatch.setattr(validation, "get_logs_eval", mock_get_logs_eval_func)
    monkeypatch.setattr(validation, "perfect_angelic_debug", lambda tid, df, pp: (set(), set(), set()))

    passed, payload = evaluate_patch(task, patch_handle, "patch_framework_error", str(output_dir_path))

    assert passed is False
    assert isinstance(payload, EvaluationPayload)
    assert payload.status == ResolvedStatus.NO_FRAMEWORK_ERROR
    assert "ResolvedStatus: RESOLVED_NO_FRAMEWORK_ERROR" in payload.message
