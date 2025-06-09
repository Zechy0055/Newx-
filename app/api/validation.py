"""
Perform validation of a patch, on a given task instance.
"""

import ast
import itertools
import json
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from subprocess import PIPE, STDOUT
from tempfile import NamedTemporaryFile

from loguru import logger
from unidiff import PatchSet

from app import config
from app import utils as apputils
from app.agents.agent_write_patch import PatchHandle
from app.analysis.sbfl import method_ranges_in_file
from app.data_structures import MethodId, EvaluationPayload # Added EvaluationPayload
from app.task import SweTask, Task
# Imports for new evaluation logic
from app.api.eval_helper import (
    get_logs_eval,
    get_eval_report,
    get_resolution_status,
    ResolvedStatus,
    TestStatus,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    FAIL_TO_FAIL,
    PASS_TO_FAIL,
)


def perfect_angelic_debug(
    task_id: str, diff_file: str, project_path: str
) -> tuple[
    set[tuple[str, MethodId]], set[tuple[str, MethodId]], set[tuple[str, MethodId]]
]:
    """Do perfect angelic debugging and return a list of incorrect fix locations.

    Args:
        task_id: the task id, used to find developer patch
        diff_file: path of diff file

    Returns:
        A list of (filename, MethodId) that should not have been changed by diff_file
    """
    return compare_fix_locations(
        diff_file, get_developer_patch_file(task_id), project_path
    )


def compare_fix_locations(
    diff_file: str, dev_diff_file: str, project_path: str
) -> tuple[
    set[tuple[str, MethodId]], set[tuple[str, MethodId]], set[tuple[str, MethodId]]
]:
    """Compare the changed methods in two diff files

    Args:
        diff_file: path to diff file
        dev_diff_file: path to a "correct" diff file

    Returns:
        list of (filename, MethodId) that are changed in diff_file but not in dev_diff_file
    """
    methods_map = get_changed_methods(diff_file, project_path)
    dev_methods_map = get_changed_methods(dev_diff_file, project_path)

    methods_set = set(
        itertools.chain.from_iterable(
            [(k, method_id) for method_id in v] for k, v in methods_map.items()
        )
    )
    dev_methods_set = set(
        itertools.chain.from_iterable(
            [(k, method_id) for method_id in v] for k, v in dev_methods_map.items()
        )
    )

    return (
        methods_set - dev_methods_set,
        methods_set & dev_methods_set,
        dev_methods_set - methods_set,
    )


def get_developer_patch_file(task_id: str) -> str:
    processed_data_lite = Path(__file__).parent.parent.with_name("processed_data_lite")
    dev_patch_file = Path(
        processed_data_lite, "test", task_id, "developer_patch.diff"
    ).resolve()
    if not dev_patch_file.is_file():
        raise RuntimeError(f"Failed to find developer patch at {dev_patch_file!s}")
    return str(dev_patch_file)


def get_method_id(file: str, line: int) -> MethodId | None:
    ranges = method_ranges_in_file(file)
    for method_id, (lower, upper) in ranges.items():
        if lower <= line <= upper:
            return method_id
    return None


def get_changed_methods(
    diff_file: str, project_path: str = ""
) -> dict[str, set[MethodId]]:
    with apputils.cd(project_path):
        apputils.repo_clean_changes()

    with open(diff_file) as f:
        patch_content = f.read()

    changed_files = []

    patch = PatchSet(patch_content)
    for file in patch:
        file_name = file.source_file.removeprefix("a/").removeprefix("b/")
        changed_files.append(file_name)

    orig_definitions: dict[tuple[str, MethodId], str] = {}
    for file in changed_files:
        def_map = collect_method_definitions(Path(project_path, file))

        for method_id, definition in def_map.items():
            orig_definitions[(file, method_id)] = definition

    temp_dir = tempfile.mkdtemp(dir="/tmp", prefix="apply_patch_")
    for file in changed_files:
        copy_path = Path(temp_dir, file)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(project_path, file), copy_path)
    patch_cmd = f"patch -p1 -f -i {diff_file}"
    cp = subprocess.run(
        shlex.split(patch_cmd), cwd=temp_dir, stdout=PIPE, stderr=STDOUT, text=True
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"Patch command in directory {temp_dir} exit with {cp.returncode}: {patch_cmd}\n {cp.stdout} {cp.stderr}"
        )

    new_definitions: dict[tuple[str, MethodId], str] = {}
    for file in changed_files:
        def_map = collect_method_definitions(Path(temp_dir, file))

        for method_id, definition in def_map.items():
            new_definitions[(file, method_id)] = definition

    shutil.rmtree(temp_dir)

    result = {}
    for key, definition in orig_definitions.items():
        if new_definitions.get(key, "") != definition:
            file, method_id = key
            result[file] = result.get(file, set()) | {method_id}

    return result


def collect_method_definitions(file: str | PathLike) -> dict[MethodId, str]:
    if not str(file).endswith(".py"):
        return {}

    collector = MethodDefCollector()

    source = Path(file).read_text()
    tree = ast.parse(source, file)

    collector.visit(tree)
    return collector.def_map


class MethodDefCollector(ast.NodeVisitor):
    def __init__(self):
        self.def_map: dict[MethodId, str] = {}
        self.class_name = ""

    def calc_method_id(self, method_name: str) -> MethodId:
        return MethodId(self.class_name, method_name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_name = node.name
        super().generic_visit(node)
        self.class_name = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        method_id = self.calc_method_id(node.name)
        self.def_map[method_id] = ast.unparse(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        method_id = self.calc_method_id(node.name)
        self.def_map[method_id] = ast.unparse(node)


def evaluate_patch(
    task: Task, patch_handle: PatchHandle, patch_content: str, output_dir: str
) -> tuple[bool, EvaluationPayload]: # Changed return type
    # Renamed for clarity and to reflect it returns log paths
    task_id = task.get_instance_id() # Get task_id for logging context

    # Renamed for clarity and to reflect it returns log paths
    def validate_and_move_logs(patch_content: str) -> tuple[bool, str, str, str]:
        try:
            # patch_is_correct here means "no new test failures were introduced by the patch"
            patch_is_correct, err_message, log_file_path, orig_log_file_path = task.validate(
                patch_content
            )
        except TimeoutError as e:
            logger.bind(task_id=task_id, patch_handle=patch_handle).warning("Validation timed out", exc_info=True)
            return False, f"TimeoutError occurred: {e}", "", "" # Return empty paths for logs

        eval_log_dest = Path(output_dir, f"run_test_suite_{patch_handle}.log")
        orig_log_dest = Path(output_dir, "run_test_suite_EMPTY.log")

        if Path(log_file_path).exists():
            shutil.move(log_file_path, eval_log_dest)
        else:
            logger.bind(task_id=task_id, patch_handle=patch_handle).warning(f"Evaluation log file {log_file_path} not found. Creating placeholder.")
            eval_log_dest.touch(exist_ok=True)

        if Path(orig_log_file_path).exists():
            shutil.move(orig_log_file_path, orig_log_dest)
        else:
            logger.bind(task_id=task_id).warning(f"Original log file {orig_log_file_path} not found. Creating placeholder.") # No patch_handle for original log
            orig_log_dest.touch(exist_ok=True)

        Path(output_dir, f"regression_{patch_handle}.json").write_text(
            json.dumps({"no_additional_failure": patch_is_correct}, indent=4)
        )

        # err_message is about new failures. Detailed resolution status comes from eval_helper.
        return patch_is_correct, err_message, str(eval_log_dest), str(orig_log_dest)

    def perfect_angelic_debugging(task: Task, diff_file: str) -> tuple[bool, str]:
        if not isinstance(task, SweTask):
            raise NotImplementedError(
                f"Angelic debugging not implemented for {type(task).__name__}"
            )

        # FIXME: perfect_angelic_debug has changed since this is written. Fix this
        # if angelic debugging is ever used again.
        incorrect_locations = perfect_angelic_debug(
            task.task_id, diff_file, task.project_path
        )

        msg = angelic_debugging_message(incorrect_locations[0])

        return not incorrect_locations, msg

    def angelic_debugging(task: Task, diff_file: str) -> tuple[bool, str]:
        raise NotImplementedError("Angelic debugging has not been integrated")

    # Determine repo_name, needed for log parsing
    repo_name = ""
    if isinstance(task, SweTask): # Ensure task is SweTask to access instance details
        repo_name = task.instance["repo"]
    else:
        parts = task.get_instance_id().split("__")
        if len(parts) > 1 and '/' in parts[0]:
            repo_name = parts[0]

    if not repo_name:
        # Use task_id if available, otherwise task.get_instance_id() if task_id variable isn't set yet (it should be by now)
        current_task_id = task_id if 'task_id' in locals() else task.get_instance_id()
        logger.bind(task_id=current_task_id).error(f"Could not determine repo_name. Log parsing will likely fail.")
        # Return a default error payload
        return False, EvaluationPayload(
            status=ResolvedStatus.NO_FRAMEWORK_ERROR,
            message="Configuration error: Could not determine repository name for test log parsing.",
            details={"error": "Missing repo_name for task " + current_task_id}
        )

    if not config.enable_validation:
        return True, EvaluationPayload(
            status=ResolvedStatus.FULL, # Assuming success if validation disabled
            message="Validation disabled. Skipped all checks.",
            details={}
        )

    # Call the renamed inner function
    regression_check_passed, regression_err_msg, eval_log_path, orig_log_path = validate_and_move_logs(patch_content)

    # Initialize eval_report and final_status with default/error values
    eval_report = {}
    final_status = ResolvedStatus.NO_FRAMEWORK_ERROR # Default to an error status

    # Parse original (gold) logs
    gold_sm, gold_overall_status = {}, TestStatus.FRAMEWORK_ERROR # Default if log missing/empty
    if Path(orig_log_path).exists() and Path(orig_log_path).stat().st_size > 0:
        gold_sm, gold_overall_status = get_logs_eval(repo_name, orig_log_path)
    else:
        logger.bind(task_id=task_id, patch_handle=patch_handle).warning(f"Original log file {orig_log_path} is missing or empty.")
        return False, EvaluationPayload(
            status=ResolvedStatus.NO_FRAMEWORK_ERROR,
            message=f"Original test log file missing or empty ({orig_log_path}). Cannot determine full evaluation status.",
            details={"error": "Missing original log file", "path": orig_log_path}
        )

    # Parse evaluation (patched) logs
    eval_sm, eval_overall_status = {}, TestStatus.FRAMEWORK_ERROR # Default if log missing/empty
    if Path(eval_log_path).exists() and Path(eval_log_path).stat().st_size > 0:
        eval_sm, eval_overall_status = get_logs_eval(repo_name, eval_log_path)
    else:
        logger.bind(task_id=task_id, patch_handle=patch_handle).warning(f"Evaluation log file {eval_log_path} is missing or empty.")
        # eval_overall_status remains FRAMEWORK_ERROR from its initialization

    # Construct placeholder_gold_results (ideally this comes from benchmark definition)
    gold_f2p_tests = [test for test, status in gold_sm.items() if status not in [TestStatus.PASSED.value, TestStatus.SKIPPED.value]]
    gold_p2p_tests = [test for test, status in gold_sm.items() if status == TestStatus.PASSED.value]
    placeholder_gold_results = {
        FAIL_TO_PASS: gold_f2p_tests, PASS_TO_PASS: gold_p2p_tests,
        FAIL_TO_FAIL: [], PASS_TO_FAIL: [] # Cannot infer these without more info
    }

    if gold_overall_status and gold_overall_status not in [TestStatus.PASSED, TestStatus.SKIPPED]: # Allow SKIPPED if all tests were skipped
         logger.bind(task_id=task_id).warning(f"Gold standard test run had issues: {gold_overall_status}. Evaluation may be affected.")
    if not gold_sm and gold_overall_status != TestStatus.FRAMEWORK_ERROR: # No tests in gold log, but no framework error
        logger.bind(task_id=task_id).warning(f"Gold standard log contains no test results. Evaluation may be affected.")


    eval_report = get_eval_report(eval_sm, placeholder_gold_results, eval_overall_status)
    final_status = get_resolution_status(eval_report, eval_overall_status)

    # Build the message for EvaluationPayload
    current_payload_message = f"ResolvedStatus: {final_status.value}."
    if not regression_check_passed: # From task.validate()
        current_payload_message += f" Regression check failed: {regression_err_msg}."
    if eval_overall_status == TestStatus.FRAMEWORK_ERROR and Path(eval_log_path).exists() and Path(eval_log_path).stat().st_size == 0:
        current_payload_message += " Evaluation log was empty, indicating a possible test framework crash."
    elif eval_overall_status == TestStatus.EXECUTION_TIMEOUT:
         current_payload_message += " Test execution timed out during evaluation run."


    # Angelic debugging (if enabled and validation failed in some sense)
    # For now, we assume angelic debugging info is just appended to the message if it runs.
    angelic_msg = ""
    # We might only run angelic if final_status is not FULL/FULL_P2P_NA and regression_check_passed is False
    run_angelic = config.enable_perfect_angelic or config.enable_angelic
    # Condition to run angelic: if the detailed status shows issues OR if regression check failed.
    detailed_status_indicates_failure = final_status not in [ResolvedStatus.FULL, ResolvedStatus.FULL_P2P_NA]

    if run_angelic and (detailed_status_indicates_failure or not regression_check_passed) :
        with NamedTemporaryFile(mode="w", buffering=0, suffix=".diff", delete=False) as f: # write patch to temp file
            f.write(patch_content)
            tmp_patch_file_name = f.name
        # Ensure file is closed before angelic debugging reads it

        _angelic_passed, _angelic_msg_detail = True, ""
        if config.enable_perfect_angelic:
            if not isinstance(task, SweTask): # perfect_angelic_debug expects SweTask
                 logger.bind(task_id=task_id, patch_handle=patch_handle).warning("Perfect angelic debugging skipped: task is not an SweTask instance.")
            else:
                try:
                    # perfect_angelic_debug returns tuple of sets: (incorrectly_changed, correctly_changed, missed_dev_changes)
                    incorrect_locs, _, _ = perfect_angelic_debug(task.task_id, tmp_patch_file_name, task.project_path)
                    _angelic_msg_detail = angelic_debugging_message(incorrect_locs)
                    _angelic_passed = not bool(incorrect_locs)
                except Exception as e:
                    logger.bind(task_id=task_id, patch_handle=patch_handle).error(f"Error during perfect angelic debugging: {e}", exc_info=True)
                    _angelic_msg_detail = "Error during perfect angelic debugging."
        elif config.enable_angelic: # Placeholder for general angelic debugging
             _angelic_passed, _angelic_msg_detail = False, "Standard angelic debugging not fully implemented here."

        if not _angelic_passed and _angelic_msg_detail:
            angelic_msg = f" Angelic Debugging: {_angelic_msg_detail}"

        Path(tmp_patch_file_name).unlink() # Clean up temp file

    if angelic_msg:
        current_payload_message += angelic_msg

    # Determine overall pass/fail for the boolean return based on ResolvedStatus
    # Typically, only FULL or FULL_P2P_NA is a "pass".
    # Also consider regression_check_passed: if new tests fail, it's not a pass.
    overall_eval_passed = final_status in [ResolvedStatus.FULL, ResolvedStatus.FULL_P2P_NA] and regression_check_passed

    return overall_eval_passed, EvaluationPayload(
        status=final_status,
        message=current_payload_message,
        details=eval_report
    )


def angelic_debugging_message(
    incorrect_locations: Iterable[tuple[str, MethodId]],
) -> str:
    msg = []

    if incorrect_locations:
        msg.append("The following methods should not have been changed:")
        msg.extend(
            f"    {filename}: {method_id!s}"
            for filename, method_id in incorrect_locations
        )

    return "\n".join(msg)
