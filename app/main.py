"""
The main driver.
"""

import json
import logging
import platform
import shutil
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
from itertools import chain
from multiprocessing import set_start_method
from os import PathLike
from os.path import abspath
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import config, inference, log, result_analysis, task_counter
from app import utils as apputils
from app.manage import ProjectApiManager
from app.model import common
from app.model.register import register_all_models
from app.post_process import (
    extract_organize_and_form_input,
    get_final_patch_path,
    organize_and_form_input,
    reextract_organize_and_form_inputs,
)
from app.raw_tasks import RawGithubTask, RawLocalTask, RawSweTask, RawTask
from app.task import SweTask, Task

# Imports for FastAPI endpoint
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict

# FastAPI app instance
# This should ideally be in a dedicated server setup file, but adding here as per prompt.
# If an 'app' instance already exists for other API routes, use that.
# For now, assume we are defining it here.
app = FastAPI()

# Pydantic model for frontend log payload
class FrontendLogPayload(BaseModel):
    level: str # DEBUG, INFO, WARNING, ERROR
    message: str
    component: Optional[str] = None
    function: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    frontend_timestamp: Optional[str] = None

@app.post("/api/log_frontend_event")
async def log_frontend_event(payload: FrontendLogPayload, request: Request):
    client_host = request.client.host if request.client else "unknown"
    log_context = {
        "client_host": client_host,
        "frontend_component": payload.component,
        "frontend_function": payload.function,
        "frontend_timestamp": payload.frontend_timestamp,
        **(payload.context or {}) # Spread any additional context from frontend
    }

    # Use a bound logger for cleaner log structure
    bound_logger = logger.bind(**{k: v for k, v in log_context.items() if v is not None})

    level_upper = payload.level.upper()
    if level_upper == "DEBUG":
        bound_logger.debug("[FRONTEND] " + payload.message)
    elif level_upper == "INFO":
        bound_logger.info("[FRONTEND] " + payload.message)
    elif level_upper == "WARNING":
        bound_logger.warning("[FRONTEND] " + payload.message)
    elif level_upper == "ERROR":
        bound_logger.error("[FRONTEND] " + payload.message)
    else:
        bound_logger.warning(f"[FRONTEND] Unknown log level '{payload.level}': {payload.message}") # Log as warning if level is unknown

    return {"status": "Log received"}


def main():
    # NOTE: The FastAPI app instance 'app' is defined above.
    # To run this FastAPI app, a Uvicorn server would typically be used, e.g.:
    # uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
    # The existing if __name__ == "__main__": block below handles CLI execution.
    # The two (CLI and Server) would usually be run as separate processes or managed by a process manager.

    register_all_models()
    parser = ArgumentParser()

    subparser_dest_attr_name = "command"
    subparsers = parser.add_subparsers(dest=subparser_dest_attr_name)

    swe_parser = subparsers.add_parser(
        "swe-bench", help="Run one or multiple swe-bench tasks"
    )
    set_swe_parser_args(swe_parser)

    github_parser = subparsers.add_parser(
        "github-issue",
        help="Run an online github issue",
    )
    set_github_parser_args(github_parser)

    local_parser = subparsers.add_parser("local-issue", help="Run a local issue.")
    set_local_parser_args(local_parser)

    extract_patches_parser = subparsers.add_parser(
        "extract-patches", help="Only extract patches from the raw results dir"
    )
    extract_patches_parser.add_argument("experiment_dir", type=str)
    add_task_related_args(extract_patches_parser)

    re_extract_patches_parser = subparsers.add_parser(
        "re-extract-patches",
        help=(
            "same as extract-patches, except that individual dirs"
            " are moved out of their categories first"
        ),
    )
    re_extract_patches_parser.add_argument("experiment_dir", type=str)
    add_task_related_args(re_extract_patches_parser)

    args = parser.parse_args()

    ## common options
    config.output_dir = args.output_dir
    if config.output_dir is not None:
        config.output_dir = abspath(config.output_dir)
    num_processes: int = int(args.num_processes)
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout

    # model related
    config.models = list(chain.from_iterable(args.model))
    if not config.models:
        config.models.append("gpt-3.5-turbo-0125")
    common.set_model(config.models[0])

    # FIXME: make temperature part of the Model class
    common.MODEL_TEMP = args.model_temperature
    logger.info("Set model temperature to: {}", common.MODEL_TEMP)

    # acr related
    config.conv_round_limit = args.conv_round_limit
    config.enable_sbfl = args.enable_sbfl
    config.enable_validation = args.enable_validation
    config.enable_angelic = args.enable_angelic
    config.enable_perfect_angelic = args.enable_perfect_angelic
    config.only_save_sbfl_result = args.save_sbfl_result
    config.only_reproduce = args.reproduce

    subcommand = getattr(args, subparser_dest_attr_name)
    if subcommand == "swe-bench":
        if args.result_analysis:
            logger.info("Starting result analysis for directory: {}", config.output_dir)
            result_analysis.analyze(config.output_dir)
            exit(0)

        tasks = make_swe_tasks(
            args.task, args.task_list_file, args.setup_map, args.tasks_map
        )

        config.only_eval_reproducer = args.eval_reproducer

        config.reproduce_and_review = args.reproduce_and_review

        groups = group_swe_tasks_by_env(tasks)
        run_task_groups(groups, num_processes, organize_output=True)
    elif subcommand == "github-issue":
        setup_dir = args.setup_dir
        if setup_dir is not None:
            setup_dir = abspath(setup_dir)

        task = RawGithubTask(
            args.task_id,
            args.clone_link,
            args.commit_hash,
            args.issue_link,
            setup_dir,
        )
        groups = {"github": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "local-issue":
        local_repo = args.local_repo
        if local_repo is not None:
            local_repo = abspath(local_repo)
        issue_file = args.issue_file
        if issue_file is not None:
            issue_file = abspath(issue_file)
        task = RawLocalTask(
            args.task_id,
            local_repo,
            issue_file,
        )
        groups = {"local": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "extract-patches":
        logger.info("Extracting patches from: {}", args.experiment_dir)
        extract_organize_and_form_input(args.experiment_dir)
    elif subcommand == "re-extract-patches":
        logger.info("Re-extracting patches from: {}", args.experiment_dir)
        reextract_organize_and_form_inputs(args.experiment_dir)


def set_swe_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)

    parser.add_argument(
        "--setup-map",
        type=str,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks-map",
        type=str,
        help="Path to json file that contains the tasks information.",
    )
    parser.add_argument(
        "--task-list-file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument("--task", type=str, help="Task id to be run.")
    parser.add_argument(
        "--eval-reproducer",
        action="store_true",
        default=False,
        help="Only check if reproducer.py is a correct test",
    )
    parser.add_argument(
        "--reproduce-and-review",
        action="store_true",
        default=False,
        help="Experimental: for swe-bench tasks, reproduce and review the generated patch",
    )
    parser.add_argument(
        "--result-analysis",
        action="store_true",
        default=False,
        help="Perform some analysis on the experiment result and exit.",
    )


def set_github_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id",
        type=str,
        help="Assign an id to the current fresh issue task.",
    )
    parser.add_argument(
        "--clone-link",
        type=str,
        help="The link to the repository to clone.",
    )
    parser.add_argument("--commit-hash", type=str, help="The commit hash to checkout.")
    parser.add_argument("--issue-link", type=str, help="The link to the issue.")
    parser.add_argument(
        "--setup-dir",
        type=str,
        help="The directory where repositories should be cloned to.",
    )


def set_local_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current local issue task."
    )
    parser.add_argument(
        "--local-repo", type=str, help="Path to a local copy of the target repo."
    )
    parser.add_argument("--issue-file", type=str, help="Path to a local issue file.")


def add_task_related_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory that stores the run results.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Do not print most messages to stdout.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(common.MODEL_HUB.keys()),
        nargs="+",
        action="append",
        help="The model to use. Currently only OpenAI models are supported.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.0,
        help="The model temperature to use, for OpenAI models.",
    )
    parser.add_argument(
        "--conv-round-limit",
        type=int,
        default=15,
        help="Conversation round limit for the main agent.",
    )
    parser.add_argument(
        "--enable-layered",
        action="store_true",
        default=True,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--enable-sbfl", action="store_true", default=False, help="Enable SBFL."
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable validation in our workflow.",
    )
    parser.add_argument(
        "--enable-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable angelic debugging",
    )
    parser.add_argument(
        "--enable-perfect-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable perfect angelic debugging; overrides --enable-angelic",
    )
    parser.add_argument(
        "--save-sbfl-result",
        action="store_true",
        default=False,
        help="Special mode to only save SBFL results for future runs.",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        default=False,
        help="Special mode to only generate reproducer tests",
    )
    parser.add_argument(
        "--num-processes",
        type=str,
        default=1,
        help="Number of processes to run the tasks in parallel.",
    )


def make_swe_tasks(
    task_id: str | None,
    task_list_file: str | None,
    setup_map_file: str,
    tasks_map_file: str,
) -> list[RawSweTask]:
    if task_id is not None and task_list_file is not None:
        logger.error("Cannot specify both --task and --task-list-file simultaneously.")
        raise ValueError("Cannot specify both task and task-list.")

    all_task_ids = []
    if task_list_file is not None:
        all_task_ids = parse_task_list_file(task_list_file)
        logger.info("Loaded {} task IDs from task list file: {}", len(all_task_ids), task_list_file)
    if task_id is not None:
        all_task_ids = [task_id]
        logger.info("Running for single task ID: {}", task_id)
    if len(all_task_ids) == 0:
        logger.error("No task IDs provided to run.")
        raise ValueError("No task ids to run.")

    with open(setup_map_file) as f:
        setup_map = json.load(f)
    with open(tasks_map_file) as f:
        tasks_map = json.load(f)

    # Check if all task ids are in the setup and tasks map
    # This allows failing safely if some tasks are not set up properly
    missing_task_ids = [
        x for x in all_task_ids if not (x in setup_map and x in tasks_map)
    ]
    if missing_task_ids:
        for task_id_missing in sorted(missing_task_ids): # Renamed task_id to avoid conflict
            logger.warning("Skipping task {} which was not found in setup_map or tasks_map.", task_id_missing)
        all_task_ids = [tid for tid in all_task_ids if tid not in missing_task_ids] # Corrected filter

    all_task_ids = sorted(all_task_ids)
    logger.debug("Final list of task IDs to process: {}", all_task_ids)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    for task_id in all_task_ids:
        setup_info = setup_map[task_id]
        task_info = tasks_map[task_id]
        task = RawSweTask(task_id, setup_info, task_info)
        all_tasks.append(task)
    return all_tasks


def parse_task_list_file(task_list_file: str) -> list[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file) as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


def group_swe_tasks_by_env(tasks: list[RawSweTask]) -> dict[str, list[RawSweTask]]:
    groups = {}
    for task in tasks:
        key = task.setup_info["env_name"]
        if key not in groups:
            groups[key] = []
        groups[key].append(task)
    return groups


def run_task_groups(
    task_groups: Mapping[str, Sequence[RawTask]],
    num_processes: int,
    organize_output: bool = False,
):
    """
    Main entry for running tasks.
    """
    all_tasks = list(chain.from_iterable(task_groups.values()))
    num_tasks = len(all_tasks)

    task_counter.init_total_num_tasks(num_tasks)

    logger.info("Total number of tasks: {}", num_tasks)
    logger.info("Total number of processes: {}", num_processes)
    logger.info("Task group info: (number of groups: {})", len(task_groups))
    for key, tasks_in_group in task_groups.items(): # Renamed tasks to avoid conflict
        logger.info("\tGroup '{}': {} tasks", key, len(tasks_in_group))

    # single process mode
    if num_processes == 1:
        logger.info("Running in single process mode.")
        run_tasks_serial(all_tasks)
        logger.info("Finished all tasks sequentially.")
    else:
        run_task_groups_parallel(task_groups, num_processes)

    if config.only_save_sbfl_result:
        logger.info("Only saving SBFL results. Exiting.")
        return

    if organize_output:
        logger.info("Post-processing completed experiment results.")
        swe_input_file = organize_and_form_input(config.output_dir)
        logger.info("SWE-Bench input file created: {}", swe_input_file)


def run_tasks_serial(tasks: list[RawTask]) -> None:
    for task in tasks:
        run_task_in_subprocess(task)


def run_task_groups_parallel(
    task_groups: Mapping[str, Sequence[RawTask]],
    num_processes: int,
):
    num_task_groups = len(task_groups)
    task_counter.init_total_num_task_groups(num_task_groups)
    num_processes = min(num_processes, num_task_groups)

    task_group_ids_items = sorted(
        task_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    logger.info("Sorted task groups for parallel execution: {}", [x[0] for x in task_group_ids_items])
    try:
        # Use ProcessPoolExecutor instead of multiprocessing.Pool,
        # to support nested sub-processing

        group_ids, group_tasks = zip(*task_group_ids_items)
        with ProcessPoolExecutor(num_processes) as executor:
            executor.map(run_task_group, group_ids, group_tasks)
    finally:
        logger.info("Finishing all tasks in the pool.")


def run_task_group(task_group_id: str, task_group_items: list[RawTask]) -> None:
    """
    Run all tasks in a task group sequentially.
    Main entry to parallel processing.
    """
    bound_logger = logger.bind(task_group_id=task_group_id, process_type="task_group_runner")
    bound_logger.info("Starting process for task group. Number of tasks: {}.", len(task_group_items))

    for task_item in task_group_items: # Renamed task to avoid conflict
        # within a group, the runs are always sequential
        run_task_in_subprocess(task_item) # This will eventually call run_raw_task which logs with task_id
        bound_logger.info(task_counter.incre_task_return_msg()) # This message is dynamic

    bound_logger.info("{} Finished task group.", task_counter.incre_task_group_return_msg())


def run_task_in_subprocess(task: RawTask) -> None:
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(run_raw_task, task)


def run_raw_task(task: RawTask) -> bool:
    """
    High-level entry for running one task.

    Args:
        - task: The Task instance to run.

    Returns:
        Whether the task completed successfully.
    """
    if config.only_eval_reproducer:
        assert isinstance(task, RawSweTask)
        evaluate_swe_issue_reproducers(task)
        return True

    task_id = task.task_id
    start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_output_dir = pjoin(config.output_dir, f"{task_id}_{start_time_s}")
    apputils.create_dir_if_not_exists(task_output_dir)

    task.dump_meta_data(task_output_dir)

    log.log_and_always_print(
        f"============= Running task {task_id} =============",
    )

    run_ok = False

    try:
        run_ok = do_inference(task.to_task(), task_output_dir)

        if run_ok:
            run_status_message = f"Task {task_id} completed successfully."
        else:
            run_status_message = f"Task {task_id} failed without exception."
    except Exception as e:
        # logger.exception(e) is good, but binding task_id provides more context
        logger.bind(task_id=task_id).exception("Task failed with an unhandled exception.")
        # The original exception string is included by logger.exception
        run_status_message = f"Task {task_id} failed with exception: {type(e).__name__}."

    # log_and_always_print already uses logger.info, so the message will go to JSON.
    # The console output has its own timestamp.
    log.log_and_always_print(run_status_message)

    final_patch_path = get_final_patch_path(task_output_dir)
    if final_patch_path is not None:
        log.log_and_always_print(
            f"Please find the generated patch at: {final_patch_path}"
        )
        if isinstance(task, RawSweTask):
            log.log_and_always_print(
                "[SWE-bench mode] Note that the patch may be move to other paths in SWE-bench mode. "
                "Please check the SWE-bench input file containing generated patches for all tasks."
            )
    else:
        log.log_and_always_print("No patch generated. You can try running ACR again.")

    return run_ok


def evaluate_swe_issue_reproducers(raw_task: RawSweTask) -> None:
    swe_task = raw_task.to_task()
    swe_task.setup_project()

    reproducer_files = glob(
        pjoin(
            config.output_dir, "**", f"*{swe_task.task_id}*", "**", "reproducer_*.py"
        ),
        recursive=True,
    )
    for reproducer_file in reproducer_files:
        evaluate_swe_issue_reproducer(swe_task, reproducer_file)


def evaluate_swe_issue_reproducer(
    task: SweTask, reproducer_file: str | PathLike
) -> None:
    reproducer_file = Path(reproducer_file)

    individual_expr_dir = reproducer_file.parent

    developer_patch_file = individual_expr_dir.joinpath("developer_patch.diff")
    if not developer_patch_file.exists():
        individual_expr_dir = individual_expr_dir.parent
        developer_patch_file = individual_expr_dir.joinpath("developer_patch.diff")
    assert developer_patch_file.exists()

    report_dir = individual_expr_dir.joinpath(
        "reproducer-eval",
        *reproducer_file.relative_to(individual_expr_dir).with_suffix("").parts,
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    task.evaluate_reproducer(reproducer_file, developer_patch_file, report_dir)


def do_inference(python_task: Task, task_output_dir: str) -> bool:
    apputils.create_dir_if_not_exists(task_output_dir)

    log_file_name = "info.log"

    logger.add(
        pjoin(task_output_dir, log_file_name),
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
            " | <level>{message}</level>"
        ),
    )
    # Add new sink for JSON logging
    logger.add(
        pjoin(task_output_dir, "json_info.log"),
        level="DEBUG",
        serialize=True, # Enable JSON output
    )

    start_time = datetime.now()

    python_task.setup_project()

    try:
        if config.only_save_sbfl_result:
            api_manager = ProjectApiManager(python_task, task_output_dir)
            _, _, run_ok = api_manager.fault_localization()

        elif config.only_reproduce:
            api_manager = ProjectApiManager(python_task, task_output_dir)
            _, _, run_ok = api_manager.reproduce()

        else:
            # normal mode - actually running the task

            try:
                run_ok = inference.run_one_task(
                    python_task, task_output_dir, config.models # run_one_task in inference.py logs with task_id
                )

            except common.ClaudeContentPolicyViolation as e:
                logger.bind(task_id=python_task.get_instance_id()).error(f"Content policy violation from Claude: {e}", exc_info=True)
                # log_and_always_print already uses logger.info
                log.log_and_always_print(
                    "Content policy violation. Trying with backup model if configured."
                )

                if not config.backup_model:
                    logger.bind(task_id=python_task.get_instance_id()).error("No backup model configured. Cannot retry.")
                    run_ok = False # Ensure run_ok is False
                    # Potentially re-raise or handle as a definitive failure for this task
                else:
                    # retry with backup model
                    python_task.setup_project() # This might log internally

                    # remove everything other than the info.log file, and
                    # also some meta data file dumped by RawTask
                    log.log_and_always_print( # This logs via logger.info
                        "Removing intermediate files except logs and meta files before retrying with backup model."
                    )
                    logger.bind(task_id=python_task.get_instance_id()).debug("Starting file cleanup for retry.")
                    for f in Path(task_output_dir).iterdir():
                    if f.is_file() and f.name not in [
                        log_file_name,
                        "meta.json",
                        "problem_statement.txt",
                        "developer_patch.diff",
                    ]:
                        f.unlink()
                    if f.is_dir():
                        shutil.rmtree(str(f))

                run_ok = inference.run_one_task(
                    python_task, task_output_dir, config.backup_model
                )

            end_time = datetime.now()
            with apputils.cd(python_task.project_path):
                dump_cost(start_time, end_time, task_output_dir)
    finally:
        python_task.reset_project()

    return run_ok


def dump_cost(
    start_time: datetime,
    end_time: datetime,
    task_output_dir: str,
):
    model_stats = common.SELECTED_MODEL.get_overall_exec_stats()
    stats = {
        "commit": apputils.get_current_commit_hash(),
        "start_epoch": start_time.timestamp(),
        "end_epoch": end_time.timestamp(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }
    stats.update(model_stats)

    with open(pjoin(task_output_dir, "cost.json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        # Macos specific requirement for Multi-Processing
        set_start_method("fork", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.remove()
    main()
