import json
from collections import defaultdict
from collections.abc import Iterable
from itertools import cycle
from os import PathLike
from os.path import samefile
from pathlib import Path
from shutil import copy2

from loguru import logger
from natsort import natsorted

from app import config
from app.agents import agent_select
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import NoReproductionStep, TestAgent
from app.agents.agent_write_patch import PatchAgent
from app.api import validation
from app.api.review_manage import ReviewManager
from app.api.validation import evaluate_patch
from app.data_structures import BugLocation
from app.log import print_banner, print_issue
from app.manage import ProjectApiManager
from app.model.common import set_model
from app.task import Task


def write_patch_iterative_with_review(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries=3,
) -> bool:
    task_id = task.get_instance_id()
    bound_logger = logger.bind(task_id=task_id, component="write_patch_iterative_with_review")
    bound_logger.info("Start generating patches with reviewer")
    patch_gen = review_manager.generator()

    eval_payload = None # Will now be EvaluationPayload | None
    for _ in range(retries):
        try:
            # Send the previous iteration's eval_payload (or None for the first iteration)
            patch_handle, patch_content = patch_gen.send(eval_payload)
            bound_logger.info("Reviewer approved patch: {}", patch_handle)
        except StopIteration:
            bound_logger.info("Patch generator finished.")
            break
        except InvalidLLMResponse as e:
            bound_logger.error(f"Invalid LLM response from patch generator: {e}", exc_info=True)
            break # Stop if LLM fails badly

        bound_logger.info("Begin evaluating patch: {}", patch_handle)
        # validation.evaluate_patch now returns tuple[bool, EvaluationPayload]
        overall_eval_passed, eval_payload_obj = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )
        eval_payload = eval_payload_obj # Store for the next send()

        # Log the detailed status
        bound_logger.info(f"Patch evaluation status: {eval_payload.status.value}, Message: {eval_payload.message}")
        if eval_payload.details:
             bound_logger.debug(f"Evaluation details: {json.dumps(eval_payload.details, indent=2)}")


        if overall_eval_passed:
            patch_gen.close()

            bound_logger.info(
                "Patch {} passed evaluation. Ending patch generation.", patch_handle
            )
            return True

        bound_logger.warning("Patch {} failed evaluation.", patch_handle) # Changed to warning

    return False


def write_patch_iterative(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries=3,
) -> bool:
    task_id = task.get_instance_id()
    bound_logger = logger.bind(task_id=task_id, component="write_patch_iterative")
    bound_logger.info("Start generating patches without reviewer")

    patch_gen = review_manager.patch_only_generator()

    for _ in range(retries):
        try:
            patch_handle, patch_content = patch_gen.send(None) # No feedback sent in this mode initially
            bound_logger.info("Generated applicable patch: {}", patch_handle)
        except StopIteration:
            bound_logger.info("Patch (no review) generator finished.")
            break
        except InvalidLLMResponse as e:
            bound_logger.error(f"Invalid LLM response from patch (no review) generator: {e}", exc_info=True)
            break

        bound_logger.info("Begin evaluating patch: {}", patch_handle)
        # Capture the full payload to log details
        eval_passed, eval_payload = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )
        bound_logger.info(f"Patch evaluation status: {eval_payload.status.value}, Message: {eval_payload.message}")
        if eval_payload.details:
            bound_logger.debug(f"Evaluation (no review) details: {json.dumps(eval_payload.details, indent=2)}")


        if eval_passed:
            patch_gen.close()
            bound_logger.info(
                "Patch {} passed evaluation. Ending patch generation.", patch_handle
            )
            return True

        bound_logger.warning("Patch {} failed evaluation.", patch_handle) # Changed to warning

    return False


def run_one_task(task: Task, output_dir: str, model_names: Iterable[str]) -> bool:
    """
    Main entry point to run inference on one task.
    Args:
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
        problem_stmt (str): The original problem statement submitted to the task issue.
    """
    assert model_names

    model_name_cycle = cycle(model_names)

    for idx in range(config.overall_retry_limit):
        model_name = next(model_name_cycle)
        set_model(model_name)
        task_id_for_retry = task.get_instance_id() # Get task_id for binding

        logger.bind(task_id=task_id_for_retry, component="run_one_task").info("Starting overall retry {} with model {}", idx, model_name)

        out_dir = Path(output_dir, f"output_{idx}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # meta.json is used later by convert_response_to_diff(),
        # so it needs to be copied over
        meta_file = Path(output_dir, "meta.json")
        if meta_file.exists():
            copy2(meta_file, out_dir)

        api_manager = ProjectApiManager(task, str(out_dir))

        if _run_one_task(str(out_dir), api_manager, task.get_issue_statement()): # This will have its own task_id binding
            logger.bind(task_id=task_id_for_retry, component="run_one_task").info("Overall retry {} succeeded; ending workflow.", idx)
            break

        logger.bind(task_id=task_id_for_retry, component="run_one_task").warning("Overall retry {} failed; proceeding to next retry.", idx)

    logger.bind(task_id=task.get_instance_id(), component="run_one_task").info("Starting patch selection") # Bind task_id here too

    selected, details = select_patch(task, output_dir) # select_patch will bind its own task_id
    Path(output_dir, "selected_patch.json").write_text(json.dumps(details, indent=4))

    logger.bind(task_id=task.get_instance_id(), component="run_one_task").info("Selected patch {}. Reason: {}", selected, details["reason"])

    return True


def select_patch(task: Task, output_dir: str | PathLike) -> tuple[str, dict]:

    patches = natsorted(list(Path(output_dir).glob("**/extracted_patch_*.diff")))

    # TODO: These candidate patches must have been dismissed by reviewer. Maybe an
    # assertion should be added to confirm this.

    # Bind task_id for select_patch logs
    task_id = task.get_instance_id()
    bound_logger = logger.bind(task_id=task_id, component="select_patch")

    candidate_patches = [p for p in patches if may_pass_regression_tests(task, p)] # may_pass_regression_tests also needs context if it logs

    agent_comment = None
    thread = None

    bound_logger.debug("Candidate patches for selection: {}", candidate_patches)

    for p in candidate_patches:
        index = p.with_suffix("").name.rpartition("_")[2]
        reviews = natsorted(
            list(p.parent.glob(f"review_p{index}_t*.json")), reverse=True
        )
        if not reviews:
            continue
        assert len(reviews) == 1, p
        if json.loads(reviews[0].read_text())["patch-correct"] == "yes":
            last_patch = natsorted(patches)[-1]
            assert samefile(
                p, last_patch
            ), f"{p} is approved and passes validation, but the last patch was {last_patch}"
            selected_patch = p
            reason = "reviewer-approved"
            break
    else:
        if len(candidate_patches) > 1:
            content_to_indices = defaultdict(list)
            for idx, p in enumerate(candidate_patches):
                content_to_indices[p.read_text()].append(idx)
            items = sorted(
                content_to_indices.items(),
                key=lambda item: (len(item[1]), -item[1][0]),
                reverse=True,
            )

            # if len(items[0]) > 1:
            if False:
                index = items[0][1][0]
                selected_patch = candidate_patches[index]
                reason = "majority,multiple-pass-regression"
            else:
                try:
                    index, agent_comment, thread = agent_select.run(
                        task.get_issue_statement(),
                        [p.read_text() for p in candidate_patches],
                    )
                    reason = "agent-selected,multiple-pass-regression"
                except Exception as e:
                    bound_logger.error(f"Agent select for multiple candidates failed: {e}", exc_info=True)
                    index = -1 # Ensure index is set for selected_patch access
                    reason = "agent-error,multiple-pass-regression"
                # Check if candidate_patches is not empty and index is valid
                if candidate_patches and index >= 0 and index < len(candidate_patches):
                    selected_patch = candidate_patches[index]
                elif candidate_patches: # Fallback if agent failed or returned invalid index
                    bound_logger.warning("Agent selection failed or returned invalid index, falling back to first candidate.")
                    selected_patch = candidate_patches[0]
                    reason = "agent-fallback,multiple-pass-regression"
                else: # No candidates, this branch shouldn't be hit if len(candidate_patches) > 1
                      # but as a safeguard:
                    bound_logger.error("No candidate patches found in agent selection fallback logic.")
                    # This state is problematic, how to select a patch?
                    # For now, this will likely error out if no patches exist and this path is taken.
                    # Or, ensure selected_patch is defined if this else path is taken.
                    # This part of the logic might need revisiting if candidate_patches can be empty here.
                    # If no patches, this whole 'else' for len(candidate_patches) > 1 won't run.
                    # The code will proceed to the len(candidate_patches) == 1 or == 0 cases.
                    pass # Let it fall through, outer logic handles no patches

        elif len(candidate_patches) == 1:
            selected_patch = candidate_patches[0]
            reason = "no-agent,single-pass-regression"
        else:
            content_to_indices = defaultdict(list)
            for idx, p in enumerate(patches):
                content_to_indices[p.read_text()].append(idx)
            items = sorted(
                content_to_indices.items(),
                key=lambda item: (len(item[1]), -item[1][0]),
                reverse=True,
            )

            # if len(items[0]) > 1:
            if False:
                index = items[0][1][0]
                selected_patch = patches[index]
                reason = "majority,none-pass-regression"
            else:
                try:
                    index, agent_comment, thread = agent_select.run(
                        task.get_issue_statement(), [p.read_text() for p in patches]
                    )
                    reason = "agent-selected,none-pass-regression"
                except Exception as e:
                    bound_logger.error(f"Agent select for no regression pass candidates failed: {e}", exc_info=True)
                    index = -1
                    reason = "agent-error,none-pass-regression"

                if patches and index >= 0 and index < len(patches):
                    selected_patch = patches[index]
                elif patches: # Fallback if agent failed or returned invalid index
                    bound_logger.warning("Agent selection failed or returned invalid index, falling back to first patch.")
                    selected_patch = patches[0]
                    reason = "agent-fallback,none-pass-regression"
                else: # No patches at all
                    bound_logger.error("No patches available for selection by agent.")
                    # This indicates a problem, as select_patch should ideally always select something if patches exist.
                    # Or handle the case where 'patches' list itself is empty.
                    # If 'patches' is empty, this 'else' block for len(candidate_patches) == 0 is hit.
                    # The function must return a valid path string, or raise error.
                    # For now, assuming 'patches' is non-empty if this logic path is taken.
                    # If patches is empty, this code will fail. Need to check len(patches) == 0 first.

    # Handle case where no patches are available at all
    if not patches:
        bound_logger.error("No patches found in the output directory for selection.")
        # Return a placeholder or raise an exception, as no patch can be selected.
        # Depending on expected behavior, this might be an error state.
        # For now, returning a dummy path and reason.
        return "no_patch_found.diff", {"selected_patch": "no_patch_found.diff", "reason": "no-patches-available"}


    # Ensure selected_patch is defined before attempting to use it.
    # This can happen if candidate_patches was empty and patches was also empty through some logic error.
    if 'selected_patch' not in locals() and patches: # If not defined, but patches exist, default to first overall patch
        bound_logger.warning("Selected_patch was not defined despite patches existing. Defaulting to first patch.")
        selected_patch = patches[0]
        reason = "fallback-undefined-selection"
    elif 'selected_patch' not in locals() and not patches: # Should have been caught by 'if not patches:'
         # This case should ideally not be reached if the above 'if not patches:' handles it.
         # Re-asserting for clarity or if logic changes.
         bound_logger.error("Critical: No patches available and selected_patch is undefined.")
         return "critical_no_patch.diff", {"selected_patch": "critical_no_patch.diff", "reason": "critical-no-patches"}


    rel_selected_patch = str(selected_patch.relative_to(output_dir))

    rel_selected_patch = str(selected_patch.relative_to(output_dir))

    result = {
        "selected_patch": rel_selected_patch,
        "reason": reason,
    }

    if agent_comment is not None:
        result["agent_comment"] = agent_comment

    if thread is not None:
        thread.save_to_file(Path(output_dir, "agent_selection.json"))

    return str(selected_patch.relative_to(output_dir)), result


def may_pass_regression_tests(task: Task, patch_file: str | PathLike) -> bool:
    if not config.enable_validation:
        return True

    patch_file = Path(patch_file)

    patch_idx = patch_file.with_suffix("").name.rpartition("_")[2]

    regression_file = patch_file.with_name(f"regression_{patch_idx}.json")
    if regression_file.exists():
        return json.loads(regression_file.read_text())["no_additional_failure"]

    task.reset_project()
    pass_evaluation, _ = evaluate_patch(
        task, patch_idx, patch_file.read_text(), str(patch_file.parent)
    )

    return pass_evaluation


def _run_one_task(
    output_dir: str, api_manager: ProjectApiManager, problem_stmt: str
) -> bool:
    task_id = api_manager.task.get_instance_id()
    bound_logger = logger.bind(task_id=task_id, component="_run_one_task")

    # These print_banner and print_issue calls already log to JSON via app.log modifications
    print_banner("Starting AutoCodeRover on the following issue")
    print_issue(problem_stmt)

    test_agent = TestAgent(api_manager.task, output_dir) # TestAgent might need task_id for its own logging

    repro_result_map = {}
    repro_stderr = ""
    reproduced = False
    reproduced_test_content = None
    try:
        test_handle, test_content, orig_repro_result = (
            test_agent.write_reproducing_test_without_feedback()
        )
        test_agent.save_test(test_handle)

        coord = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        repro_result_map[coord] = orig_repro_result

        if orig_repro_result.reproduced:
            repro_stderr = orig_repro_result.stderr
            reproduced = True
            reproduced_test_content = test_content
        # TODO: utilize the test for localization
    except NoReproductionStep as e:
        bound_logger.info(
            "Test agent decides that the issue statement does not contain "
            f"reproduction steps; skipping reproducer tracing. Details: {e}"
        )
    except InvalidLLMResponse as e:
        bound_logger.warning(f"Failed to write a reproducer test; skipping reproducer tracing: {e}", exc_info=True)

    if config.enable_sbfl:
        sbfl_result, *_ = api_manager.fault_localization()
    else:
        sbfl_result = ""

    bug_locs: list[BugLocation]
    bug_locs, search_msg_thread = api_manager.search_manager.search_iterative( # This itself might log
        api_manager.task, sbfl_result, repro_stderr, reproduced_test_content
    )

    bound_logger.info("Search completed. Bug locations found: {}", len(bug_locs))
    bound_logger.debug("Bug locations details: {}", bug_locs)


    # logger.info("Additional class context code: {}", class_context_code) # This was commented out
    # done with search; dump the tool calls used for recording
    api_manager.search_manager.dump_tool_call_layers_to_file() # This might log

    # Write patch
    # print_banner already logs to JSON
    print_banner("PATCH GENERATION")
    bound_logger.debug("Gathered enough information. Invoking write_patch.")

    review_manager = ReviewManager( # ReviewManager might need task_id for its logging
        search_msg_thread,
        bug_locs,
        api_manager.search_manager,
        api_manager.task,
        output_dir,
        test_agent,
        repro_result_map,
    )

    if config.reproduce_and_review and reproduced:
        try:
            return write_patch_iterative_with_review( # This function now binds its own task_id
                api_manager.task, output_dir, review_manager
            )
        # this exception can arise when writing new reproducers
        except NoReproductionStep as e:
            bound_logger.warning(f"NoReproductionStep caught during review process: {e}. Falling back to non-review iteration.", exc_info=True)
            # Fall through to non-review version

    result = write_patch_iterative(api_manager.task, output_dir, review_manager) # This function also binds its own task_id
    # The message below might be misleading if result from write_patch_iterative is True (meaning patch was found and passed)
    if not reproduced:
        bound_logger.info(
            "write_patch_iterative finished. Original issue had no reproducer or reproducer failed. Workflow outcome depends on patch evaluation."
        )
    else:
        bound_logger.info("write_patch_iterative finished. Workflow outcome depends on patch evaluation.")
    return result


if __name__ == "__main__":
    from app.raw_tasks import RawSweTask

    config.enable_validation = True

    applicable_path = Path(
        "/media/media0/haifeng/projects/reverse-prompt/acr-plus/experiment/06-13-docker-val-loop-lite-try-2-rand/applicable_patch/"
    )
    task_dirs = list(applicable_path.glob("*"))
    for task_dir in task_dirs:
        meta = json.loads(task_dir.joinpath("meta.json").read_text())
        raw_task = RawSweTask(meta["task_id"], meta["setup_info"], meta["task_info"])
        task = raw_task.to_task()
        selected_patch, reason = select_patch(task, task_dir)

        task_dir.joinpath("selected_patch.json").write_text(
            json.dumps({"selected_patch": selected_patch, "reason": reason}, indent=4)
        )
