import json
from collections.abc import Generator
from pathlib import Path

from loguru import logger

from app.agents import agent_reviewer
from app.agents.agent_common import InvalidLLMResponse, LLMErrorCode # Added LLMErrorCode
# EvaluationPayload will be imported from data_structures
# from dataclasses import dataclass
# from typing import Any

from app.agents.agent_reproducer import TestAgent, TestHandle
from app.agents.agent_reviewer import Review, ReviewDecision
from app.agents.agent_write_patch import PatchAgent, PatchHandle
# from app.api.eval_helper import ResolvedStatus # No longer needed here directly for definition
from app.data_structures import BugLocation, MessageThread, ReproResult, EvaluationPayload # Import EvaluationPayload
from app.log import print_acr, print_review
from app.search.search_manage import SearchManager
from app.task import SweTask, Task


class ReviewManager:
    def __init__(
        self,
        context_thread: MessageThread,
        bug_locs: list[BugLocation],
        search_manager: SearchManager,
        task: Task,
        output_dir: str,
        test_agent: TestAgent,
        repro_result_map: (
            dict[tuple[PatchHandle, TestHandle], ReproResult] | None
        ) = None,
    ) -> None:
        self.issue_stmt = task.get_issue_statement()
        self.patch_agent = PatchAgent(
            task,
            search_manager,
            self.issue_stmt,
            context_thread,
            bug_locs,
            output_dir,
        )
        # self.test_agent = TestAgent(task, output_dir)
        self.test_agent = test_agent
        self.task: Task = task
        self.task_id = task.get_instance_id() # For logging context
        self.repro_result_map: dict[tuple[PatchHandle, TestHandle], ReproResult] = dict(
            repro_result_map or {}
        )
        self.output_dir = output_dir
        self.bound_logger = logger.bind(task_id=self.task_id, component="ReviewManager")

    def patch_only_generator(
        self,
    ) -> Generator[tuple[PatchHandle, str], EvaluationPayload | None, None]: # Changed sent type
        try:
            while True:
                patch_handle, patch_content = (
                    self.patch_agent.write_applicable_patch_without_feedback()
                )
                self.save_patch(patch_handle, patch_content)

                # Yield the patch and expect an EvaluationPayload or None back
                evaluation_feedback: EvaluationPayload | None = yield patch_handle, patch_content

                if evaluation_feedback: # If feedback is provided
                    self.patch_agent.add_feedback(patch_handle, evaluation_feedback)
                    # This generator currently doesn't loop to create a new patch based on this feedback.
                    # It would yield the same patch_handle/patch_content again or exit.
                    # For iterative improvement in patch_only_generator, this part would need a loop
                    # that calls write_applicable_patch_with_feedback.
                    # For now, it just records feedback if sent.
                    # Consider if this generator should enter a feedback loop or if it's truly "patch_only_once_then_maybe_feedback".
                    # If it's the latter, then we might not need to call add_feedback here,
                    # or the calling code should be aware that this feedback won't result in a new patch from this specific generator call.
                    # Assuming for now that feedback is recorded for potential later use, even if not immediately iterative.
                    pass # Explicitly doing nothing more in this loop after feedback
                else: # No feedback received
                    self.bound_logger.debug("Patch only generator yielded patch, no evaluation feedback received to process.")


        except InvalidLLMResponse as e:
            self.bound_logger.error(f"Aborting patch-only with InvalidLLMResponse: {e}", exc_info=True)
            # Re-raise with more specific context if desired, or let it propagate
            # For example: raise InvalidLLMResponse(message=str(e), error_code=e.error_code or LLMErrorCode.OTHER, detail=e.detail) from e
            raise

    def generator(
        self, rounds: int = 5
    ) -> Generator[tuple[PatchHandle, str], EvaluationPayload | None, None]:
        """
        This is the generator when reproducer is available.
        """
        assert isinstance(
            self.task, SweTask
        ), "Only SweTask is supported for reproducer+patch generator."

        try:
            yield from self._generator(rounds)
        except InvalidLLMResponse as e:
            self.bound_logger.error(f"Aborting review with InvalidLLMResponse: {e}", exc_info=True)
            raise # Re-raise or handle as appropriate

    def _generator(
        self, rounds: int
    ) -> Generator[tuple[PatchHandle, str], EvaluationPayload | None, None]: # Matched type with outer generator
        # issue_statement = self.task.get_issue_statement() # Already in self.issue_stmt
        bound_logger = self.bound_logger.bind(generator_step="_generator")

        bound_logger.info(f"Starting review generation process for {rounds} rounds.")

        # TODO: fall back to iterative patch generation when reproduction fails
        if not self.test_agent._history:
            bound_logger.info("No test history found, writing initial reproducing test.")
            (
                test_handle,
                test_content,
                orig_repro_result,
            ) = self.test_agent.write_reproducing_test_without_feedback() # This agent should log internally
            self.test_agent.save_test(test_handle) # save_test could log
        else:
            test_handle = self.test_agent._history[-1]
            test_content = self.test_agent._tests[test_handle]
            orig_repro_result = self.repro_result_map[
                (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
            ]
            bound_logger.info(f"Using existing test {test_handle} from history.")

        coords = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        self.repro_result_map[coords] = orig_repro_result
        self.save_execution_result(orig_repro_result, *coords) # save_execution_result could log
        bound_logger.info(f"Initial test execution saved for test {test_handle}. Reproduced: {orig_repro_result.reproduced}")

        # write the first patch
        bound_logger.info("Writing initial patch.")
        (
            patch_handle,
            patch_content,
        ) = self.patch_agent.write_applicable_patch_without_feedback() # This agent should log internally
        self.save_patch(patch_handle, patch_content) # save_patch could log

        for i in range(rounds):
            round_logger = bound_logger.bind(round=i+1, patch_handle=patch_handle, test_handle=test_handle)
            round_logger.info(f"Starting review round {i+1}.")
            patched_repro_result = self.task.execute_reproducer( # This task method should log
                test_content, patch_content
            )

            coords = (patch_handle, test_handle)
            self.repro_result_map[coords] = patched_repro_result
            self.save_execution_result(patched_repro_result, *coords)
            round_logger.info(f"Patched test execution saved. Reproduced: {patched_repro_result.reproduced}")

            review, review_thread = agent_reviewer.run( # This agent should log internally
                self.issue_stmt, # Use self.issue_stmt
                test_content,
                patch_content,
                orig_repro_result,
                patched_repro_result,
            )

            # print_review already logs a structured message
            print_review(str(review), desc=f"p{patch_handle}_t{test_handle}")
            self.save_review(patch_handle, test_handle, review) # save_review could log
            review_conv_path = Path(self.output_dir, f"conv_review_{patch_handle}_{test_handle}.json")
            review_thread.save_to_file(review_conv_path)
            round_logger.debug(f"Review conversation saved to {review_conv_path}")

            if review.patch_decision == ReviewDecision.YES:
                round_logger.info("Reviewer approved patch.")
                evaluation_payload = yield patch_handle, patch_content
                assert evaluation_payload is not None, "Evaluation payload cannot be None if ReviewDecision is YES"

                # print_acr already logs a structured message
                print_acr(evaluation_payload.to_llm_feedback_string(), f"Patch evaluation (p{patch_handle})")
                self.patch_agent.add_feedback(patch_handle, evaluation_payload)
                # If patch is approved and evaluated, typically the loop might break or behavior changes.
                # Current logic continues, which might mean it's seeking further review or trying other rounds regardless.
                # For now, let's assume this is intended. If overall_eval_passed from payload, could break.
                # Example: if evaluation_payload.status == ResolvedStatus.FULL: break

            if review.patch_decision == ReviewDecision.NO:
                round_logger.info("Reviewer rejected patch. Reason: {}", review.patch_rejection_reason)
                feedback = self.compose_feedback_for_patch_generation(
                    review, test_content
                )
                self.patch_agent.add_feedback(patch_handle, feedback) # PatchAgent should log this
                round_logger.info("Generating new patch based on feedback.")
                (
                    patch_handle, # New patch_handle
                    patch_content,
                ) = self.patch_agent.write_applicable_patch_with_feedback()
                self.save_patch(patch_handle, patch_content) # save_patch could log

            if review.test_decision == ReviewDecision.NO:
                round_logger.info("Reviewer rejected test. Reason: {}", review.test_rejection_reason)
                feedback = self.compose_feedback_for_test_generation(
                    review, patch_content # current patch_content, might be new if patch was rejected and regenerated
                )
                self.test_agent.add_feedback(test_handle, feedback) # TestAgent should log this
                round_logger.info("Generating new test based on feedback.")
                (
                    test_handle, # New test_handle
                    test_content,
                    orig_repro_result, # New baseline result for the new test
                ) = self.test_agent.write_reproducing_test_with_feedback()
                self.test_agent.save_test(test_handle) # save_test could log
                coords = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle) # Update coords for new test
                self.repro_result_map[coords] = orig_repro_result
                self.save_execution_result(orig_repro_result, *coords) # save_execution_result could log
                round_logger.info(f"New test {test_handle} generated and baseline execution saved. Reproduced: {orig_repro_result.reproduced}")

        bound_logger.info("Maximum review rounds reached or loop terminated.")

    @classmethod
    def compose_feedback_for_patch_generation(cls, review: Review, test: str) -> str:
        return (
            f"The previous patch failed a test written by another developer.\n"
            f"Rethink about the code context, reflect, and write another patch.\n"
            f"You can also write the new patch at other locations.\n"
            f"Here is the test file:\n"
            "```\n"
            f"{test}"
            "```\n"
            f"By executing the test file with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.patch_analysis}\n"
            "\n"
            "Therefore, the patch does not correctly resovle the issue.\n"
            "\n"
            "To correct the patch, here is the advice given by another engineer:\n"
            "\n"
            f"{review.patch_advice}"
        )

    @classmethod
    def compose_feedback_for_test_generation(cls, review: Review, patch: str) -> str:
        return (
            f"Here is a patch to the program:\n"
            "```\n"
            f"{patch}"
            "```\n"
            f"By executing your test with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.test_analysis}"
            "\n"
            "Therefore, the test does not correctly reproduce the issue.\n"
            "\n"
            "To correct the test, here is my advice:\n"
            "\n"
            f"{review.test_advice}"
        )

    def save_patch(self, handle: PatchHandle, content: str) -> None:
        file_path = Path(self.output_dir, f"extracted_patch_{handle}.diff")
        file_path.write_text(content)
        self.bound_logger.debug("Saved patch {} to {}", handle, file_path)

    def save_test(self, handle: TestHandle, content: str) -> None:
        file_path = Path(self.output_dir, f"reproducer_{handle}.py")
        file_path.write_text(content)
        self.bound_logger.debug("Saved test {} to {}", handle, file_path)

    def save_review(
        self, patch_handle: PatchHandle, test_handle: TestHandle, review: Review
    ) -> None:
        file_path = Path(self.output_dir, f"review_p{patch_handle}_t{test_handle}.json")
        file_path.write_text(json.dumps(review.to_json(), indent=4))
        self.bound_logger.debug("Saved review for p{}, t{} to {}", patch_handle, test_handle, file_path)

    def save_execution_result(
        self, result: ReproResult, patch_handle: str, test_handle: str
    ) -> None:
        file_path = Path(
            self.output_dir, f"execution_{patch_handle}_{test_handle}.json"
        )
        data_to_save = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "triggered": result.reproduced,
        }
        file_path.write_text(json.dumps(data_to_save, indent=4))
        self.bound_logger.debug("Saved execution result for p{}, t{} to {}. Triggered: {}", patch_handle, test_handle, file_path, result.reproduced)


if __name__ == "__main__":
    pass
    # import json
    # from pathlib import Path
    # from shutil import copy2

    # from icecream import ic

    # from app.model import common
    # from app.model.register import register_all_models
    # from app.raw_tasks import RawSweTask

    # register_all_models()
    # common.set_model("gpt-4-0125-preview")
    # ic(common.SELECTED_MODEL)

    # meta_path = Path(
    #     "/home/crhf/projects/reverse-prompt/acr-plus/output/applicable_patch/django__django-11999_2024-05-18_10-45-42/meta.json"
    # )
    # meta = json.loads(meta_path.read_text())
    # raw_task = RawSweTask(**meta)
    # task = raw_task.to_task()

    # conv_path = Path(
    #     "/home/crhf/projects/reverse-prompt/acr-private/results/acr-run-1/applicable_patch/django__django-11999_2024-04-06_12-54-29/conversation_round_2.json"
    # )
    # msgs = json.loads(conv_path.read_text())
    # thread = MessageThread()
    # for msg in msgs:
    #     content = msg["content"]
    #     role = msg["role"]
    #     if role == "system":
    #         thread.add_system(content)
    #     elif role == "user":
    #         thread.add_user(content)
    #     elif role == "assistant":
    #         thread.add_model(content, [])
    #     else:
    #         assert False, role

    # output_dir = Path("output", "test_review")
    # output_dir.mkdir(exist_ok=True)
    # copy2(meta_path, output_dir)
    # rm = ReviewManager(thread, task, str(output_dir))
    # # patch_gen = rm.generator()
    # # ic(patch_gen.send(None))
    # # ic(patch_gen.send("patch ain't correct"))
    # write_patch_iterative_with_review(thread, task, str(output_dir))
