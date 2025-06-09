"""
An agent, which is only responsible for the write_patch tool call.
"""

from collections import defaultdict
from collections.abc import Generator
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeAlias

from loguru import logger

from app.agents import agent_common
from app.agents.agent_common import InvalidLLMResponse, LLMErrorCode
from app.data_structures import BugLocation, MessageThread
from app.log import print_acr, print_patch_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    convert_response_to_diff,
    extract_diff_one_instance,
    record_extract_status,
)
from app.search.search_manage import SearchManager
from app.task import Task

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Another developer has already collected code context related to the issue for you.
Your task is to write a patch that resolves this issue.
Do not make changes to test files or write tests; you are only interested in crafting a patch.
REMEMBER:
- You should only make minimal changes to the code to resolve the issue.
- Your patch should preserve the program functionality as much as possible.
- In your patch, DO NOT include the line numbers at the beginning of each line!
"""


USER_PROMPT_INIT = """Write a patch for the issue, based on the relevant code context.
First explain the reasoning, and then write the actual patch.
When writing the patch, remember the following:
 - You do not have to modify every location provided - just make the necessary changes.
 - Pay attention to the addtional context as well - sometimes it might be better to fix there.
 - You should import necessary libraries if needed.

Return the patch in the format below.
Within `<file></file>`, replace `...` with actual file path.
Within `<original></original>`, replace `...` with the original code snippet from the program.
Within `<patched></patched>`, replace `...` with the fixed version of the original code.
When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.

Example format:

# modification 1
```
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 2
```
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 3
...
```
NOTE:
- In your patch, DO NOT include the line numbers at the beginning of each line!
- Inside <original> and </original>, you should provide the original code snippet from the program.
This original code snippet MUST match exactly to a continuous block of code in the original program,
since the system will use this to locate the code to be modified.
"""


PatchHandle: TypeAlias = str


class PatchAgent:
    EMPTY_PATCH_HANDLE = "EMPTY"

    def __init__(
        self,
        task: Task,
        search_manager: SearchManager,
        issue_stmt: str,
        context_thread: MessageThread,
        bug_locs: list[BugLocation],
        task_dir: str,
    ) -> None:
        self.task = task
        self.search_manager = search_manager
        self.issue_stmt = issue_stmt
        self.context_thread = context_thread  # the search conv historh thread
        # TODO: merge class_context_code into bug_loc_info, and make them one type
        self.bug_locs: list[BugLocation] = bug_locs
        self.task_dir = task_dir

        self._request_idx: int = -1
        self._responses: dict[PatchHandle, str] = {}
        self._diffs: dict[PatchHandle, str] = {}
        # _feedbacks can now store strings or structured EvaluationPayload objects
        self._feedbacks: dict[PatchHandle, list[Any]] = defaultdict(list) # Using Any for list type
        self._history: list[PatchHandle] = []
        self.task_id = task.get_instance_id()
        self.bound_logger = logger.bind(task_id=self.task_id, agent="PatchAgent")

    def write_applicable_patch_without_feedback(
        self, retries: int = 3
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(max_feedbacks=0, retries=retries)

    def write_applicable_patch_with_feedback(
        self, max_feedbacks: int = 1, retries: int = 3
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(
            max_feedbacks=max_feedbacks, retries=retries
        )

    # Changed feedback type to Any to accommodate str or EvaluationPayload
    def add_feedback(self, handle: PatchHandle, feedback: Any) -> None:
        if handle not in self._diffs:
            if handle != PatchAgent.EMPTY_PATCH_HANDLE:
                self.bound_logger.error("Attempted to add feedback to non-existent patch handle: {}", handle)
                raise ValueError(f"Patch handle {handle} does not exist in diffs to add feedback.")

        self._feedbacks[handle].append(feedback)
        self.bound_logger.debug("Added feedback for patch handle {}. Feedback type: {}", handle, type(feedback).__name__)

    def _write_applicable_patch(
        self, max_feedbacks: int, retries: int
    ) -> tuple[PatchHandle, str]:
        max_feedbacks = max_feedbacks if max_feedbacks >= 0 else len(self._history)
        # Ensure num_feedbacks is not negative if self._history is empty.
        num_feedbacks = min(max_feedbacks, len(self._history)) if self._history else 0
        history_handles = self._history[-num_feedbacks:] if num_feedbacks > 0 else []

        for i in range(retries):
            self.bound_logger.info("Attempt {}/{} to write applicable patch. Max feedbacks: {}. History handles: {}",
                                i + 1, retries, num_feedbacks, history_handles) # Use num_feedbacks not max_feedbacks

            applicable, response, diff_content, thread = self._write_patch(
                history_handles
            )
            self._request_idx += 1

            # print_patch_generation already logs to JSON
            print_patch_generation(response, desc=f"attempt_{self._request_idx}")

            raw_response_path = Path(self.task_dir, f"patch_raw_{self._request_idx}.md")
            raw_response_path.write_text(response)
            self.bound_logger.debug("Saved raw patch LLM response to {}", raw_response_path)

            conv_path = Path(self.task_dir, f"conv_patch_{self._request_idx}.json")
            thread.save_to_file(conv_path)
            self.bound_logger.debug("Saved patch generation conversation to {}", conv_path)

            msg = "Patch is applicable" if applicable else "Patch is not applicable"
            # print_acr already logs to JSON
            print_acr(msg, desc=f"applicability_check_{self._request_idx}")

            if applicable:
                # print_acr already logs to JSON
                print_acr(f"```diff\n{diff_content}\n```", desc=f"extracted_patch_{self._request_idx}")
                handle = self._register_applicable_patch(response, diff_content)
                self.bound_logger.info("Registered applicable patch. Handle: {}, Request Index: {}", handle, self._request_idx)

                return handle, diff_content

        raise InvalidLLMResponse(
            message=f"Failed to write an applicable patch in {retries} attempts",
            error_code=LLMErrorCode.OTHER,
            detail="The agent could not produce a valid, applicable patch after multiple attempts and feedback cycles."
        )

    def _write_patch(
        self,
        history_handles: list[PatchHandle] | None = None,
    ) -> tuple[bool, str, str, MessageThread]:
        history_handles = history_handles or []

        thread = self._construct_init_thread()

        is_first_try = not any(handle in self._feedbacks for handle in history_handles)
        self.bound_logger.debug(f"Patch writing attempt. Is first try for this agent instance: {is_first_try}. History handles provided: {history_handles}")

        for handle in history_handles:
            feedbacks_for_llm = self._feedbacks.get(handle, [])
            if not feedbacks_for_llm:
                if handle in self._responses:
                    thread.add_model(self._responses[handle], [])
            else:
                if handle in self._responses:
                     thread.add_model(self._responses[handle], [])
                else:
                    if handle != PatchAgent.EMPTY_PATCH_HANDLE:
                        self.bound_logger.warning(f"Feedback exists for handle {handle} but no prior response stored.")

                for feedback_item in feedbacks_for_llm:
                    if isinstance(feedback_item, str):
                        thread.add_user(feedback_item)
                    else:
                        try:
                            to_llm_string_method = getattr(feedback_item, "to_llm_feedback_string", None)
                            if callable(to_llm_string_method):
                                feedback_str = to_llm_string_method()
                            else:
                                self.bound_logger.warning(f"Feedback item for handle {handle} is not str and has no 'to_llm_feedback_string' method. Using str().")
                                feedback_str = str(feedback_item)
                            thread.add_user(feedback_str)
                        except Exception as e:
                            self.bound_logger.error(f"Error converting feedback item to string for handle {handle}: {e}. Using str().", exc_info=True)
                            thread.add_user(str(feedback_item))

        thread.add_user(USER_PROMPT_INIT)

        if not history_handles: # First attempt overall for this agent
             # print_acr already logs to JSON
            print_acr(USER_PROMPT_INIT, desc=f"initial_prompt_request_{self._request_idx}")

        self.bound_logger.debug("Calling LLM for patch generation. Current thread length: {}", len(thread.messages))
        patch_resp, *_ = common.SELECTED_MODEL.call(thread.to_msg())
        thread.add_model(patch_resp) # Add model response to thread for saving

        self.bound_logger.debug("Attempting to convert LLM response to diff.")
        extract_status, _, diff_content = convert_response_to_diff(
            patch_resp, self.task_dir # This function might need internal logging improvements
        )
        record_extract_status(self.task_dir, extract_status) # This could log
        self.bound_logger.info("Patch extraction status: {}. Diff length: {}", extract_status, len(diff_content or ""))

        return (
            extract_status == ExtractStatus.APPLICABLE_PATCH,
            patch_resp,
            diff_content,
            thread,
        )

    def _construct_init_thread(self) -> MessageThread:
        """
        Construct the initial patch gen conv thread, based on whether bug location is available.
        """
        if self.bug_locs:
            self.bound_logger.debug("Constructing patch thread with bug_locs.")
            thread = MessageThread()
            thread.add_system(SYSTEM_PROMPT)
            thread.add_user(f"Here is the issue:\n{self.issue_stmt}")
            thread.add_user(self._construct_code_context_prompt())
        else:
            self.bound_logger.debug("Constructing patch thread from search conversation history as no bug_locs provided.")
            messages = deepcopy(self.context_thread.messages)
            thread = MessageThread(messages)
            thread = agent_common.replace_system_prompt(thread, SYSTEM_PROMPT)

        return thread

    def _construct_code_context_prompt(self) -> str:
        prompt = "Here are the possible buggy locations collected by someone else. "
        prompt += (
            "Each location contains the actual code snippet and the intended behavior of "
            "the code for resolving the issue.\n"
        )

        prompt += BugLocation.multiple_locs_to_str_for_model(self.bug_locs)
        prompt += (
            "Note that you DO NOT NEED to modify every location; you should think what changes "
            "are necessary for resolving the issue, and only propose those modifications."
        )
        return prompt

    def _register_applicable_patch(
        self, response: str, diff_content: str
    ) -> PatchHandle:
        handle = str(self._request_idx)

        assert handle not in self._responses
        assert handle not in self._feedbacks
        assert handle not in self._diffs
        assert handle not in self._history

        self._responses[handle] = response
        self._diffs[handle] = diff_content
        self._history.append(handle)

        return handle


def generator(
    context_thread: MessageThread,
    output_dir: str,
) -> Generator[tuple[bool, str, str], str | None, None]:
    """
    Since the agent may not always write an applicable patch, we allow for retries.
    This is a wrapper around the actual run.

    Yields: is_success, result_message, patch_content
    """
    # (1) replace system prompt
    messages = deepcopy(context_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

    # (2) add the initial user prompt
    new_thread.add_user(USER_PROMPT_INIT)
    print_acr(USER_PROMPT_INIT, "patch generation")

    index = 1
    while True:
        if index > 1:
            debug_file = pjoin(output_dir, f"debug_agent_write_patch_{index - 1}.json")
            new_thread.save_to_file(debug_file)

        logger.info(f"Trying to write a patch. Try {index}.")

        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, tools=[])
        print_patch_generation(res_text, f"try {index}")

        logger.info(f"Raw patch produced in try {index}. Writing patch into file.")

        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{index}")
        Path(raw_patch_file).write_text(res_text)

        # Attemp to extract a real patch from the raw patch
        with NamedTemporaryFile(prefix="extracted_patch-", suffix=".diff") as f:
            extract_status, extract_msg = extract_diff_one_instance(
                raw_patch_file, f.name
            )
            patch_content = Path(f.name).read_text()

        # record the extract status. This is for classifying the task at the end of workflow
        record_extract_status(output_dir, extract_status)

        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            print_acr(f"```diff\n{patch_content}\n```", "extracted patch")

            validation_msg = yield True, "written an applicable patch", patch_content

            assert validation_msg is not None

            new_prompt = f"Your patch is invalid. {validation_msg}. Please try again:\n\n{USER_PROMPT_INIT}"
        else:
            _ = yield False, "failed to write an applicable patch", ""

            new_prompt = (
                "Your edit could not be applied to the program. "
                + extract_msg
                + " Please try again."
            )

        # TODO: we may not want to stick to a same thread, or the LLM may
        # be reluctant to try again.
        new_thread.add_user(new_prompt)
        print_patch_generation(new_prompt, f"feedback {index}")

        index += 1
