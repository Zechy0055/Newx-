from __future__ import annotations

import json
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from app.agents.agent_common import InvalidLLMResponse, LLMErrorCode
from app.data_structures import MessageThread, ReproResult
from app.model import common

# TEMP, for testing

SYSTEM_PROMPT = (
    "You are an experienced software engineer responsible for maintaining a project."
    "An issue has been submitted. "
    "Engineer A has written a reproduction test for the issue. "
    "Engineer B has written a patch for the issue. "
    "Your task is to decide whether the created patch resolves the issue."
    "NOTE: both the test and the patch may be wrong."
)

INITIAL_REQUEST = ()


@dataclass
class Review:
    patch_decision: ReviewDecision
    patch_analysis: str
    patch_advice: str
    patch_rejection_reason: RejectionReason | None = None  # Added
    test_decision: ReviewDecision
    test_analysis: str
    test_advice: str
    test_rejection_reason: RejectionReason | None = None  # Added

    def __str__(self):
        patch_reason_str = f"Patch rejection reason: {self.patch_rejection_reason.value}\n\n" if self.patch_rejection_reason else ""
        test_reason_str = f"Test rejection reason: {self.test_rejection_reason.value}\n\n" if self.test_rejection_reason else ""
        return (
            f"Patch decision: {self.patch_decision.value}\n\n"
            f"{patch_reason_str}"
            f"Patch analysis: {self.patch_analysis}\n\n"
            f"Patch advice: {self.patch_advice}\n\n"
            f"Test decision: {self.test_decision.value}\n\n"
            f"{test_reason_str}"
            f"Test analysis: {self.test_analysis}\n\n"
            f"Test advice: {self.test_advice}"
        )

    def to_json(self):
        return {
            "patch-correct": self.patch_decision.value,
            "patch-analysis": self.patch_analysis,
            "patch-advice": self.patch_advice,
            "patch-rejection-reason": self.patch_rejection_reason.value if self.patch_rejection_reason else None,
            "test-correct": self.test_decision.value,
            "test-analysis": self.test_analysis,
            "test-advice": self.test_advice,
            "test-rejection-reason": self.test_rejection_reason.value if self.test_rejection_reason else None,
        }


class ReviewDecision(Enum):
    YES = "yes"
    NO = "no"


class RejectionReason(Enum):
    # Patch related
    PATCH_APPLY_FAIL = "PATCH_APPLY_FAIL"  # Technical issue: Patch does not apply cleanly
    PATCH_NO_IMPROVEMENT = "PATCH_NO_IMPROVEMENT"  # Functional issue: Patch applies but test still fails as before / issue not resolved
    PATCH_REGRESSION = "PATCH_REGRESSION"  # Functional issue: Patch causes new issues / makes things worse (e.g. test that passed now fails)
    PATCH_BAD_SYNTAX = "PATCH_BAD_SYNTAX" # Technical issue: Patch has syntax errors
    PATCH_OTHER = "PATCH_OTHER" # Other patch-related reasons

    # Test related
    TEST_DOES_NOT_FAIL = "TEST_DOES_NOT_FAIL"  # Functional issue: Test passes on original code (doesn't reproduce bug)
    TEST_FLAKY = "TEST_FLAKY"  # Technical issue: Test produces inconsistent results
    TEST_IRRELEVANT = "TEST_IRRELEVANT"  # Functional issue: Test is not relevant to the described issue
    TEST_BAD_SYNTAX = "TEST_BAD_SYNTAX" # Technical issue: Test has syntax errors
    TEST_OTHER = "TEST_OTHER" # Other test-related reasons


def extract_review_result(content: str) -> Review | None:
    try:
        data = json.loads(content)

        patch_decision = ReviewDecision(data["patch-correct"].lower())
        test_decision = ReviewDecision(data["test-correct"].lower())

        # Placeholder for actual LLM providing these reasons
        # For now, if it's a "no", we'll assign a generic "OTHER" reason.
        # A more sophisticated approach would involve parsing analysis/advice for keywords
        # or having the LLM explicitly state the reason code.
        patch_rejection_reason = None
        if patch_decision == ReviewDecision.NO:
            patch_rejection_reason = RejectionReason(data.get("patch-rejection-reason", RejectionReason.PATCH_OTHER.value).upper()) \
                if data.get("patch-rejection-reason") else RejectionReason.PATCH_OTHER

        test_rejection_reason = None
        if test_decision == ReviewDecision.NO:
            test_rejection_reason = RejectionReason(data.get("test-rejection-reason", RejectionReason.TEST_OTHER.value).upper()) \
                if data.get("test-rejection-reason") else RejectionReason.TEST_OTHER

        review = Review(
            patch_decision=patch_decision,
            patch_analysis=data["patch-analysis"],
            patch_advice=data["patch-advice"],
            patch_rejection_reason=patch_rejection_reason,
            test_decision=test_decision,
            test_analysis=data["test-analysis"],
            test_advice=data["test-advice"],
            test_rejection_reason=test_rejection_reason,
        )

        # If advice is missing for a "no" decision, it might be an invalid review
        if (patch_decision == ReviewDecision.NO and not review.patch_advice) or \
           (test_decision == ReviewDecision.NO and not review.test_advice):
            # Allow if a specific reason is given, even if advice is minimal
            if not (patch_rejection_reason and patch_rejection_reason != RejectionReason.PATCH_OTHER) and \
               not (test_rejection_reason and test_rejection_reason != RejectionReason.TEST_OTHER):
                logger.warning("Review marked NO but missing advice and specific reason.")
                # Depending on strictness, could return None here
                # For now, let it pass if JSON is valid.

        return review

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Error parsing review result from LLM content.", exc_info=True, extra={"llm_content": content})
        return None


def run(
    issue_statement: str,
    test_content: str,
    patch_content: str,
    orig_repro: ReproResult,
    patched_repro: ReproResult,
    retries: int = 5,
) -> tuple[Review, MessageThread]:
    # The actual LLM call and extraction logic is in run_with_retries.
    # For now, the RejectionReason will be PATCH_OTHER or TEST_OTHER if not provided by LLM.
    # This function's signature doesn't change, but the Review object it returns is now richer.
    review_generator = run_with_retries(
        issue_statement,
        test_content,
        patch_content,
        orig_repro.stdout,
        orig_repro.stderr,
        patched_repro.stdout,
        patched_repro.stderr,
        retries=retries,
    )
    for review, thread in review_generator:
        if review is not None:
            return review, thread

    # If run_with_retries fails, it will raise InvalidLLMResponse from there or after loop finishes.
    logger.error("Failed to obtain valid review after {} retries.", retries)
    raise InvalidLLMResponse(
        message=f"Failed to review in {retries} attempts",
        error_code=LLMErrorCode.OTHER,
        detail="Agent failed to get a valid review from LLM after multiple retries."
        )


# TODO: remove this
def run_with_retries( # This function is what actually calls the LLM
    issue_statement: str,
    test: str,
    patch: str,
    orig_test_stdout: str,
    orig_test_stderr: str,
    patched_test_stdout: str,
    patched_test_stderr: str,
    retries: int = 5,
) -> Generator[tuple[Review | None, MessageThread], None, None]:
    bound_logger = logger.bind(agent="ReviewerAgent", method="run_with_retries")
    bound_logger.debug("Initiating review generation with {} retries.", retries)
    prefix_thread = MessageThread()
    prefix_thread.add_system(SYSTEM_PROMPT)

    issue_prompt = f"Here is the issue: <issue>{issue_statement}</issue>.\n"
    prefix_thread.add_user(issue_prompt)

    test_prompt = f"Here is the test written by Engineer A: <test>{test}</test>.\n"
    prefix_thread.add_user(test_prompt)

    orig_exec_prompt = (
        "Here is the result of executing the test on the original buggy program:\n"
        f"stdout:\n\n{orig_test_stdout}\n"
        "\n"
        f"stderr:\n\n{orig_test_stderr}\n"
        "\n"
    )

    prefix_thread.add_user(orig_exec_prompt)

    patch_prompt = f"Here is the patch written by Engineer B: <patch>{patch}</patch>.\n"
    prefix_thread.add_user(patch_prompt)

    patched_exec_prompt = (
        "Here is the result of executing the test on the patched program:\n"
        f"stdout:\n\n{patched_test_stdout}\n"
        "\n"
        f"stderr:\n\n{patched_test_stderr}."
    )
    prefix_thread.add_user(patched_exec_prompt)

    # Updated question to ask for rejection reason if applicable
    question = (
        "Think about (1) whether the test correctly reproduces the issue, and "
        "(2) whether the patch resolves the issue. "
        "Provide your answer in the following json schema:\n"
        "\n"
        "```json\n"
        "{\n"
        '    "patch-correct": "yes|no",\n'
        '    "patch-rejection-reason": "PATCH_APPLY_FAIL|PATCH_NO_IMPROVEMENT|PATCH_REGRESSION|PATCH_BAD_SYNTAX|PATCH_OTHER",\n' # Optional if patch-correct is yes
        '    "patch-analysis": "...",\n'
        '    "patch-advice": "...",\n' # Optional if patch-correct is yes
        '    "test-correct": "yes|no",\n'
        '    "test-rejection-reason": "TEST_DOES_NOT_FAIL|TEST_FLAKY|TEST_IRRELEVANT|TEST_BAD_SYNTAX|TEST_OTHER",\n' # Optional if test-correct is yes
        '    "test-analysis": "...",\n'
        '    "test-advice": "..."\n' # Optional if test-correct is yes
        "}\n"
        "```\n"
        "\n"
        'If "patch-correct" is "no", provide a "patch-rejection-reason" from the given list and "patch-advice".\n'
        'If "test-correct" is "no", provide a "test-rejection-reason" from the given list and "test-advice".\n'
        'If the decision is "yes", the corresponding rejection reason and advice fields can be omitted or null.\n'
        "Ensure analysis fields are always filled.\n"
        "NOTE: not only the patch, but also the test case, can be wrong."
    )
    prefix_thread.add_user(question)

    for attempt_num in range(1, retries + 1):
        bound_logger.debug("Review generation attempt {}/{}.", attempt_num, retries)
        response, *_ = common.SELECTED_MODEL.call(
            prefix_thread.to_msg(), response_format="json_object"
        )

        thread = deepcopy(prefix_thread)
        thread.add_model(response, [])

        bound_logger.debug("LLM raw response for review (attempt {}): {}", attempt_num, response)

        review = extract_review_result(response) # extract_review_result logs errors internally

        if review is None:
            bound_logger.warning("Review extraction failed for attempt {}. LLM response was: {}", attempt_num, response)
            if attempt_num < retries: # Only yield None if it's not the last attempt and we might retry
                yield None, thread
            # On last attempt, if review is None, loop finishes and run() function will raise error.
            continue

        bound_logger.info("Successfully extracted review on attempt {}.", attempt_num)
        yield review, thread
        return # Successfully yielded a review, so exit generator


if __name__ == "__main__":
    pass

#     # setup before test

#     register_all_models()
#     common.set_model("gpt-4-0125-preview")

#     # TEST
#     instance_id = "matplotlib__matplotlib-23299"

#     problem_stmt = "[Bug]: get_backend() clears figures from Gcf.figs if they were created under rc_context\n### Bug summary\r\n\r\ncalling `matplotlib.get_backend()` removes all figures from `Gcf` if the *first* figure in `Gcf.figs` was created in an `rc_context`.\r\n\r\n### Code for reproduction\r\n\r\n```python\r\nimport matplotlib.pyplot as plt\r\nfrom matplotlib import get_backend, rc_context\r\n\r\n# fig1 = plt.figure()  # <- UNCOMMENT THIS LINE AND IT WILL WORK\r\n# plt.ion()            # <- ALTERNATIVELY, UNCOMMENT THIS LINE AND IT WILL ALSO WORK\r\nwith rc_context():\r\n    fig2 = plt.figure()\r\nbefore = f'{id(plt._pylab_helpers.Gcf)} {plt._pylab_helpers.Gcf.figs!r}'\r\nget_backend()\r\nafter = f'{id(plt._pylab_helpers.Gcf)} {plt._pylab_helpers.Gcf.figs!r}'\r\n\r\nassert before == after, '\\n' + before + '\\n' + after\r\n```\r\n\r\n\r\n### Actual outcome\r\n\r\n```\r\n---------------------------------------------------------------------------\r\nAssertionError                            Traceback (most recent call last)\r\n<ipython-input-1-fa4d099aa289> in <cell line: 11>()\r\n      9 after = f'{id(plt._pylab_helpers.Gcf)} {plt._pylab_helpers.Gcf.figs!r}'\r\n     10 \r\n---> 11 assert before == after, '\\n' + before + '\\n' + after\r\n     12 \r\n\r\nAssertionError: \r\n94453354309744 OrderedDict([(1, <matplotlib.backends.backend_qt.FigureManagerQT object at 0x7fb33e26c220>)])\r\n94453354309744 OrderedDict()\r\n```\r\n\r\n### Expected outcome\r\n\r\nThe figure should not be missing from `Gcf`.  Consequences of this are, e.g, `plt.close(fig2)` doesn't work because `Gcf.destroy_fig()` can't find it.\r\n\r\n### Additional information\r\n\r\n_No response_\r\n\r\n### Operating system\r\n\r\nXubuntu\r\n\r\n### Matplotlib Version\r\n\r\n3.5.2\r\n\r\n### Matplotlib Backend\r\n\r\nQtAgg\r\n\r\n### Python version\r\n\r\nPython 3.10.4\r\n\r\n### Jupyter version\r\n\r\nn/a\r\n\r\n### Installation\r\n\r\nconda\n"

#     test = """# reproducer.py
# import matplotlib.pyplot as plt
# from matplotlib import get_backend, rc_context

# def main():
#     # Uncommenting either of the lines below would work around the issue
#     # fig1 = plt.figure()
#     # plt.ion()
#     with rc_context():
#         fig2 = plt.figure()
#     before = f'{id(plt._pylab_helpers.Gcf)} {plt._pylab_helpers.Gcf.figs!r}'
#     get_backend()
#     after = f'{id(plt._pylab_helpers.Gcf)} {plt._pylab_helpers.Gcf.figs!r}'

#     assert before == after, '\n' + before + '\n' + after

# if __name__ == "__main__":
#     main()
# """

#     patch = """diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
# index c268a56724..b40f1246b9 100644
# --- a/lib/matplotlib/__init__.py
# +++ b/lib/matplotlib/__init__.py
# @@ -1087,7 +1087,9 @@ def rc_context(rc=None, fname=None):
#               plt.plot(x, y)  # uses 'print.rc'

#      \"\"\"
# +    from matplotlib._pylab_helpers import Gcf
#      orig = rcParams.copy()
# +    orig_figs = Gcf.figs.copy()  # Preserve the original figures
#      try:
#          if fname:
#              rc_file(fname)
# @@ -1096,6 +1098,7 @@ def rc_context(rc=None, fname=None):
#          yield
#      finally:
#          dict.update(rcParams, orig)  # Revert to the original rcs.
# +        Gcf.figs.update(orig_figs)  # Restore the original figures


#  def use(backend, *, force=True):"""

#     # run_with_retries(problem_stmt, test, patch)

#     success = False

#     for attempt_idx, (raw_response, thread, review_result) in enumerate(
#         run_with_retries(problem_stmt, test, patch), start=1
#     ):

#         success |= review_result is not None

#         # dump raw results for debugging
#         Path(f"agent_reviewer_raw_{attempt_idx}.json").write_text(
#             json.dumps(thread.to_msg(), indent=4)
#         )

#         if success:
#             print(f"Success at attempt {attempt_idx}. Review result is {review_result}")
#             break

#     if not success:
#         print("Still failing to produce valid review results after 5 attempts")
