"""Core data structures used throughout the AutoCodeRover application.

This module defines various classes and dataclasses for representing entities
such as method identifiers, LLM function call intents, message threads for
conversations, results of reproduction attempts, search results, identified
bug locations, and payloads for evaluation results.
"""
import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from pprint import pformat

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)

from app import utils as apputils
from app.search import search_utils
# Need ResolvedStatus for EvaluationPayload, but eval_helper might import data_structures.
# To avoid circular dependency, consider defining EvaluationPayload in a different file
# or using forward declaration if Python supported it easily for dataclasses/enums.
# For now, let's try importing. If it causes issues, we'll need to move EvaluationPayload.
from app.api.eval_helper import ResolvedStatus # Potential circular import
from typing import Any # For EvaluationPayload details


@dataclass
class MethodId:
    """Represents a unique identifier for a method within a class or globally.

    Attributes:
        class_name (str): The name of the class containing the method.
                          Empty if the method is not part of a class (e.g., a global function).
        method_name (str): The name of the method or function.
    """
    class_name: str
    method_name: str

    def __str__(self):
        if self.class_name:
            return f"{self.class_name}.{self.method_name}"
        return self.method_name

    def __hash__(self):
        return hash((self.class_name, self.method_name))


class FunctionCallIntent:
    """An intent to call a tool function.

    This object created from OpenAI API response.
    """

    def __init__(
        self,
        func_name: str,
        arguments: Mapping[str, str],
        openai_func: OpenaiFunction | None,
    ):
        """Initializes a FunctionCallIntent.

        Args:
            func_name (str): The name of the function to be called.
            arguments (Mapping[str, str]): A mapping of argument names to their string values.
            openai_func (OpenaiFunction | None): The original OpenAI function object, if available.
                                                 Used for resending tool call information to the model.
        """
        self.func_name = func_name
        self.arg_values = dict()
        self.arg_values.update(arguments)
        # record the original openai function object,
        # which is used when we want tell the model that it has
        # previously called this function/tool
        self.openai_func = openai_func or OpenaiFunction(
            arguments=json.dumps(arguments), name=func_name
        )

    def __str__(self):
        return f"FunctionCallIntent(func_name={self.func_name}, arguments={self.arg_values})"

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the function call intent.

        Returns:
            dict[str, Any]: Dictionary with 'func_name' and 'arguments'.
        """
        return {"func_name": self.func_name, "arguments": self.arg_values}

    def to_dict_with_result(self, call_ok: bool) -> dict[str, Any]:
        """Returns a dictionary representation including the call success status.

        Args:
            call_ok (bool): Whether the function call was successful.

        Returns:
            dict[str, Any]: Dictionary with 'func_name', 'arguments', and 'call_ok'.
        """
        return {
            "func_name": self.func_name,
            "arguments": self.arg_values,
            "call_ok": call_ok,
        }


class MessageThread:
    """
    Represents a thread of conversation with the model.
    Represents a thread of conversation with an LLM model.

    This class manages a list of messages, allowing for messages to be added from
    different roles (system, user, assistant, tool) and for the entire thread
    to be serialized, saved, or loaded. It's designed to be compatible with
    OpenAI's message format.
    """

    def __init__(self, messages: list[dict] | None = None):
        """Initializes a MessageThread.

        Args:
            messages (list[dict] | None, optional): An optional list of initial messages
                to populate the thread. Defaults to None, starting an empty thread.
        """
        self.messages: list[dict] = messages or []

    def add(self, role: str, message: str):
        """Adds a generic message to the thread.

        Args:
            role (str): The role of the message sender (e.g., "user", "system").
            message (str): The content of the message.
        """
        self.messages.append({"role": role, "content": message})

    def add_system(self, message: str):
        """Adds a system message to the thread.

        Args:
            message (str): The content of the system message.
        """
        self.messages.append({"role": "system", "content": message})

    def add_user(self, message: str):
        """Adds a user message to the thread.

        Args:
            message (str): The content of the user message.
        """
        self.messages.append({"role": "user", "content": message})

    def add_tool(self, message: str, tool_call_id: str):
        """Adds a tool message (response from a tool call) to the thread.

        Args:
            message (str): The content returned by the tool.
            tool_call_id (str): The ID of the tool call this message is a response to.
        """
        m = {"role": "tool", "content": message, "tool_call_id": tool_call_id}
        self.messages.append(m)

    def add_model(
        self, message: str | None, tools: list[ChatCompletionMessageToolCall] = []
    ):
        """Adds a message from the model (assistant) to the thread.

        This can be a simple text message or include tool calls requested by the model.

        Args:
            message (str | None): The textual content from the model. Can be None if only tool calls are present.
            tools (list[ChatCompletionMessageToolCall], optional): A list of tool calls
                requested by the model. Defaults to an empty list.
        """
        json_tools = []
        for tool in tools:
            this_tool_dict = {}
            this_tool_dict["id"] = tool.id
            this_tool_dict["type"] = tool.type
            # now serialize function as well
            func_obj: OpenaiFunction = tool.function
            func_args: str = func_obj.arguments
            func_name: str = func_obj.name
            this_tool_dict["function"] = {"name": func_name, "arguments": func_args}
            json_tools.append(this_tool_dict)

        if json_tools == []:
            # there is no tool calls from the model last time,
            # the best we could do is to return the generated text
            self.messages.append({"role": "assistant", "content": message})
        else:
            self.messages.append(
                {"role": "assistant", "content": None, "tool_calls": json_tools}
            )

    def to_msg(self) -> list[dict]:
        """
        Convert to the format to be consumed by the model.
        Returns:
            List[Dict]: The message thread.
        """
        return self.messages

    def __str__(self):
        return pformat(self.messages, width=160, sort_dicts=False)

    def save_to_file(self, file_path: str | PathLike):
        """
        Save the current state of the message thread to a file.
        Args:
            file_path (str): The path to the file.
        """
        Path(file_path).write_text(json.dumps(self.messages, indent=4))

    def get_round_number(self) -> int:
        """
        From the current message history, decide how many rounds have been completed.
        """
        completed_rounds = 0
        for message in self.messages:
            if message["role"] == "assistant":
                completed_rounds += 1
        return completed_rounds

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load the message thread from a file.
        Args:
            file_path (str): The path to the file.
        Returns:
            MessageThread: The message thread.
        """
        with open(file_path) as f:
            messages = json.load(f)
        return cls(messages)


class ReproResult:
    """Represents the result of an attempt to reproduce an issue.

    Attributes:
        stdout (str): The standard output from the reproduction attempt.
        stderr (str): The standard error output from the reproduction attempt.
        returncode (int): The exit code of the reproduction script.
        reproduced (bool): True if the issue was successfully reproduced (typically indicated
                           by a non-zero return code and "AssertionError" in stderr),
                           False otherwise.
    """
    reproduced: bool
    stdout: str
    stderr: str
    returncode: int

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        """Initializes a ReproResult.

        Args:
            stdout (str): Standard output from the execution.
            stderr (str): Standard error from the execution.
            returncode (int): Exit code of the execution.
        """
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.reproduced = returncode != 0 and "AssertionError" in stderr

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Reproduced: {self.reproduced}",
                "",
                "Stdout:",
                self.stdout,
                "",
                "Stderr:",
                self.stderr,
            ]
        )


@dataclass
class SearchResult:
    """Dataclass to hold search results."""

    """Dataclass to hold search results from code search tools.

    Attributes:
        file_path (str): Absolute path to the file containing the search result.
        start (int | None): Starting line number of the result (1-based). None if not applicable.
        end (int | None): Ending line number of the result (1-based). None if not applicable.
        class_name (str | None): Name of the class containing the result, if any.
        func_name (str | None): Name of the function/method containing the result, if any.
        code (str): The actual code snippet that matched the search.
    """
    file_path: str
    start: int | None
    end: int | None
    class_name: str | None
    func_name: str | None
    code: str

    def to_tagged_upto_file(self, project_root: str) -> str:
        """Converts the search result to a tagged string, up to the file path.

        Args:
            project_root (str): The root path of the project, used to make file_path relative.

        Returns:
            str: A string like "<file>relative/path/to/file.py</file>".
        """
        rel_path = apputils.to_relative_path(self.file_path, project_root)
        file_part = f"<file>{rel_path}</file>"
        return file_part

    def to_tagged_upto_class(self, project_root: str) -> str:
        """Converts the search result to a tagged string, up to the class name.

        Args:
            project_root (str): The root path of the project.

        Returns:
            str: String including file and class tags.
        """
        prefix = self.to_tagged_upto_file(project_root)
        class_part = (
            f"<class>{self.class_name}</class>" if self.class_name is not None else ""
        )
        return f"{prefix}\n{class_part}"

    def to_tagged_upto_func(self, project_root: str) -> str:
        """Converts the search result to a tagged string, up to the function name.

        Args:
            project_root (str): The root path of the project.

        Returns:
            str: String including file, class, and function tags.
        """
        prefix = self.to_tagged_upto_class(project_root)
        func_part = (
            f" <func>{self.func_name}</func>" if self.func_name is not None else ""
        )
        return f"{prefix}{func_part}"

    def to_tagged_str(self, project_root: str) -> str:
        """Converts the full search result to a tagged string including the code.

        Args:
            project_root (str): The root path of the project.

        Returns:
            str: A comprehensive tagged string representation of the search result.
        """
        prefix = self.to_tagged_upto_func(project_root)
        code_part = f"<code>\n{self.code}\n</code>"
        return f"{prefix}\n{code_part}"

    @staticmethod
    def collapse_to_file_level(lst: list[SearchResult], project_root: str) -> str:
        """Collapses a list of search results to a summary string grouped by file.

        Args:
            lst (list[SearchResult]): The list of search results.
            project_root (str): The root path of the project.

        Returns:
            str: A summary string with counts of matches per file.
        """
        res = dict()  # file -> count
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = 1
            else:
                res[r.file_path] += 1
        res_str = ""
        for file_path, count in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            res_str += f"- {file_part} ({count} matches)\n"
        return res_str

    @staticmethod
    def collapse_to_method_level(lst: list[SearchResult], project_root: str) -> str:
        """Collapses a list of search results to a summary string grouped by method/function.

        Args:
            lst (list[SearchResult]): The list of search results.
            project_root (str): The root path of the project.

        Returns:
            str: A summary string with counts of matches per method/function within files.
        """
        res = dict()  # file -> dict(method -> count)
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = dict()
            func_str = r.func_name if r.func_name is not None else "Not in a function"
            if func_str not in res[r.file_path]:
                res[r.file_path][func_str] = 1
            else:
                res[r.file_path][func_str] += 1
        res_str = ""
        for file_path, funcs in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            for func, count in funcs.items():
                if func == "Not in a function":
                    func_part = func
                else:
                    func_part = f" <func>{func}</func>"
                res_str += f"- {file_part}{func_part} ({count} matches)\n"
        return res_str


class BugLocation:
    """Represents an identified potential bug location in the codebase.

    This class encapsulates information about a specific code snippet that is
    suspected to be related to a bug, including its location, the code itself,
    and the intended behavior if it were fixed.

    Attributes:
        rel_file_path (str): Relative path to the file from the project root.
        abs_file_path (str): Absolute path to the file.
        start (int | None): Starting line number of the bug location (1-based).
        end (int | None): Ending line number of the bug location (1-based).
        class_name (str | None): Name of the class containing the bug, if any.
        method_name (str | None): Name of the method/function containing the bug, if any.
                                 (Note: referred to as func_name in SearchResult).
        code (str): The actual code snippet at the bug location.
        intended_behavior (str): A description of what the code *should* do if fixed.
    """
    rel_file_path: str
    abs_file_path: str
    start: int | None
    end: int | None
    class_name: str | None
    method_name: str | None # NOTE: from patch generation onwards, call this method_name
    code: str
    intended_behavior: str

    def __init__(
        self, search_res: SearchResult, project_path: str, intended_bebavior: str # Typo in intended_bebavior
    ):
        """Initializes a BugLocation from a SearchResult.

        Args:
            search_res (SearchResult): The search result pointing to the potential bug.
            project_path (str): The root path of the project.
            intended_bebavior (str): Description of the correct/intended behavior of this code.
        """
        assert search_res.start is not None
        assert search_res.end is not None

        # turn a search result into bug location
        self.abs_file_path = search_res.file_path
        self.rel_file_path = apputils.to_relative_path(
            search_res.file_path, project_path
        )

        self.start = search_res.start
        self.end = search_res.end

        self.class_name = search_res.class_name
        self.method_name = search_res.func_name # Align with SearchResult's func_name

        self.intended_behavior = intended_bebavior # Corrected typo from param

        # we know the line numbers are reliable, so just get the actual
        # code here again to be safe
        self.code = search_utils.get_code_snippets(
            self.abs_file_path, self.start, self.end
        )

    def to_dict(self):
        return {
            "rel_file_path": self.rel_file_path,
            "abs_file_path": self.abs_file_path,
            "start": self.start,
            "end": self.end,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "code": self.code,
            "intended_behavior": self.intended_behavior,
        }

    def __eq__(self, other: object) -> bool:
        """Checks equality based on file path, start, and end lines."""
        return (
            self.rel_file_path == other.rel_file_path
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.rel_file_path, self.start, self.end))

    def __str__(self):
        return (
            f"<file>{self.rel_file_path}</file>\n"
            f"<class>{self.class_name}</class>\n"
            f"<method>{self.method_name}</method>\n" # Consistent naming with attribute
            f"<code>\n{self.code}\n</code>"
            f"<intended_behavior>{self.intended_behavior}</intended_behavior>"
        )

    def __repr__(self) -> str:
        """Returns the string representation of the BugLocation."""
        return self.__str__()

    def to_str_for_model(self) -> str:
        """Returns a string representation suitable for including in LLM prompts."""
        return self.__str__()

    @classmethod
    def multiple_locs_to_str_for_model(cls, locs: list["BugLocation"]) -> str:
        """Formats a list of BugLocation objects into a single string for LLM prompts.

        Args:
            locs (list[BugLocation]): A list of BugLocation objects.

        Returns:
            str: A formatted string enumerating each bug location.
        """
        res = ""
        for idx, loc in enumerate(locs):
            actual_idx = idx + 1
            res += f"Location #{actual_idx}:\n"
            res += loc.to_str_for_model() + "\n\n"
        return res


@dataclass
class EvaluationPayload:
    """Represents the structured result of a patch evaluation.

    This data is typically sent from the backend evaluation process to a
    component that needs to act on the evaluation outcome, such as providing
    feedback to an LLM agent.

    Attributes:
        status (ResolvedStatus): The overall resolution status from `eval_helper.ResolvedStatus`.
        message (str): A human-readable summary message of the evaluation.
        details (dict[str, Any] | None, optional): A dictionary containing detailed
            breakdowns of the evaluation, such as pass/fail counts for different
            test categories (e.g., FAIL_TO_PASS, PASS_TO_PASS). Defaults to None.
    """
    status: ResolvedStatus
    message: str
    details: dict[str, Any] | None = None

    def to_llm_feedback_string(self) -> str:
        """Converts the evaluation payload into a string formatted for LLM feedback.

        This string typically includes the overall status, the main message, and
        a summary of test outcomes if details are available.

        Returns:
            str: A formatted string summarizing the evaluation result for an LLM.
        """
        detail_str = ""
        if self.details:
            detail_items = []
            # Example: Convert dict details to a readable string
            f2p_info = self.details.get("FAIL_TO_PASS", {})
            if isinstance(f2p_info, dict):
                f2p_s = len(f2p_info.get("success", []))
                f2p_f = len(f2p_info.get("failure", []))
                f2p_m = len(f2p_info.get("missing", []))
                if f2p_s > 0 or f2p_f > 0 or f2p_m > 0:
                    detail_items.append(f"  Fail-to-Pass: {f2p_s} succeeded, {f2p_f} failed, {f2p_m} missing.")

            p2p_info = self.details.get("PASS_TO_PASS", {})
            if isinstance(p2p_info, dict):
                p2p_s = len(p2p_info.get("success", []))
                p2p_f = len(p2p_info.get("failure", []))
                p2p_m = len(p2p_info.get("missing", []))
                if p2p_s > 0 or p2p_f > 0 or p2p_m > 0:
                    detail_items.append(f"  Pass-to-Pass: {p2p_s} succeeded, {p2p_f} failed, {p2p_m} missing.")

            if detail_items:
                detail_str = "\nTest outcome details:\n" + "\n".join(detail_items)
            elif self.details: # Fallback for other details not specifically formatted
                # Use json.dumps for a generic dict representation, ensure json is imported
                try:
                    detail_str = f"\nDetails: {json.dumps(self.details)}"
                except NameError: # json not imported. Should not happen as json is imported at top level.
                    detail_str = f"\nDetails: {str(self.details)}" # Fallback

        return f"Patch evaluation result: {self.status.value}. {self.message}{detail_str}"
