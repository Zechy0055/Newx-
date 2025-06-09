"""
Common stuff for task agents.
"""

from app.data_structures import MessageThread


def replace_system_prompt(msg_thread: MessageThread, prompt: str) -> MessageThread:
    """
    Replace the system prompt in the message thread.
    This is because the main agent system prompt main invole tool_calls info, which
    should not be known to task agents.
    """
    msg_thread.messages[0]["content"] = prompt
    return msg_thread


from enum import Enum

class LLMErrorCode(Enum):
    RATE_LIMIT = "RATE_LIMIT"
    CONTENT_FILTER = "CONTENT_FILTER"
    BAD_FORMAT = "BAD_FORMAT"  # Output JSON or structure is not as expected
    API_ERROR = "API_ERROR"    # General API error from the LLM provider
    OTHER = "OTHER"          # Other LLM related errors


class InvalidLLMResponse(RuntimeError):
    def __init__(self, message: str, error_code: LLMErrorCode | None = None, detail: str | None = None):
        super().__init__(message)
        self.message = message # The original message is stored in self.args[0]
        self.error_code = error_code
        self.detail = detail

    def __str__(self):
        base_str = f"InvalidLLMResponse: {self.message}"
        if self.error_code:
            base_str += f" (Error Code: {self.error_code.value})"
        if self.detail:
            base_str += f" (Detail: {self.detail})"
        return base_str
