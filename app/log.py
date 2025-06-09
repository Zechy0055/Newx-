"""Handles logging and console output for the AutoCodeRover application.

This module provides utility functions for:
- Richly formatted console output using `rich` library (e.g., banners, panels for different agents).
- Structured logging to files (including JSON format) using `loguru`.
- Conditional printing to stdout based on a global flag (`print_stdout`).
- Helper functions for consistently logging messages that also need console visibility.
"""
import time
from os import get_terminal_size
from typing import Any # For kwargs in log_and_cprint

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def terminal_width() -> int:
    """Determines the current terminal width.

    Returns:
        int: The width of the terminal in columns, or 80 if unable to determine.
    """
    try:
        return get_terminal_size().columns
    except OSError:
        return 80


WIDTH: int = min(120, terminal_width() - 10)
"""int: Standard width for console panels, adjusted for terminal size."""

console: Console = Console()
"""rich.console.Console: Global Rich console instance for styled output."""

print_stdout: bool = True
"""bool: Global flag to enable or disable printing to stdout for most custom print functions.
   Logging to file still occurs independently of this flag.
"""


def log_exception(exception: Exception) -> None:
    """Logs an exception with its full stack trace using logger.exception.

    Args:
        exception (Exception): The exception to log.
    """
    logger.exception(exception)


def print_banner(msg: str) -> None:
    """Prints a centered banner message to the console with a specific style.

    Output is conditional on `print_stdout`. Does not log to file by default.

    Args:
        msg (str): The message to display in the banner.
    """
    if not print_stdout:
        return

    banner = f" {msg} ".center(WIDTH, "=")
    console.print()
    console.print(banner, style="bold")
    console.print()


def replace_html_tags(content: str):
    """
    Helper method to process the content before printing to markdown.

    Replaces pseudo-HTML tags like <file> with markdown-friendly syntax like [file]
    to prevent issues with Markdown rendering while keeping tags visually distinct.

    Args:
        content (str): The input string content.

    Returns:
        str: The processed string with tags replaced.
    """
    replace_dict = {
        "<file>": "[file]",
        "<class>": "[class]",
        "<func>": "[func]",
        "<method>": "[method]",
        "<code>": "[code]",
        "<original>": "[original]",
        "<patched>": "[patched]",
        "</file>": "[/file]",
        "</class>": "[/class]",
        "</func>": "[/func]",
        "</method>": "[/method]",
        "</code>": "[/code]",
        "</original>": "[/original]",
        "</patched>": "[/patched]",
    }
    for key, value in replace_dict.items():
        content = content.replace(key, value)
    return content


def print_acr(msg: str, desc="") -> None:
    if not print_stdout:
        return

    msg = replace_html_tags(msg)
    markdown = Markdown(msg)

    name = "AutoCodeRover"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="magenta",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "acr_message", "description": desc, "details": msg, "console_printed": print_stdout})


def print_retrieval(msg: str, desc: str = "") -> None:
    """Prints a message from the Context Retrieval Agent in a styled panel.

    Also logs the message as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message content, potentially containing pseudo-HTML tags.
        desc (str, optional): A description or subtype for the message. Defaults to "".
    """
    if not print_stdout:
        logger.debug({"type": "retrieval_message", "description": desc, "details": msg, "stdout_skipped": True})
        return

    msg = replace_html_tags(msg)
    markdown = Markdown(msg)

    name = "Context Retrieval Agent"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="blue",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "retrieval_message", "description": desc, "details": msg, "console_printed": print_stdout})


def print_patch_generation(msg: str, desc: str = "") -> None:
    """Prints a message related to Patch Generation in a styled panel.

    Also logs the message as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message content, potentially containing pseudo-HTML tags.
        desc (str, optional): A description or subtype for the message. Defaults to "".
    """
    if not print_stdout:
        logger.debug({"type": "patch_generation_message", "description": desc, "details": msg, "stdout_skipped": True})
        return

    msg = replace_html_tags(msg)
    markdown = Markdown(msg)

    name = "Patch Generation"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="yellow",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "patch_generation_message", "description": desc, "details": msg, "console_printed": print_stdout})


def print_issue(content: str) -> None:
    """Prints the issue description in a styled panel.

    Also logs a summary of the issue content as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        content (str): The full text of the issue description.
    """
    if not print_stdout:
        logger.debug({"type": "issue_description", "content_summary": content[:200], "stdout_skipped": True})
        return

    title = "Issue description"
    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style="red",
    )
    console.print(panel)
    logger.info({"type": "issue_description", "content_summary": content[:200], "console_printed": print_stdout})


def print_reproducer(msg: str, desc: str = "") -> None:
    """Prints a message related to Reproducer Test Generation in a styled panel.

    Also logs the message as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message content.
        desc (str, optional): A description or subtype for the message. Defaults to "".
    """
    if not print_stdout:
        logger.debug({"type": "reproducer_generation_message", "description": desc, "details": msg, "stdout_skipped": True})
        return

    markdown = Markdown(msg)

    name = "Reproducer Test Generation"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="green",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "reproducer_generation_message", "description": desc, "details": msg, "console_printed": print_stdout})


def print_exec_reproducer(msg: str, desc: str = "") -> None:
    """Prints a message related to Reproducer Execution Result in a styled panel.

    Also logs the message as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message content.
        desc (str, optional): A description or subtype for the message. Defaults to "".
    """
    if not print_stdout:
        logger.debug({"type": "reproducer_execution_message", "description": desc, "details": msg, "stdout_skipped": True})
        return

    markdown = Markdown(msg)

    name = "Reproducer Execution Result"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="blue",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "reproducer_execution_message", "description": desc, "details": msg, "console_printed": print_stdout})


def print_review(msg: str, desc: str = "") -> None:
    """Prints a message related to a Review in a styled panel.

    Also logs the message as a structured JSON entry.
    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message content.
        desc (str, optional): A description or subtype for the message. Defaults to "".
    """
    if not print_stdout:
        logger.debug({"type": "review_message", "description": desc, "details": msg, "stdout_skipped": True})
        return

    markdown = Markdown(msg)

    name = "Review"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="purple",
        width=WIDTH,
    )
    console.print(panel)
    logger.info({"type": "review_message", "description": desc, "details": msg, "console_printed": print_stdout})


def log_and_print(msg: str) -> None:
    """Logs a message using logger.info and prints it to console if print_stdout is True.

    Args:
        msg (str): The message to log and print.
    """
    logger.info(msg)
    if print_stdout:
        console.print(msg)


def log_and_cprint(msg: str, **kwargs: Any) -> None:
    """Logs a message using logger.info and prints it to console with Rich styling options.

    Console output is conditional on `print_stdout`.

    Args:
        msg (str): The message to log and print.
        **kwargs: Additional keyword arguments to pass to `rich.console.Console.print()`.
    """
    logger.info(msg)
    if print_stdout:
        console.print(msg, **kwargs)


def log_and_always_print(msg: str) -> None:
    """Logs a message using logger.info and always prints it to console with a timestamp.

    This function ignores the `print_stdout` flag for console output.
    Useful for critical messages that should always be visible during batch runs.

    Args:
        msg (str): The message to log and print.
    """
    logger.info(msg)
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    console.print(f"\n[{t}] {msg}")


def print_with_time(msg: str) -> None:
    """Prints a message to console with a timestamp. Does not log to file.

    Args:
        msg (str): The message to print.
    """
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    console.print(f"\n[{t}] {msg}")
