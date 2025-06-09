"""Global configuration variables for the AutoCodeRover application.

This module defines various settings that control the behavior of the application,
including model selection, feature flags for debugging and validation, and
operational parameters like retry limits and timeouts.
"""

output_dir: str = ""
"""str: The root directory where all experiment results, logs, and patches are saved."""

overall_retry_limit: int = 3
"""int: Maximum number of times the main workflow (including context retrieval, patch generation, etc.) is retried for a task."""

conv_round_limit: int = 15
"""int: Upper bound for the number of conversation rounds allowed between the user/system and the LLM agent."""

enable_sbfl: bool = False
"""bool: Flag to enable Spectrum-Based Fault Localization (SBFL)."""

enable_validation: bool = False
"""bool: Flag to enable the validation of generated patches against the project's test suite."""

enable_angelic: bool = False
"""bool: Flag to enable experimental angelic debugging, which may use heuristics or weaker checks."""

enable_perfect_angelic: bool = False
"""bool: Flag to enable experimental perfect angelic debugging, which typically compares generated patches against developer patches. Overrides `enable_angelic` if both are true."""

only_save_sbfl_result: bool = False
"""bool: If True, the application will only run SBFL, save its results, and then exit. Useful for pre-calculating SBFL data."""

only_reproduce: bool = False
"""bool: If True, the application will only attempt to generate a reproducer test for the issue and then exit."""

only_eval_reproducer: bool = False
"""bool: If True, the application will only evaluate an existing reproducer test and then exit."""

reproduce_and_review: bool = False
"""bool: Experimental flag to integrate reproducer generation and patch review into the main workflow."""

test_exec_timeout: int = 300
"""int: Timeout in seconds for executing test commands (e.g., running a project's test suite). Default is 5 minutes."""

models: list[str] = []
"""list[str]: A list of primary model names to be used for generation tasks. The application may cycle through these if retries occur."""

backup_model: list[str] = ["gpt-4o-2024-05-13"]
"""list[str]: A list of backup model names to be used if primary models fail (e.g., due to content policy violations)."""

disable_angelic: bool = False
"""bool: (Potentially deprecated or conflicting, see `enable_angelic` and `enable_perfect_angelic`) General flag to disable angelic debugging features."""
