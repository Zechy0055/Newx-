"""
Node functions for LangGraph that perform repository analysis.

This module includes functions to:
- Analyze the structure of a cloned repository (list files and directories).
- Analyze the content of individual files using an LLM.
- Aggregate these analyses into a repository-level summary.
"""
import os
from typing import List, Dict, Any, Set

from loguru import logger

from app.data_structures import AgentState
from app.model.common import Model, SELECTED_MODEL # Assuming SELECTED_MODEL is set appropriately

# Common directories and file extensions to ignore during structure analysis
DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".vscode", ".idea", "dist", "build",
    "venv", ".venv", "env", ".env", "target", "out", "bin", "obj",
    ".DS_Store",
}
DEFAULT_IGNORE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".o", ".a", ".lib", ".dll", ".exe", ".class",
    ".jar", ".war", ".ear", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".DS_Store", ".db", ".sqlite", ".sqlite3",
    ".log", ".tmp", ".bak", ".swp",
    # Common image/media types if not specifically targeted for analysis
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",
    ".mp3", ".wav", ".ogg", ".mp4", ".mov", ".avi", ".wmv", ".mkv",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}


def analyze_repo_structure(repo_path: str,
                           ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
                           ignore_extensions: Set[str] = DEFAULT_IGNORE_EXTENSIONS) -> Dict[str, Any]:
    """
    Analyzes the directory structure of a repository, listing files and directories.

    Ignores common unnecessary directories (e.g., .git) and file extensions.

    Args:
        repo_path (str): The absolute path to the cloned repository.
        ignore_dirs (Set[str]): A set of directory names to ignore.
        ignore_extensions (Set[str]): A set of file extensions to ignore.

    Returns:
        Dict[str, Any]: A dictionary representing the repository structure (e.g.,
                        a tree or a flat list of relevant file paths).
                        For now, returns a dict with "files" and "directories" lists.
    """
    repo_map: Dict[str, Any] = {"files": [], "directories": []}

    if not os.path.isdir(repo_path):
        logger.error(f"Repository path does not exist or is not a directory: {repo_path}")
        return {"error": "Repository path not found.", "files": [], "directories": []}

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Modify dirs in-place to exclude ignored directories from further traversal
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for name in files:
            if not any(name.endswith(ext) for ext in ignore_extensions):
                full_path = os.path.join(root, name)
                relative_path = os.path.relpath(full_path, repo_path)
                repo_map["files"].append(relative_path)

        for name in dirs: # Only non-ignored dirs are here
            full_path = os.path.join(root, name)
            relative_path = os.path.relpath(full_path, repo_path)
            repo_map["directories"].append(relative_path)

    # Sort for consistent output
    repo_map["files"].sort()
    repo_map["directories"].sort()

    logger.info(f"Analyzed structure: found {len(repo_map['files'])} relevant files and {len(repo_map['directories'])} directories.")
    return repo_map


def analyze_file_content(file_path: str, project_base_path: str, model: Model) -> Dict[str, Any]:
    """
    Analyzes the content of a single file using the provided LLM.

    Reads the file content, constructs a prompt for analysis (identifying smells,
    TODOs, refactoring areas, and summary), calls the LLM, and parses its response.

    Args:
        file_path (str): The relative path of the file within the repository.
        project_base_path (str): The absolute base path of the cloned repository.
        model (Model): The language model instance to use for analysis.

    Returns:
        Dict[str, Any]: A structured dictionary containing the analysis results for the file
                        (e.g., {"file_path": str, "summary": "...", "findings": [...]}),
                        or an error structure if analysis fails.
    """
    full_file_path = os.path.join(project_base_path, file_path)
    analysis_result: Dict[str, Any] = {
        "file_path": file_path,
        "summary": "",
        "findings": [],
        "error": None
    }

    try:
        with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Limit content size to avoid oversized prompts (e.g., first/last N lines or total char limit)
            # This is a very basic truncation, more sophisticated chunking might be needed for large files.
            max_chars = 20000 # Example limit (approx 5k tokens)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [TRUNCATED] ..."
                logger.warning(f"File {file_path} was truncated for LLM analysis due to size.")


        # TODO: Check if file is binary or text more reliably before reading.
        # For now, simple text read with error handling.

        prompt = (
            f"Analyze the following code from file '{file_path}'. "
            "Identify potential code smells, TODO comments, areas for refactoring, "
            "and provide a brief one-sentence summary of its main purpose. "
            "Format your findings clearly. If the file is not a source code file (e.g. data, config), just provide a summary. "
            "If providing findings, try to include line numbers where relevant.\n\n"
            "Example finding format:\n"
            "- Type: Code Smell\n  Description: Long function 'xyz'.\n  Line: 42\n"
            "- Type: TODO\n  Description: Add error handling for edge case.\n  Line: 101\n\n"
            "File Content:\n"
            "```\n"
            f"{content}\n"
            "```\n\n"
            "Your analysis:"
        )

        messages = [{"role": "user", "content": prompt}]

        # This is a placeholder call. The actual implementation depends on the Model interface
        # and how Qwen (or any selected model) expects input and provides output.
        # Assuming model.call() returns a string response.
        # response_text, cost, input_tokens, output_tokens = model.call(messages)

        # For now, as QwenModel's call method isn't implemented, we'll use a placeholder.
        # In a real scenario, this would be:
        # response_text, _, _, _ = model.call(messages=messages) # Assuming model is SELECTED_MODEL or passed in
        # For testing purposes:
        logger.debug(f"Sending content of {file_path} to LLM for analysis (model: {model.name}). First 100 chars: {content[:100]}")
        # This will raise NotImplementedError if SELECTED_MODEL is QwenModel and call isn't implemented.
        # Let's assume a dummy response for now until QwenModel.call() is functional.
        # response_text = f"Placeholder analysis for {file_path}: Summary - It does stuff. Findings - TODO: Implement real analysis."

        # To make this runnable without implementing QwenModel.call yet:
        if model.name.startswith("together_ai/qwen") or model.name.startswith("qwen"): # or any other Qwen identifier
            logger.warning(f"Using placeholder analysis for Qwen model {model.name} as call() might not be fully implemented for it yet.")
            response_text = f"Placeholder LLM analysis for {file_path}:\nSummary: This file seems to define some functionality. Needs real LLM analysis.\n- Type: TODO\n  Description: Implement actual LLM call in analyze_file_content.\n  Line: 0"
        else:
            # Use actual model call if not Qwen or if Qwen is expected to work via LiteLLM
            try:
                response_text, _, _, _ = model.call(messages=messages)
            except Exception as e:
                logger.error(f"LLM call failed for file {file_path}: {e}", exc_info=True)
                analysis_result["error"] = f"LLM call failed: {str(e)}"
                return analysis_result


        # Basic parsing of a hypothetical structured response (needs refinement based on actual LLM output format)
        # For now, let's assume the LLM tries to follow the requested format or provides a readable text.
        analysis_result["summary"] = f"LLM Analysis for {file_path}: {response_text[:150]}..." # Placeholder summary
        # A more robust parser for the "Type: ... Description: ... Line: ..." format would be needed here.
        # For this placeholder, we'll just put the whole response as a single "finding".
        analysis_result["findings"].append({
            "type": "general_analysis",
            "description": response_text,
            "line": None
        })
        logger.info(f"Successfully analyzed file: {file_path} with model {model.name}")

    except FileNotFoundError:
        error_msg = f"File not found: {full_file_path}"
        logger.error(error_msg)
        analysis_result["error"] = error_msg
    except Exception as e:
        error_msg = f"Error analyzing file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        analysis_result["error"] = error_msg

    return analysis_result


def analyze_repo(state: AgentState) -> AgentState:
    """
    Performs analysis of the cloned repository.

    This involves:
    1. Analyzing the repository structure (files and directories).
    2. For a subset of files, analyzing their content using an LLM.
    3. (Future) Generating an overall repository summary based on individual analyses.

    Updates `state["analysis_results"]` with the findings. Sets `error_message`
    if critical failures occur.

    Args:
        state (AgentState): The current agent state, expected to contain `local_repo_path`
                           and `selected_model` name.

    Returns:
        AgentState: The updated state.
    """
    task_id = state.get("task_id", "unknown_task")
    bound_logger = logger.bind(task_id=task_id, function="analyze_repo")

    local_repo_path = state.get("local_repo_path")
    selected_model_name = state.get("selected_model") # Name of the model

    if not local_repo_path or not os.path.isdir(local_repo_path):
        error_msg = "Local repository path not set or invalid."
        bound_logger.error(error_msg)
        state["error_message"] = error_msg
        state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
        return state

    if not selected_model_name:
        error_msg = "Selected model name not found in state."
        bound_logger.error(error_msg)
        state["error_message"] = error_msg
        state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
        return state

    # Get the actual model instance from SELECTED_MODEL (assuming it's set correctly)
    # This relies on app.model.common.SELECTED_MODEL being the one to use.
    # If a different model instance is needed, the logic to fetch it would be different.
    active_model = SELECTED_MODEL
    if active_model.name != selected_model_name:
        # This case should ideally not happen if set_model was called correctly prior to this graph.
        # Or, we might need to fetch from MODEL_HUB if state["selected_model"] is just a name.
        # For now, assume SELECTED_MODEL is the one to use.
        bound_logger.warning(f"State's selected_model '{selected_model_name}' differs from globally SELECTED_MODEL '{active_model.name}'. Using globally selected one.")
        # Or, if we must use the one from state:
        # from app.model.common import MODEL_HUB
        # if selected_model_name not in MODEL_HUB:
        #     # handle error, model not registered
        # active_model = MODEL_HUB[selected_model_name]


    bound_logger.info(f"Starting repository analysis for {local_repo_path} using model {active_model.name}.")
    state.setdefault("log_messages", []).append(f"Starting repository analysis with {active_model.name}.")

    analysis_summary: Dict[str, Any] = {
        "file_analyses": [],
        "overall_summary": "Overall summary not yet implemented.", # Placeholder
        "repo_map": {}
    }

    try:
        # 1. Analyze repository structure
        repo_structure = analyze_repo_structure(local_repo_path)
        analysis_summary["repo_map"] = repo_structure
        if repo_structure.get("error"):
            bound_logger.error(f"Failed to analyze repo structure: {repo_structure.get('error')}")
            # Continue with empty file list if structure analysis fails partially

        # 2. Analyze content of a subset of files
        files_to_analyze = repo_structure.get("files", [])
        # Limit the number of files for now to keep it manageable (e.g., first 5 Python files)
        # This selection logic should be more sophisticated in a real scenario.
        python_files = [f for f in files_to_analyze if f.endswith(".py")]
        subset_files_to_analyze = python_files[:5] # Analyze first 5 Python files

        bound_logger.info(f"Identified {len(files_to_analyze)} total relevant files. Analyzing content of {len(subset_files_to_analyze)} files: {subset_files_to_analyze}")

        for rel_file_path in subset_files_to_analyze:
            bound_logger.debug(f"Analyzing file: {rel_file_path}")
            file_analysis = analyze_file_content(rel_file_path, local_repo_path, active_model)
            analysis_summary["file_analyses"].append(file_analysis)
            if file_analysis.get("error"):
                bound_logger.warning(f"Could not fully analyze {rel_file_path}: {file_analysis.get('error')}")

        # 3. (Optional) Overall Repo Summary (placeholder for now)
        if analysis_summary["file_analyses"]:
            # Potentially make another LLM call with summaries of file_analyses and repo_map
            # to generate an overall_summary.
            # For now, just a note.
            analysis_summary["overall_summary"] = "Overall repository summary generation is pending implementation."
            bound_logger.info("Individual file analyses completed. Overall summary generation pending.")
        else:
            analysis_summary["overall_summary"] = "No files were selected for content analysis."
            bound_logger.info("No files analyzed for content.")


        state["analysis_results"] = analysis_summary
        success_msg = f"Repository analysis completed. Analyzed structure and content of {len(subset_files_to_analyze)} files."
        bound_logger.info(success_msg)
        state.setdefault("log_messages", []).append(success_msg)
        state["error_message"] = None

    except Exception as e:
        error_msg = f"An unexpected error occurred during repository analysis: {str(e)}"
        bound_logger.exception(error_msg)
        state["error_message"] = error_msg
        state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
        state["analysis_results"] = None # Ensure results are None on error

    return state


if __name__ == '__main__':
    # Example Usage (for testing this node standalone)
    # This requires a model to be set in app.model.common.SELECTED_MODEL
    # and potentially API keys for that model.
    # For a self-contained test, you might mock SELECTED_MODEL.

    # Configure logger for direct script run
    # import sys
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")

    print("Testing analyze_repo node...")

    # Create a dummy repo for testing
    test_repo_dir_obj = tempfile.TemporaryDirectory()
    test_repo_dir = test_repo_dir_obj.name

    (Path(test_repo_dir) / "file1.py").write_text("def hello():\n  print('Hello')\n# TODO: Add more features")
    (Path(test_repo_dir) / "file2.txt").write_text("Some text data.")
    (Path(test_repo_dir) / ".git").mkdir() # Ignored dir
    (Path(test_repo_dir) / "file3.pyc").write_text("compiled") # Ignored extension

    # Mock SELECTED_MODEL for this test if it's not configured globally
    class MockModel:
        def __init__(self, name="mock_model"):
            self.name = name
        def call(self, messages: list[dict], **kwargs) -> tuple[str | None, dict | None, dict | None, dict | None]:
            prompt_content = messages[0]["content"]
            return f"Mocked analysis for: {prompt_content[:100]}...", None, None, None # type: ignore

    original_selected_model = SELECTED_MODEL
    SELECTED_MODEL = cast(Model, MockModel()) # Use cast if types mismatch for test

    test_state = AgentState(
        task_id="test_analysis_001",
        local_repo_path=test_repo_dir,
        selected_model=SELECTED_MODEL.name, # Use the name of the mocked model
        log_messages=[],
        # Fill other required fields for AgentState if any were made strictly Required
        repo_url="dummy", # Not used by analyze_repo directly but good for state completeness
        github_token_available=False
    )

    print(f"\nInitial state for analysis test: {test_state}")
    updated_state = analyze_repo(test_state)
    print(f"Updated state after analysis test: {json.dumps(updated_state, indent=2, default=str)}")

    if updated_state.get("analysis_results"):
        print("\nAnalysis Results:")
        print(f"  Repo Map: {json.dumps(updated_state['analysis_results']['repo_map'], indent=2)}")
        print(f"  File Analyses ({len(updated_state['analysis_results']['file_analyses'])}):")
        for fa in updated_state['analysis_results']['file_analyses']:
            print(f"    - {fa['file_path']}: Summary: {fa['summary'][:50]}... Error: {fa['error']}")
        print(f"  Overall Summary: {updated_state['analysis_results']['overall_summary']}")

    if updated_state.get("error_message"):
        print(f"Error during analysis: {updated_state.get('error_message')}")

    # Clean up
    test_repo_dir_obj.cleanup()
    SELECTED_MODEL = original_selected_model # Restore original
    print("\nTest finished.")
    pass
