"""
Node functions for LangGraph that perform Git operations.

This module includes functions for cloning repositories, checking out branches/commits,
applying patches, committing changes, and creating pull requests, all designed to
be used as nodes in a LangGraph agent workflow.
"""

import os
import subprocess
import tempfile
import shutil # For cleaning up directories if needed
from typing import cast # For type casting if needed with AgentState

from loguru import logger

from app.data_structures import AgentState

# Define a base working directory for clones, if desired
# This could also be configured externally.
BASE_WORKING_DIR = "workspace/git_clones"

def clone_repo(state: AgentState) -> AgentState:
    """
    Clones a Git repository based on the `repo_url` provided in the agent state.

    If `repo_url` is not found in the state, or if cloning fails, an error message
    is set in the state. On success, `local_repo_path` is updated in the state,
    and a success message is logged to `state["log_messages"]`.

    Args:
        state (AgentState): The current state of the LangGraph agent.
                           Expected to contain `repo_url`.

    Returns:
        AgentState: The updated state with `local_repo_path` and `log_messages` modified,
                    or an `error_message` set on failure.
    """
    task_id = state.get("task_id", "unknown_task")
    bound_logger = logger.bind(task_id=task_id, function="clone_repo")

    repo_url = state.get("repo_url")
    if not repo_url:
        error_msg = "Repository URL (`repo_url`) not found in state."
        bound_logger.error(error_msg)
        state["error_message"] = error_msg
        state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
        return state

    try:
        # Create a unique directory for this clone based on task_id or a UUID
        # Using a combination of BASE_WORKING_DIR and a task-specific name
        os.makedirs(BASE_WORKING_DIR, exist_ok=True)
        # Sanitize repo_url to create a friendlier directory name if needed, or use task_id
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        clone_target_dir_name = f"{repo_name}_{task_id}"
        # Ensure this is unique if multiple runs for same task_id can happen concurrently (use tempfile.mkdtemp for that)
        # For simplicity, assuming task_id provides enough uniqueness for now or it's handled by orchestration.
        # A safer approach for true isolation:
        # clone_dir = tempfile.mkdtemp(prefix=f"{repo_name}_", dir=BASE_WORKING_DIR)

        # More controlled naming:
        clone_dir_path = os.path.join(BASE_WORKING_DIR, clone_target_dir_name)
        if os.path.exists(clone_dir_path):
            bound_logger.warning(f"Clone directory {clone_dir_path} already exists. Attempting to remove and re-clone.")
            shutil.rmtree(clone_dir_path) # Remove existing directory to ensure fresh clone

        os.makedirs(clone_dir_path) # Recreate after potential removal

        bound_logger.info(f"Cloning repository from {repo_url} into {clone_dir_path}...")

        # Execute git clone
        process = subprocess.run(
            ["git", "clone", repo_url, clone_dir_path],
            capture_output=True,
            text=True,
            check=False # Don't raise exception on non-zero exit, handle manually
        )

        if process.returncode == 0:
            success_msg = f"Successfully cloned repository {repo_url} to {clone_dir_path}."
            bound_logger.info(success_msg)
            state["local_repo_path"] = clone_dir_path
            state.setdefault("log_messages", []).append(success_msg)
            state["error_message"] = None # Clear any previous error
        else:
            error_msg = f"Failed to clone repository {repo_url}. Error: {process.stderr}"
            bound_logger.error(error_msg)
            state["error_message"] = error_msg
            state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
            state["local_repo_path"] = None # Ensure path is None on failure
            # Optionally, clean up the failed clone directory
            if os.path.exists(clone_dir_path) and not os.listdir(clone_dir_path): # Only remove if empty
                 os.rmdir(clone_dir_path)
            elif os.path.exists(clone_dir_path): # If not empty, something went wrong, might keep for inspection
                 bound_logger.warning(f"Failed clone directory {clone_dir_path} is not empty. Not removing automatically.")


    except Exception as e:
        error_msg = f"An unexpected error occurred during cloning: {str(e)}"
        bound_logger.exception(error_msg) # Logs with stack trace
        state["error_message"] = error_msg
        state.setdefault("log_messages", []).append(f"ERROR: {error_msg}")
        state["local_repo_path"] = None

    return state

if __name__ == '__main__':
    # Example Usage (for testing this node standalone)

    # Ensure the logger is configured to see output if running this directly
    # from loguru import logger
    # import sys
    # logger.remove()
    # logger.add(sys.stderr, level="DEBUG")

    print("Testing clone_repo node...")

    # Test case 1: Successful clone
    test_state_success = AgentState(
        task_id="test_clone_001",
        repo_url="https://github.com/loguru/loguru.git", # A real, public repo
        log_messages=[],
        selected_model="test_model", # Fulfilling Required fields from AgentState for testing
        github_token_available=False
    )

    print(f"\nInitial state for success test: {test_state_success}")
    updated_state_success = clone_repo(test_state_success)
    print(f"Updated state after success test: {updated_state_success}")

    if updated_state_success.get("local_repo_path") and os.path.exists(cast(str, updated_state_success.get("local_repo_path"))):
        print(f"Repository cloned to: {updated_state_success.get('local_repo_path')}")
        # Clean up the cloned repo
        try:
            shutil.rmtree(cast(str, updated_state_success.get("local_repo_path")))
            print(f"Cleaned up: {updated_state_success.get('local_repo_path')}")
        except Exception as e:
            print(f"Error cleaning up {updated_state_success.get('local_repo_path')}: {e}")
    else:
        print(f"Cloning failed or local_repo_path not set. Error: {updated_state_success.get('error_message')}")


    # Test case 2: Missing repo_url
    test_state_no_url = AgentState(
        task_id="test_clone_002",
        log_messages=[],
        selected_model="test_model",
        github_token_available=False
    ) # repo_url is missing
    print(f"\nInitial state for missing URL test: {test_state_no_url}")
    updated_state_no_url = clone_repo(test_state_no_url)
    print(f"Updated state after missing URL test: {updated_state_no_url}")
    if updated_state_no_url.get("error_message"):
        print(f"Correctly handled missing repo_url: {updated_state_no_url.get('error_message')}")
    else:
        print("Missing repo_url test failed to set an error message.")

    # Test case 3: Invalid repo_url
    test_state_invalid_url = AgentState(
        task_id="test_clone_003",
        repo_url="https://invalid.url/that/does/not/exist.git",
        log_messages=[],
        selected_model="test_model",
        github_token_available=False
    )
    print(f"\nInitial state for invalid URL test: {test_state_invalid_url}")
    updated_state_invalid_url = clone_repo(test_state_invalid_url)
    print(f"Updated state after invalid URL test: {updated_state_invalid_url}")
    if updated_state_invalid_url.get("error_message"):
        print(f"Correctly handled invalid repo_url: {updated_state_invalid_url.get('error_message')}")
    else:
        print("Invalid repo_url test failed to set an error message.")

    # Clean up base working directory if it was created and is empty
    if os.path.exists(BASE_WORKING_DIR) and not os.listdir(BASE_WORKING_DIR):
        os.rmdir(BASE_WORKING_DIR)
        print(f"Cleaned up base working directory: {BASE_WORKING_DIR}")
    elif os.path.exists(BASE_WORKING_DIR):
         print(f"Base working directory {BASE_WORKING_DIR} not empty, not removing.")


    # Example of how to ensure AgentState required fields are present for type checking during dev
    # This is more for dev time, not runtime if total=False
    # try:
    #     # This would fail type checking if task_id was not provided and Required was enforced by a static checker
    #     # that understands Required with total=False (like Pydantic would enforce).
    #     # TypedDict with total=False doesn't strictly enforce at runtime before assignment.
    #     s: AgentState = {"repo_url": "test"} # type: ignore
    # except TypeError as e:
    #     print(f"Error creating state: {e}") # This won't typically raise TypeError for TypedDict
    pass
