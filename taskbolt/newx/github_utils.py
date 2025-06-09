import os
import shutil
from git import Repo, GitCommandError # type: ignore[import-untyped]
import logging

logger = logging.getLogger(__name__)
# Basic config for the logger if this module is run directly or imported by a non-configured app
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

CLONED_REPOS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloned_repos")
os.makedirs(CLONED_REPOS_BASE_DIR, exist_ok=True)

def clone_repo(repo_url: str, task_id: str) -> str | None:
    repo_local_path = os.path.join(CLONED_REPOS_BASE_DIR, task_id)
    if os.path.exists(repo_local_path):
        logger.warning(f"Existing directory found at {repo_local_path} for task {task_id}. Removing it before cloning.")
        try:
            shutil.rmtree(repo_local_path)
        except Exception as e:
            logger.error(f"Failed to remove existing directory {repo_local_path} for task {task_id}: {e}")
            return None
    logger.info(f"Attempting to clone {repo_url} to {repo_local_path} for task {task_id}...")
    try:
        cloned_repo = Repo.clone_from(repo_url, repo_local_path)
        logger.info(f"Successfully cloned {repo_url} to {repo_local_path} for task {task_id}. Repo head: {cloned_repo.head.commit.hexsha}")
        return repo_local_path
    except GitCommandError as e:
        # Ensure stderr is decoded if it's bytes, common in GitPython exceptions
        stderr_decoded = e.stderr.decode('utf-8', 'replace') if isinstance(e.stderr, bytes) else str(e.stderr)
        error_details = f"GitCommandError: {' '.join(e.command)} - Status {e.status}\nStderr: {stderr_decoded.strip()}"
        logger.error(f"Git clone failed for {repo_url} (task {task_id}). Details: {error_details}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during clone of {repo_url} (task {task_id}): {e}", exc_info=True)
        return None

def cleanup_repo(task_id: str) -> bool:
    repo_local_path = os.path.join(CLONED_REPOS_BASE_DIR, task_id)
    if os.path.exists(repo_local_path):
        logger.info(f"Cleaning up repository at {repo_local_path} for task {task_id}.")
        try:
            shutil.rmtree(repo_local_path)
            logger.info(f"Successfully removed {repo_local_path} for task {task_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to remove directory {repo_local_path} for task {task_id} during cleanup: {e}", exc_info=True)
            return False
    else:
        logger.info(f"No directory found at {repo_local_path} for cleanup (task {task_id}). Assumed already cleaned or never cloned.")
        return True

def commit_and_push_changes(repo_local_path: str, commit_message: str, branch_name: str, remote_name: str = 'origin') -> bool:
    logger.info(f"(Placeholder) Simulating commit and push for {repo_local_path} on branch {branch_name} with message: '{commit_message}'")
    # In a real scenario:
    # repo = Repo(repo_local_path)
    # repo.git.add(update=True) # Stage all changes
    # repo.index.commit(commit_message)
    # origin = repo.remote(name=remote_name)
    # origin.push(branch_name)
    return True

def create_pull_request(repo_local_path: str, title: str, body: str, head_branch: str, base_branch: str = 'main') -> str | None:
    logger.info(f"(Placeholder) Simulating PR creation for {repo_local_path}: Title '{title}', head '{head_branch}', base '{base_branch}'")
    return f"http://github.com/example/placeholder/pull/new/{head_branch}"

if __name__ == "__main__":
    # This block will only run when the script is executed directly.
    # Ensure logger is configured for direct script execution
    if not logger.handlers:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logger.setLevel(logging.DEBUG) # Ensure existing logger is at least DEBUG for this test

    test_repo_url = "https://github.com/gitpython-developers/GitPython.git"
    test_task_id_clone = "direct_script_test_clone_005"
    logger.info(f"--- Starting direct test for github_utils.py with task_id: {test_task_id_clone} ---")
    cloned_path = clone_repo(test_repo_url, test_task_id_clone)
    if cloned_path:
        logger.info(f"Test clone successful. Path: {cloned_path}")
        if os.path.exists(cloned_path):
             logger.info(f"Contents of cloned repo (first level, up to 5): {os.listdir(cloned_path)[:5]}")
        logger.info(f"Attempting cleanup for test task {test_task_id_clone}...")
        if cleanup_repo(test_task_id_clone):
            logger.info(f"Cleanup successful for task {test_task_id_clone}.")
        else:
            logger.error(f"Cleanup failed for task {test_task_id_clone}.")
    else:
        logger.error(f"Test clone failed for {test_repo_url}, task {test_task_id_clone}.")

    logger.info("--- Testing cleanup for a non-existent task ---")
    non_existent_task_id = "non_existent_task_102"
    if cleanup_repo(non_existent_task_id):
        logger.info(f"Cleanup behavior for non-existent task {non_existent_task_id} as expected (True).")
    else:
        logger.error(f"Cleanup behavior for non-existent task {non_existent_task_id} not as expected (should be True).")
