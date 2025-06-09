from .celery_app import celery_app
import time
import logging
import os
import random

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence # Sequence kept for potential future use
import operator

from .github_utils import clone_repo, cleanup_repo

logger = logging.getLogger(__name__)
# BasicConfig should ideally be called once at application entry point.
# If celery worker/beat might be separate entry points, ensure it's configured there too.
if not logger.handlers: # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

LOGS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "tasks")
os.makedirs(LOGS_BASE_DIR, exist_ok=True)

def get_task_log_file_path(task_id: str) -> str:
    return os.path.join(LOGS_BASE_DIR, f"{task_id}.log")

def append_log(task_id: str, message: str, level: str = "INFO"):
    log_file = get_task_log_file_path(task_id)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    log_entry = f"{timestamp} - {level.upper()} - {message}\n"
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"ALERT: Failed to write to task log file {log_file}: {e}")
    # Log to Celery logger as well
    celery_log_message = f"Task Log [{task_id}]: {message}"
    if level.upper() == "ERROR":
        logger.error(celery_log_message)
    elif level.upper() == "WARNING":
        logger.warning(celery_log_message)
    else:
        logger.info(celery_log_message)

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    task_id: str
    task_details: dict
    repo_url: str
    repo_local_path: Annotated[str | None, operator.setitem]
    current_step_description: Annotated[str, operator.setitem]
    log_messages: Annotated[list[str], operator.add]
    error_message: Annotated[str | None, operator.setitem]

# --- LangGraph Nodes ---
def step_clone_repo(state: AgentState) -> dict:
    task_id = state['task_id']
    repo_url = state['repo_url']
    append_log(task_id, f"GRAPH_NODE: Executing step_clone_repo for {repo_url}")
    local_path = clone_repo(repo_url, task_id)
    if local_path:
        append_log(task_id, f"Repository cloned successfully to {local_path}")
        return {
            "repo_local_path": local_path,
            "current_step_description": "Repository Cloned Successfully",
            "log_messages": [f"Cloned {repo_url} to {local_path}"],
            "error_message": None
        }
    else:
        err_msg = f"Failed to clone repository: {repo_url}"
        append_log(task_id, err_msg, level="ERROR")
        return {
            "repo_local_path": None,
            "current_step_description": "Repository Cloning Failed",
            "log_messages": [err_msg],
            "error_message": err_msg
        }

def should_proceed_after_cloning(state: AgentState) -> str:
    append_log(state['task_id'], f"GRAPH_EDGE: Checking condition 'should_proceed_after_cloning'. Error: {state.get('error_message')}")
    if state.get("error_message"):
        return "handle_error_state"
    return "parse_repo"

def step_parse_repo(state: AgentState) -> dict:
    task_id = state['task_id']
    repo_local_path = state.get('repo_local_path')
    if not repo_local_path:
        append_log(task_id, "Skipping repository parsing: No local path available.", level="WARNING")
        # This case should ideally not be reached if conditional edges are set up correctly
        return {"current_step_description": "Parsing Skipped (No Repo)", "log_messages": ["Skipped parsing (no repo)"]}
    append_log(task_id, f"GRAPH_NODE: Executing step_parse_repo for {repo_local_path}")
    time.sleep(1) # Simulate parsing
    append_log(task_id, "Repository parsed (simulated).")
    return {"current_step_description": "Repository Parsed", "log_messages": ["Parsed repository (simulated)"]}

def step_plan_execution(state: AgentState) -> dict:
    task_id = state['task_id']
    append_log(task_id, "GRAPH_NODE: Executing step_plan_execution")
    time.sleep(1) # Simulate planning
    append_log(task_id, "Execution plan created (simulated).")
    return {"current_step_description": "Execution Planned", "log_messages": ["Created execution plan (simulated)"]}

def step_apply_modifications(state: AgentState) -> dict:
    task_id = state['task_id']
    append_log(task_id, "GRAPH_NODE: Executing step_apply_modifications")
    time.sleep(2) # Simulate modifications
    append_log(task_id, "Code modifications applied (simulated).")
    return {"current_step_description": "Modifications Applied", "log_messages": ["Applied modifications (simulated)"]}

def step_handle_error_state(state: AgentState) -> dict:
    task_id = state['task_id']
    error_msg = state.get('error_message', "Unknown error encountered in graph.")
    append_log(task_id, f"GRAPH_NODE: Executing step_handle_error_state. Error: {error_msg}", level="ERROR")
    # This node doesn't change the error_message, just processes it (e.g., logs it)
    return {"current_step_description": f"Error Processed: {error_msg}", "log_messages": [f"Error state processed: {error_msg}"]}

# --- Celery Task Definition ---
@celery_app.task(bind=True, name='taskbolt.newx.process_agent_task')
def process_agent_task(self, task_id: str, task_details: dict):
    effective_task_id_for_logging = task_id
    celery_internal_id = self.request.id
    append_log(effective_task_id_for_logging, f"Task STARTED (Celery ID: {celery_internal_id}): Initializing LangGraph agent process.")
    self.update_state(state='STARTED', meta={'current_step': 'Initializing LangGraph', 'custom_task_id': effective_task_id_for_logging, 'celery_id': celery_internal_id})
    logger.info(f"Celery task [{celery_internal_id}] (Our Task ID: [{effective_task_id_for_logging}]) using LangGraph with details: {task_details}")

    workflow = StateGraph(AgentState)
    workflow.add_node("clone_repo", step_clone_repo)
    workflow.add_node("parse_repo", step_parse_repo)
    workflow.add_node("plan_execution", step_plan_execution)
    workflow.add_node("apply_modifications", step_apply_modifications)
    workflow.add_node("handle_error_state", step_handle_error_state)

    workflow.set_entry_point("clone_repo")
    workflow.add_conditional_edges(
        "clone_repo",
        should_proceed_after_cloning,
        {
            "parse_repo": "parse_repo",
            "handle_error_state": "handle_error_state"
        }
    )
    workflow.add_edge("parse_repo", "plan_execution")
    workflow.add_edge("plan_execution", "apply_modifications")
    workflow.add_edge("apply_modifications", END) # Successful completion
    workflow.add_edge("handle_error_state", END)  # End after error handling

    app_graph = workflow.compile()
    initial_state = AgentState(
        task_id=effective_task_id_for_logging,
        task_details=task_details,
        repo_url=task_details.get('repository_link', 'N/A'),
        repo_local_path=None,
        current_step_description="Pending Graph Execution",
        log_messages=[f"Initial state for LangGraph for task {effective_task_id_for_logging}."],
        error_message=None
    )

    final_state = None
    try:
        append_log(effective_task_id_for_logging, "GRAPH_EXEC: Invoking LangGraph execution...")
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 10})
        append_log(effective_task_id_for_logging, f"GRAPH_EXEC: LangGraph execution completed. Final state description: {final_state.get('current_step_description')}")

        # Check if the graph execution itself resulted in an error state being set
        if final_state and final_state.get("error_message"):
            # This means the graph explicitly went into an error state (e.g., cloning failed and was handled by handle_error_state)
            raise Exception(f"Graph execution ended in a handled error state: {final_state['error_message']}")

        # If no error_message in final_state, assume graph steps that ran were successful
        result_meta = {
            'output': f"LangGraph process completed. Last step: {final_state.get('current_step_description') if final_state else 'Unknown'}",
            'final_step': final_state.get('current_step_description') if final_state else 'Unknown',
            'graph_internal_logs': final_state.get('log_messages', []) if final_state else ['Graph did not return a final state.'],
            'custom_task_id': effective_task_id_for_logging,
            'celery_id': celery_internal_id
        }
        append_log(effective_task_id_for_logging, f"SUCCESS: LangGraph process completed. Details: {result_meta['output']}")
        logger.info(f"Task [{effective_task_id_for_logging}]: LangGraph process finished successfully.")
        return result_meta

    except Exception as e:
        # This block catches errors from app_graph.invoke() or the explicit raise above
        error_msg = f"LangGraph execution or post-processing failed: {str(e)}"
        append_log(effective_task_id_for_logging, error_msg, level="ERROR")
        logger.error(f"Task [{effective_task_id_for_logging}]: {error_msg}", exc_info=True)
        # Re-raise the exception so Celery marks the task as FAILURE
        raise
    finally:
        # This cleanup runs regardless of success or failure of the graph execution
        append_log(effective_task_id_for_logging, f"Performing final cleanup for task {effective_task_id_for_logging} (repo path: {initial_state.get('repo_local_path', 'None set')}).")
        # The repo_local_path in initial_state might not be set if cloning failed early.
        # The cleanup_repo function handles non-existent paths gracefully.
        # If cloning succeeded, final_state should also have repo_local_path.
        path_to_cleanup = (final_state.get('repo_local_path') if final_state and final_state.get('repo_local_path') else initial_state.get('repo_local_path'))
        if path_to_cleanup and CLONED_REPOS_BASE_DIR in path_to_cleanup: # Basic safety check
             # We pass task_id to cleanup_repo, it constructs the path
             cleanup_repo(effective_task_id_for_logging)
        else:
             append_log(effective_task_id_for_logging, fSkipping cleanup: repo_local_path not available or invalid: {path_to_cleanup}, level=WARNING)
