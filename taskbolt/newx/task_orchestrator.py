import logging
import os
from .tasks import process_agent_task, get_task_log_file_path # Import the Celery task
from .celery_app import celery_app # To access Celery app instance for result backend
import time # For the __main__ test block

logger = logging.getLogger(__name__)
# Ensure basicConfig is only called if no handlers are configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TaskOrchestrator:
    def __init__(self):
        logger.info("TaskOrchestrator initialized for Celery.")

    def queue_task(self, task_id: str, task_details: dict):
        logger.info(f"Orchestrator queueing task {task_id} via Celery: {task_details}")
        try:
            # Use the API-generated task_id as Celery's task_id for easier tracking and result lookup
            result = process_agent_task.apply_async(args=[task_id, task_details], task_id=task_id)
            logger.info(f"Task {task_id} sent to Celery. Celery internal ID: {result.id} (should match our task_id)")
            return {"task_id": task_id, "celery_id": result.id, "status": "QUEUED"}
        except Exception as e:
            logger.error(f"Failed to send task {task_id} to Celery: {e}", exc_info=True)
            return {"task_id": task_id, "status": "FAILED_TO_QUEUE", "error": str(e)}

    def get_task_status(self, task_id: str):
        logger.info(f"Orchestrator fetching status for Celery task {task_id}")
        try:
            task_result = celery_app.AsyncResult(task_id)
            status = task_result.status # PENDING, STARTED, RETRY, FAILURE, SUCCESS, PROGRESS
            details = {}

            if isinstance(task_result.info, dict):
                details.update(task_result.info)
            elif task_result.info is not None: # Catches exceptions or other non-dict info for FAILURE state
                details['error_info'] = str(task_result.info)

            # Add current_step from meta if task is in a state that provides it
            if status in ['PROGRESS', 'STARTED'] and isinstance(task_result.info, dict) and 'current_step' in task_result.info:
                details['current_step'] = task_result.info['current_step']

            if status == 'PENDING':
                # Check if the task is truly unknown by checking if metadata exists
                # This is a workaround as Celery doesn't have a direct "unknown task" state via AsyncResult alone for all backends
                try:
                    # Accessing _get_task_meta is an internal API, might change between Celery versions.
                    # A more robust solution might involve a separate "task exists" check or specific backend features.
                    meta = task_result.backend._get_task_meta(task_id) # Attempt to get raw meta
                    if meta.get('status') is None: # If status is None in raw meta, it's likely unknown
                         logger.warning(f"Task {task_id} is PENDING but raw metadata indicates it might be unknown to the backend.")
                         # status = "UNKNOWN_OR_NEVER_QUEUED" # Optionally override status
                except Exception:
                    # If _get_task_meta fails or task not found by backend, it's likely unknown
                    logger.warning(f"Task {task_id} is PENDING and getting raw metadata failed; might be unknown to backend.")
                    # status = "UNKNOWN_OR_NEVER_QUEUED_EX"

                if not task_result.info and status != "UNKNOWN_OR_NEVER_QUEUED": # No custom metadata set yet, and not already marked unknown
                    logger.info(f"Task {task_id} is PENDING (likely queued or not yet started by worker).")
                elif status != "UNKNOWN_OR_NEVER_QUEUED": # PENDING but with some info (e.g. if task was revoked before starting)
                    logger.info(f"Task {task_id} is PENDING with info: {task_result.info}")


            logger.info(f"Status for Celery task {task_id}: {status}, Details from Celery: {details}")
            return {"task_id": task_id, "status": status, "details": details}
        except Exception as e:
            logger.error(f"Error fetching status for task {task_id} from Celery: {e}", exc_info=True)
            return {"task_id": task_id, "status": "ERROR_FETCHING_STATUS", "details": {"error": str(e)}}

    def get_task_logs(self, task_id: str):
        log_file_path = get_task_log_file_path(task_id)
        logger.info(f"Orchestrator fetching logs for task {task_id} from file: {log_file_path}")

        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as f:
                    logs = [line.strip() for line in f.readlines() if line.strip()]
                return {"task_id": task_id, "logs": logs}
            else:
                # If no log file, check Celery task status to provide more context
                status_info = self.get_task_status(task_id)
                current_celery_status = status_info.get("status", "UNKNOWN")

                if current_celery_status in ["PENDING", "QUEUED", "STARTED"]:
                    return {"task_id": task_id, "logs": [f"Info: Task {task_id} is {current_celery_status}. Logs will appear once generated."]}
                elif current_celery_status == "UNKNOWN_OR_NEVER_QUEUED" or                      (current_celery_status == "PENDING" and not status_info.get("details")): # Heuristic for unknown
                    return {"task_id": task_id, "logs": [f"Error: Task {task_id} not found or never processed."]}

                return {"task_id": task_id, "logs": ["No log file found. Task may not have started logging or an issue occurred."]}
        except Exception as e:
            logger.error(f"Error reading log file for task {task_id} at {log_file_path}: {e}", exc_info=True)
            return {"task_id": task_id, "logs": [f"Error fetching logs: {str(e)}"]}

if __name__ == "__main__":
    # This block is for direct testing of the orchestrator.
    # It will not run when imported by FastAPI app.
    print("Running TaskOrchestrator direct test with Celery integration...")
    orchestrator = TaskOrchestrator()
    # Using a timestamp to ensure task_id is unique for each test run if needed.
    test_task_id = f"celery_orch_test_{int(time.time())}"
    details = {"repository_link": "http://example.com/repo.git", "task_description": f"Test task at {test_task_id}"}

    print(f"--- Testing Orchestrator with Task ID: {test_task_id} ---")
    queue_response = orchestrator.queue_task(test_task_id, details)
    print(f"Queue response: {queue_response}")

    if queue_response.get("status") == "QUEUED":
        print(f"Task {test_task_id} queued. Waiting for potential processing by a worker...")

        # Poll status and logs a few times
        for i in range(6):
            time.sleep(4)
            print(f"--- Poll attempt {i+1} for task {test_task_id} ---")
            status_response = orchestrator.get_task_status(test_task_id)
            print(f"Status: {status_response}")
            logs_response = orchestrator.get_task_logs(test_task_id)
            print(f"Logs: {logs_response}")
            # Check for terminal states recognized by Celery
            if status_response["status"] in ["SUCCESS", "FAILURE", "REVOKED"] or                status_response["status"].startswith("UNKNOWN_OR_NEVER_QUEUED"): # Our custom derived status
                print("Task reached a terminal state or is no longer progressing.")
                break

        print("\nReminder: To run this test effectively, ensure a Celery worker is running:")
        print("  cd /app (project_root)")
        print("  celery -A taskbolt.newx.celery_app worker -l INFO -P solo (or eventlet/gevent for concurrency)")
        print("And ensure your Redis server (broker/backend) is running and accessible via CELERY_BROKER_URL (default redis://localhost:6379/0).")
    else:
        print(f"Failed to queue task {test_task_id}: {queue_response.get('error', 'Unknown error')}")
