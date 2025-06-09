from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from taskbolt.newx.task_orchestrator import TaskOrchestrator # Adjusted import path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = TaskOrchestrator()

class TaskLogsResponse(BaseModel):
    task_id: str
    logs: List[str]

@router.get("/logs/{task_id}", response_model=TaskLogsResponse)
async def get_task_logs(task_id: str):
    logger.info(f"API /logs fetching logs for task_id: {task_id}")

    try:
        logs_result = orchestrator.get_task_logs(task_id)
    except Exception as e:
        logger.error(f"Error getting logs from orchestrator for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting logs from orchestrator: {str(e)}")

    # Check if the task was found by the orchestrator, even if logs might be empty
    # The orchestrator's get_task_logs returns a specific message if task_id is not found
    if "Error: Task ID not found." in logs_result.get("logs", []):
         # This check could be made more robust, e.g. by orchestrator returning a specific status
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found for logs.")

    return TaskLogsResponse(
        task_id=task_id,
        logs=logs_result.get("logs", []) # Return empty list if logs are None/missing but task exists
    )
