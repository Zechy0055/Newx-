from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from taskbolt.newx.task_orchestrator import TaskOrchestrator # Adjusted import path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = TaskOrchestrator()

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    details: str | None = None

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    logger.info(f"API /status fetching status for task_id: {task_id}")

    try:
        status_result = orchestrator.get_task_status(task_id)
    except Exception as e:
        logger.error(f"Error getting status from orchestrator for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting status from orchestrator: {str(e)}")

    if status_result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    return TaskStatusResponse(
        task_id=task_id,
        status=status_result.get("status", "unknown"),
        details=status_result.get("details", "No details available.")
    )
