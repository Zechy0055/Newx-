from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import logging
from taskbolt.newx.task_orchestrator import TaskOrchestrator # Adjusted import path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
# Create a single, shared instance of the orchestrator.
# In a real app, you might use FastAPI's dependency injection for this.
orchestrator = TaskOrchestrator()

class TaskRequest(BaseModel):
    repository_link: str
    task_description: str

class TaskResponse(BaseModel):
    task_id: str
    message: str
    status: str
    status_endpoint: str
    logs_endpoint: str

@router.post("/run", response_model=TaskResponse)
async def start_agent_task(request: TaskRequest):
    logger.info(f"API /run received task request: {request.dict()}")
    task_id = str(uuid.uuid4())

    try:
        # Call the Newx backend (task_orchestrator)
        result = orchestrator.queue_task(task_id, request.dict())
        logger.info(f"Task {task_id} queued via orchestrator. Result: {result}")
    except Exception as e:
        logger.error(f"Failed to queue task with orchestrator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error queueing task with orchestrator: {str(e)}")

    return TaskResponse(
        task_id=task_id,
        message="Agent task successfully queued via orchestrator.",
        status=result.get("status", "unknown"),
        status_endpoint=f"/api/agent/status/{task_id}",
        logs_endpoint=f"/api/agent/logs/{task_id}"
    )
