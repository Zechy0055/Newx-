from fastapi import FastAPI
from .routes import run, status, logs # Use .routes for relative imports

app = FastAPI(title="Taskbolt Agent API")

# Include routers from the routes module
app.include_router(run.router, prefix="/api/agent", tags=["Agent Tasks"])
app.include_router(status.router, prefix="/api/agent", tags=["Agent Status & Logs"]) # Combined tag for now
app.include_router(logs.router, prefix="/api/agent", tags=["Agent Status & Logs"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Taskbolt Agent API. Visit /docs for API documentation."}

if __name__ == "__main__":
    # This is for local testing if needed, though usually uvicorn is used directly
    import uvicorn
    print("Starting API server with uvicorn. Access at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
