from fastapi import FastAPI
import yaml # For loading config
import os # For path manipulation

# Determine the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "agent_config.yaml")

app = FastAPI(title="Newx Agent Backend")

class MainAgent:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = self.load_config(config_path)
        if self.config:
            print(f"MainAgent initialized with config from {config_path}: {self.config.get('agent_name', 'N/A')}")
        else:
            print(f"MainAgent initialized with default/empty config due to load failure from {config_path}.")


    def load_config(self, config_path):
        print(f"Attempting to load config from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                print("Config loaded successfully.")
                return config_data
        except FileNotFoundError:
            print(f"ERROR: Config file not found at {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"ERROR: Error parsing YAML config at {config_path}: {e}")
            return {}
        except Exception as e:
            print(f"ERROR: Unexpected error loading config at {config_path}: {e}")
            return {}

    def process_task(self, task_details):
        model_to_use = self.config.get('default_model', 'default_model_not_set')
        print(f"Processing task: {task_details} using model {model_to_use}")
        # Placeholder for actual AI logic
        return {"status": "processing_placeholder", "details": f"AI logic placeholder using {model_to_use}"}

# Initialize agent instance (singleton for this basic setup)
# In a more complex app, this might be managed with dependency injection
main_agent_instance = MainAgent()

@app.get("/newx/")
async def read_newx_root():
    return {"message": "Welcome to the Newx Agent Backend. Agent loaded.", "agent_name": main_agent_instance.config.get('agent_name', 'N/A')}

@app.get("/newx/ping")
async def ping_newx():
    return {"message": "Newx backend is alive!", "agent_config": main_agent_instance.config}

@app.post("/newx/process_task_debug") # Debug endpoint
async def process_task_debug(task_details: dict):
    # This is a simplified endpoint for direct testing of process_task
    # In real flow, this would be called by orchestrator
    result = main_agent_instance.process_task(task_details)
    return result

# The following is for Uvicorn to run this app:
# uvicorn taskbolt.newx.main_agent:app --reload --port 8001
# Or, if in this directory: uvicorn main_agent:app --reload --port 8001
