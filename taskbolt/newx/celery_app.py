from celery import Celery
import os

# Configuration should ideally come from a settings file or environment variables
REDIS_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# The first argument to Celery is the name of the current module.
# We use 'taskbolt.newx' to ensure tasks are discovered correctly when worker starts with -A taskbolt.newx.celery_app
celery_app = Celery(
    'taskbolt.newx',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['taskbolt.newx.tasks'] # List of modules to import when the worker starts.
                                     # We'll define tasks in tasks.py
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # task_track_started=True, # To report 'STARTED' state
    # task_send_sent_event=True, # To send task-sent events
)

if __name__ == '__main__':
    # This is for starting the worker directly, e.g. python -m celery_app worker -l INFO
    # More commonly, you'd use: celery -A taskbolt.newx.celery_app worker -l INFO
    celery_app.start()
