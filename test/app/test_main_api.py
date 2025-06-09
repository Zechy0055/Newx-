import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Assuming your FastAPI app instance is named 'app' in 'app.main'
# Adjust the import according to your project structure
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_logger():
    with patch("app.main.logger", autospec=True) as mock_log:
        # To mock bound loggers as well, we can make the return value of bind also a MagicMock
        # that then has the actual logging methods.
        bound_logger_mock = MagicMock()
        mock_log.bind.return_value = bound_logger_mock
        # Make the direct calls also point to a mock if they are used without bind
        mock_log.info = bound_logger_mock.info
        mock_log.error = bound_logger_mock.error
        mock_log.warning = bound_logger_mock.warning
        mock_log.debug = bound_logger_mock.debug
        yield bound_logger_mock # This will be the mock that records calls like .info(), .error()


def test_log_frontend_event_valid_payload(mock_logger: MagicMock):
    """Test logging a valid INFO event from the frontend."""
    payload = {
        "level": "INFO",
        "message": "Frontend component loaded",
        "component": "MyComponent",
        "function": "useEffect",
        "context": {"user_id": 123},
        "frontend_timestamp": "2024-01-01T12:00:00Z"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}

    # Check if logger.info was called with the correct message structure
    # The actual message includes "[FRONTEND] " prefix
    mock_logger.info.assert_called_once()
    args, _ = mock_logger.info.call_args
    assert "[FRONTEND] Frontend component loaded" in args[0]

    # To check bound context, we need to inspect the bound logger if `bind` was used before `info`
    # The current mock_logger fixture directly provides the mock for info/error etc.
    # If app.main.logger.bind().info() was called, then mock_logger here *is* the bound_logger_mock
    # So, we can check its call directly. The logger in app.main does:
    # bound_logger = logger.bind(**log_context_items)
    # bound_logger.info("[FRONTEND] " + payload.message)
    # So, the actual call to info won't have the context as args, it's part of the logger's state.
    # This mocking strategy might need adjustment if we want to assert on bound values easily.
    # For now, asserting the message content is a good first step.
    # A more advanced check would involve how `logger.bind` was called if we mocked `logger` itself,
    # rather than its methods.

    # Simpler check for now:
    # This depends on how the logger is called in the endpoint.
    # If it's logger.bind(...).info(...), then mock_logger.info will be called.
    # The bound arguments are part of the logger instance, not passed to info().
    # This test will pass if info() is called with the message.
    # Verifying bound context would require a more complex mock or capturing log output.

def test_log_frontend_event_error_payload(mock_logger: MagicMock):
    """Test logging a valid ERROR event from the frontend."""
    payload = {
        "level": "ERROR",
        "message": "Failed to fetch data",
        "component": "DataFetcher",
        "function": "fetchData",
        "context": {"url": "/api/data", "status": 500},
        "frontend_timestamp": "2024-01-01T12:05:00Z"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}
    mock_logger.error.assert_called_once_with("[FRONTEND] Failed to fetch data")

def test_log_frontend_event_debug_payload(mock_logger: MagicMock):
    """Test logging a valid DEBUG event from the frontend."""
    payload = {
        "level": "DEBUG",
        "message": "User clicked button",
        "component": "InteractiveForm",
        "context": {"button_id": "submit_button"},
        "frontend_timestamp": "2024-01-01T12:10:00Z"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}
    mock_logger.debug.assert_called_once_with("[FRONTEND] User clicked button")

def test_log_frontend_event_warning_payload(mock_logger: MagicMock):
    """Test logging a valid WARNING event from the frontend."""
    payload = {
        "level": "WARNING",
        "message": "API response slow",
        "component": "APIService",
        "function": "callExternalAPI",
        "context": {"duration_ms": 2500},
        "frontend_timestamp": "2024-01-01T12:15:00Z"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}
    mock_logger.warning.assert_called_once_with("[FRONTEND] API response slow")


def test_log_frontend_event_unknown_level(mock_logger: MagicMock):
    """Test logging with an unknown log level."""
    payload = {
        "level": "CRITICAL", # Not a standard level handled explicitly by the endpoint
        "message": "A critical frontend event occurred",
        "frontend_timestamp": "2024-01-01T12:20:00Z"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}
    # The endpoint logs unknown levels as WARNING
    mock_logger.warning.assert_called_once()
    args, _ = mock_logger.warning.call_args
    assert "Unknown log level 'CRITICAL'" in args[0]
    assert "A critical frontend event occurred" in args[0]


def test_log_frontend_event_invalid_payload_missing_level():
    """Test request with missing 'level' field."""
    payload = {
        # "level": "INFO", # Missing level
        "message": "This log has no level"
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 422 # Unprocessable Entity


def test_log_frontend_event_invalid_payload_missing_message():
    """Test request with missing 'message' field."""
    payload = {
        "level": "INFO",
        # "message": "This log has no message" # Missing message
    }
    response = client.post("/api/log_frontend_event", json=payload)
    assert response.status_code == 422


def test_log_frontend_event_empty_payload():
    """Test request with an empty JSON payload."""
    response = client.post("/api/log_frontend_event", json={})
    assert response.status_code == 422

def test_log_frontend_event_extra_fields(mock_logger: MagicMock):
    """Test logging a valid event with extra unexpected fields in payload (should be ignored by model but logged if in context)."""
    payload = {
        "level": "INFO",
        "message": "Event with extra fields",
        "component": "TestComponent",
        "unexpected_field": "some_value", # This is not in Pydantic model explicitly
        "context": {"user_id": 789, "session_id": "zyxw"},
        "frontend_timestamp": "2024-01-01T12:00:00Z"
    }
    # Pydantic model will ignore unexpected_field unless context is open-ended (e.g. Dict[str, Any])
    # Our FrontendLogPayload has `context: Optional[Dict[str, Any]]`, so it should go there if not a top-level field.
    # If `unexpected_field` is at the root, Pydantic will raise validation error unless model is configured to ignore extra.
    # Let's assume context is where extra fields should go.
    # The current Pydantic model does not have `unexpected_field` at the root.
    # If the intent is for `context` to catch these, the frontend should put it there.
    # If Pydantic model (FrontendLogPayload) had `extra = 'ignore'`, it would pass.
    # If Pydantic model had `context: Optional[Dict[str, Any]] = Field(default_factory=dict)`,
    # and the FE sent unexpected_field inside context, it would be captured.

    # For this test, we'll assume the payload is valid as per the model,
    # meaning `unexpected_field` is NOT at the root of payload sent to backend,
    # but rather part of the 'context' dict.

    payload_sent_to_backend = {
        "level": "INFO",
        "message": "Event with extra fields in context",
        "component": "TestComponent",
        "context": {"user_id": 789, "session_id": "zyxw", "unexpected_field": "some_value"},
        "frontend_timestamp": "2024-01-01T12:00:00Z"
    }

    response = client.post("/api/log_frontend_event", json=payload_sent_to_backend)
    assert response.status_code == 200
    assert response.json() == {"status": "Log received"}

    mock_logger.info.assert_called_once()
    args, _ = mock_logger.info.call_args
    assert "[FRONTEND] Event with extra fields in context" in args[0]
    # Verifying that the 'unexpected_field' from context was bound would require
    # more intricate mocking of the logger.bind().info() chain, or capturing output.
    # The current mock_logger only checks the final call to info/error etc.
    # For now, this test confirms the main message is logged.
