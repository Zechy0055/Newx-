# AutoCodeRover Frontend Visualization

## Overview

This frontend application serves as a visualization tool for the AutoCodeRover agent's process. It allows users to input a GitHub issue, repository link, and commit hash, and then watch as the backend agents (simulated or real) perform tasks like context retrieval, patch generation, and test reproduction. The frontend displays messages and outputs from these agents in a structured, expandable timeline.

Built with Next.js and TypeScript.

## Setup and Running Locally

To set up and run the frontend application locally, follow these steps:

1.  **Navigate to the frontend directory:**
    ```bash
    cd demo_vis/front
    ```

2.  **Install dependencies:**
    Make sure you have Node.js and npm (or yarn) installed.
    ```bash
    npm install
    # or
    # yarn install
    ```

3.  **Run the development server:**
    ```bash
    npm run dev
    # or
    # yarn dev
    ```
    This will typically start the frontend application on `http://localhost:3000`.

4.  **Ensure the backend server is running:**
    The frontend communicates with a backend FastAPI server (typically expected on `http://localhost:5000`). Ensure the backend server (from the main `app` directory) is running before using the frontend. Refer to the main project's README for instructions on running the backend.

## Main Components and Their Roles

*   **`app/page.tsx`**:
    *   This is the main and currently only page of the application.
    *   It contains the input form for repository and issue details.
    *   Manages the overall state, including the list of messages from backend agents, loading states, and user inputs.
    *   Handles communication with the backend API to initiate the AutoCodeRover process and receive streamed updates.
    *   Renders the timeline of agent messages.

*   **`MessageDiv` (defined in `page.tsx`)**:
    *   A React component responsible for displaying individual messages from backend agents (e.g., AutoCodeRover, Context Retrieval Agent, Patch Generation Agent).
    *   Each message has a title and content (often Markdown) and can be expanded or collapsed by the user.
    *   It's memoized using `React.memo` for performance optimization.

*   **`MarkdownRender` (defined in `page.tsx`)**:
    *   A utility component that takes a Markdown string and renders it as HTML.
    *   It uses `react-syntax-highlighter` for styling code blocks within the Markdown.

*   **`LoadingDiv` (defined in `page.tsx`)**:
    *   A simple component displayed to indicate that the application is waiting for more data or for an agent process to complete.

*   **`logToBackend` (defined in `page.tsx`)**:
    *   A utility function used throughout the `App` component to send client-side log events (e.g., user actions, stream processing events, errors) to a dedicated backend logging endpoint.

## Backend API Interaction

The frontend interacts with the backend server via two main API endpoints:

1.  **`/api/run_github_issue` (POST)**:
    *   Called when the user submits the form with repository and issue details.
    *   The frontend sends a JSON payload containing `repository_link`, `commit_hash`, and `issue_link`.
    *   The backend processes this request and streams back a series of JSON objects, each representing an event or message from an agent. These messages are separated by a delimiter (`</////json_end>`).
    *   The `onsubmit_callback` function in `page.tsx` handles this request and the subsequent stream processing via its nested `read` function.

2.  **`/api/log_frontend_event` (POST)**:
    *   Called by the `logToBackend` utility function.
    *   The frontend sends a JSON payload (`LogPayload`) containing:
        *   `level`: Log severity ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        *   `message`: The log message string.
        *   `component` (optional): Name of the frontend component.
        *   `function` (optional): Name of the frontend function.
        *   `context` (optional): Additional structured data relevant to the log.
        *   `frontend_timestamp`: ISO timestamp added by `logToBackend`.
    *   The backend receives this payload and logs the event using its own `loguru` logger, allowing frontend activity to be recorded alongside backend logs.

## State Management

*   **Form Input:** Managed by `react-hook-form`.
*   **Agent Messages:** Stored in the `messages` state array (`useState<Array<AnyMessage>>([])`). Each message has an `id`, `category`, `is_open` status, and type-specific fields. Messages are added and updated immutably.
*   **Issue Information:** The problem statement from the GitHub issue is stored in the `problemStatement` state.
*   **Loading State:** A `loadingState` numerical flag manages UI indicators for different stages (idle, cloning, agent running).
*   **Message Visibility:** The `is_open` property on messages and the `handleToggleOpen` callback manage the expand/collapse state of individual messages in the UI.

## Further Development

*   Error handling for API requests can be further enhanced.
*   More sophisticated UI elements for different types of agent messages (e.g., diff views for patches).
*   Component modularization: `MessageDiv`, `MarkdownRender`, `LoadingDiv` could be moved to their own files under a `components` directory.
*   More comprehensive end-to-end testing.
