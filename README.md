# AutoCodeRover: Autonomous Program Improvement

<br>

<p align="center">
  <img src="https://github.com/nus-apr/auto-code-rover/assets/16000056/8d249b02-1db4-4f58-a5a4-bdb694d65ab1" alt="autocoderover_logo" width="200px" height="200px">
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2404.05427"><strong>ArXiv Paper</strong></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://autocoderover.dev/"><strong>Website</strong></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://discord.gg/ScXsdE49JY"><strong>Discord server</strong></a>
</p>

<br>

![overall-workflow](https://github.com/nus-apr/auto-code-rover/assets/48704330/0b8da9ad-588c-4f7d-9c99-53f33d723d35)

<br>


> [!NOTE]
> This is a public version of the AutoCodeRover project. Check the latest results on our [website](https://autocoderover.dev/).

## 📣 Updates
- [November 21, 2024] AutoCodeRover(v20240620) achieves **46.20%** efficacy on SWE-bench Verified and **24.89%** on full SWE-bench.
- [August 14, 2024] On the SWE-bench Verified dataset released by OpenAI, AutoCodeRover(v20240620) achieves **38.40%** efficacy, and AutoCodeRover(v20240408) achieves 28.8% efficacy. More details in the [blog post](https://openai.com/index/introducing-swe-bench-verified/) from OpenAI and [SWE-bench leaderboard](https://www.swebench.com/).
- [July 18, 2024] AutoCodeRover now supports a new mode that outputs the list of potential fix locations.
- [June 20, 2024] AutoCodeRover(v20240620) now achieves **30.67%** efficacy (pass@1) on SWE-bench-lite!
- [June 08, 2024] Added support for Gemini, Groq (thank you [KasaiHarcore](https://github.com/KasaiHarcore) for the contribution!) and Anthropic models through AWS Bedrock (thank you [JGalego](https://github.com/JGalego) for the contribution!).
- [April 29, 2024] Added support for Claude and Llama models. Find the list of supported models [here](#using-a-different-model)! Support for more models coming soon.
- [April 19, 2024] AutoCodeRover now supports running on [GitHub issues](#github-issue-mode-set-up-and-run-on-new-github-issues) and [local issues](#local-issue-mode-set-up-and-run-on-local-repositories-and-local-issues)! Feel free to try it out and we welcome your feedback!

## [Discord](https://discord.gg/ScXsdE49JY) - server for general discussion, questions, and feedback.

## 👋 Overview

AutoCodeRover is a system designed to automate software engineering tasks, particularly fixing bugs and implementing features based on issue descriptions. It leverages Large Language Models (LLMs) combined with program analysis, code search, and debugging capabilities to understand issues, retrieve relevant context from code repositories, and generate patches. The system can operate on SWE-bench tasks, live GitHub issues, or local projects, and includes a frontend visualization component to display the agent's actions and reasoning process.

[Update on June 20, 2024] AutoCodeRover(v20240620) now resolves **30.67%** of issues (pass@1) in SWE-bench lite! AutoCodeRover achieved this efficacy while being economical - each task costs **less than $0.7** and is completed within **7 mins**!

<p align="center">
<img src=https://github.com/nus-apr/auto-code-rover/assets/16000056/78d184b2-f15c-4408-9eac-cfd3a11a503a width=500/>
<img src=https://github.com/nus-apr/auto-code-rover/assets/16000056/83253ae9-8789-474e-942d-708495b5b310 width=500/>
</p>

[April 08, 2024] First release of AutoCodeRover(v20240408) resolves **19%** of issues in [SWE-bench lite](https://www.swebench.com/lite.html) (pass@1), improving over the current state-of-the-art efficacy of AI software engineers.


AutoCodeRover works in two stages:

- 🔎 Context retrieval: The LLM is provided with code search APIs to navigate the codebase and collect relevant context.
- 💊 Patch generation: The LLM tries to write a patch, based on retrieved context.

### ✨ Highlights

AutoCodeRover has two unique features:

- Code search APIs are *Program Structure Aware*. Instead of searching over files by plain string matching, AutoCodeRover searches for relevant code context (methods/classes) in the abstract syntax tree.
- When a test suite is available, AutoCodeRover can take advantage of test cases to achieve an even higher repair rate, by performing *statistical fault localization*.

## ✨ Key Features

*   **Automated Issue Resolution:** Addresses bugs and implements features described in issue reports.
*   **SWE-bench Integration:** Supports running and evaluating on SWE-bench tasks.
*   **GitHub & Local Issue Processing:** Can work with live GitHub issues or local codebases and issue descriptions.
*   **Advanced Context Retrieval:** Employs program structure-aware code search APIs to find relevant code context.
*   **LLM-Powered Patch Generation:** Uses Large Language Models to generate code patches.
*   **Interactive Frontend Visualization:** A Next.js application (`demo_vis/front`) to visualize the agent's workflow and messages.
*   **Structured Logging:** Comprehensive backend logging with `loguru` (including per-task JSON logs) and frontend logging to a backend endpoint.
*   **Modular Agent Architecture:** Comprises specialized agents for different tasks (context retrieval, patch writing, review, etc.).
*   **Support for Multiple LLMs:** Compatible with various models from OpenAI, Anthropic, Google (via LiteLLM), and local models via Ollama/Groq.

## 🏗️ Architecture Overview

AutoCodeRover consists of the following main components:

*   **Backend (Python, FastAPI):**
    *   Handles the core logic for task execution, including parsing issues, interacting with Git repositories, and managing the agent workflow.
    *   Contains the specialized agents for different software engineering sub-tasks.
    *   Exposes an API (e.g., for stream-based updates for the frontend, receiving frontend logs). The FastAPI app instance is defined in `app/main.py`.
*   **Frontend (Next.js, TypeScript):**
    *   Located in `demo_vis/front/`.
    *   Provides a web interface to input task details (e.g., GitHub issue links).
    *   Visualizes the step-by-step actions and reasoning of the backend agents as they work on a task.
*   **Agents (`app/agents/`):**
    *   A collection of specialized Python modules that use LLMs and other tools to perform specific actions:
        *   `ContextRetrievalAgent`: Searches the codebase to find relevant context for an issue.
        *   `PatchAgent`: Generates code patches based on the issue and retrieved context.
        *   `TestAgent`: Attempts to generate reproducing tests for issues.
        *   `ReviewAgent`: (Experimental) Reviews generated patches and tests.

**Interaction Flow (High-Level for Demo Visualization):**
1.  User submits an issue via the Frontend.
2.  Frontend sends the task details to the Backend API (e.g., `/api/run_github_issue`).
3.  Backend processes the task, invoking various agents in sequence.
4.  Agents stream messages/updates back to the Frontend, which displays them in real-time.
5.  Frontend sends its own log events (UI interactions, errors) to a separate Backend API endpoint (`/api/log_frontend_event`) for centralized logging.

## 🗎 arXiv Paper
### AutoCodeRover: Autonomous Program Improvement [[arXiv 2404.05427]](https://arxiv.org/abs/2404.05427)

<p align="center">
  <a href="https://arxiv.org/abs/2404.05427">
    <img src="https://github.com/nus-apr/auto-code-rover/assets/48704330/c6422951-a6e8-4494-9403-b5ada3d9ee7d" alt="First page of arXiv paper" width="570">
  </a>
</p>

For referring to our work, please cite and mention:
```
@inproceedings{zhang2024autocoderover,
    author = {Zhang, Yuntong and Ruan, Haifeng and Fan, Zhiyu and Roychoudhury, Abhik},
    title = {AutoCodeRover: Autonomous Program Improvement},
    year = {2024},
    isbn = {9798400706127},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3650212.3680384},
    doi = {10.1145/3650212.3680384},
    booktitle = {Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
    pages = {1592–1604},
    numpages = {13},
    keywords = {automatic program repair, autonomous software engineering, autonomous software improvement, large language model},
    location = {Vienna, Austria},
    series = {ISSTA 2024}
}
```

## ✔️ Example: Django Issue #32347

As an example, AutoCodeRover successfully fixed issue [#32347](https://code.djangoproject.com/ticket/32347) of Django. See the demo video for the full process:

https://github.com/nus-apr/auto-code-rover/assets/48704330/719c7a56-40b8-4f3d-a90e-0069e37baad3

### Enhancement: leveraging test cases

AutoCodeRover can resolve even more issues, if test cases are available. See an example in the video:

https://github.com/nus-apr/auto-code-rover/assets/48704330/26c9d5d4-04e0-4b98-be55-61c1d10a36e5

## 🚀 Setup & Running

### Prerequisites

*   **Python:** Version 3.9 or higher.
*   **Conda (Recommended):** For managing Python environments.
*   **Node.js and npm (or yarn):** For the frontend development server.
*   **Docker (Recommended for backend):** For running the backend in a containerized environment, especially for SWE-bench tasks. The provided `Dockerfile.minimal` is a good starting point.
*   **Git:** For cloning repositories.

### Backend Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/nus-apr/auto-code-rover.git
    cd auto-code-rover
    ```

2.  **Set up Python Environment (Conda Recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate auto-code-rover
    ```
    Alternatively, create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **API Keys & Configuration:**
    Set the necessary API keys as environment variables. For example:
    ```bash
    export OPENAI_KEY="sk-YOUR-OPENAI-API-KEY-HERE"
    export ANTHROPIC_API_KEY="sk-ant-api..."
    # Add other keys (e.g., GROQ_API_KEY) as needed.
    ```
    Refer to `app/config.py` for other configuration variables that can be adjusted. These are typically set via command-line arguments when running tasks (see below).

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd demo_vis/front
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    # or
    # yarn install
    ```

### Running the Application

**1. Backend Server (for Demo UI & API):**

The backend includes a FastAPI application defined in `app/main.py`. To run this server for the frontend visualization and API endpoints (like frontend logging):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```
This makes the backend API (including `/api/run_github_issue` and `/api/log_frontend_event`) available at `http://localhost:5000`.

**2. Backend for CLI Tasks (e.g., SWE-bench, specific issue processing):**

Use the command-line interface as described in the sections below (e.g., "GitHub issue mode", "Local issue mode", "SWE-bench mode"). These commands directly run the task processing logic.
Example:
```bash
PYTHONPATH=. python app/main.py github-issue --output-dir output --setup-dir setup --model gpt-4o-2024-05-13 --task-id my-test-issue --clone-link <repo_link> --commit-hash <hash> --issue-link <issue_link>
```

**3. Frontend Development Server:**

Navigate to the frontend directory and run:
```bash
cd demo_vis/front
npm run dev
```
The frontend will typically be accessible at `http://localhost:3000`. Ensure the backend server (FastAPI/Uvicorn) is running and accessible to the frontend, usually at `http://localhost:5000`.


### Original Setup API key and environment (Primarily for Docker/CLI execution)

We recommend running AutoCodeRover in a Docker container for CLI tasks.

Set the `OPENAI_KEY` env var to your [OpenAI key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key):

```
export OPENAI_KEY=sk-YOUR-OPENAI-API-KEY-HERE
```

For Anthropic model, Set the `ANTHROPIC_API_KEY` env var can be found [here](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

```
export ANTHROPIC_API_KEY=sk-ant-api...
```

The same with `GROQ_API_KEY`

Build and start the docker image for the AutoCodeRover tool:

```
docker build -f Dockerfile.minimal -t acr .
docker run -it -e OPENAI_KEY="${OPENAI_KEY:-OPENAI_API_KEY}" acr
```

### Setup: local mode

Alternatively, you can have a local copy of AutoCodeRover and manage python dependencies with `environment.yml`.
This is the recommended setup for running SWE-bench experiments with AutoCodeRover.
With a working conda installation, do `conda env create -f environment.yml`.
Similarly, set `OPENAI_KEY` or `ANTHROPIC_API_KEY` in your shell before running AutoCodeRover.

## Running AutoCodeRover

You can run AutoCodeRover in three modes:

1. GitHub issue mode: Run ACR on a live GitHub issue by providing a link to the issue page.
2. Local issue mode: Run ACR on a local repository and a file containing the issue description.
3. SWE-bench mode: Run ACR on SWE-bench task instances. (local setup of ACR recommend.)

### [GitHub issue mode] Set up and run on new GitHub issues

If you want to use AutoCodeRover for new GitHub issues in a project, prepare the following:

- Link to clone the project (used for `git clone ...`).
- Commit hash of the project version for AutoCodeRover to work on (used for `git checkout ...`).
- Link to the GitHub issue page.

Then, in the docker container (or your local copy of AutoCodeRover), run the following commands to set up the target project
and generate patch:

```
cd /opt/auto-code-rover
conda activate auto-code-rover
PYTHONPATH=. python app/main.py github-issue --output-dir output --setup-dir setup --model gpt-4o-2024-05-13 --model-temperature 0.2 --task-id <task id> --clone-link <link for cloning the project> --commit-hash <any version that has the issue> --issue-link <link to issue page>
```
Here is an example command for running ACR on an issue from the langchain GitHub issue tracker:

```
PYTHONPATH=. python app/main.py github-issue --output-dir output --setup-dir setup --model gpt-4o-2024-05-13 --model-temperature 0.2 --task-id langchain-20453 --clone-link https://github.com/langchain-ai/langchain.git --commit-hash cb6e5e5 --issue-link https://github.com/langchain-ai/langchain/issues/20453
```

The `<task id>` can be any string used to identify this issue.

If patch generation is successful, the path to the generated patch will be written to a file named `selected_patch.json` in the output directory.

### [Local issue mode] Set up and run on local repositories and local issues

Instead of cloning a remote project and run ACR on an online issue, you can also prepare the local repository and issue beforehand,
if that suits the use case.

For running ACR on a local issue and local codebase, prepare a local codebase and write an issue description into a file,
and run the following commands:

```
cd /opt/auto-code-rover
conda activate auto-code-rover
PYTHONPATH=. python app/main.py local-issue --output-dir output --model gpt-4o-2024-05-13 --model-temperature 0.2 --task-id <task id> --local-repo <path to the local project repository> --issue-file <path to the file containing issue description>
```

If patch generation is successful, the path to the generated patch will be written to a file named `selected_patch.json` in the output directory.

### [SWE-bench mode] Set up and run on SWE-bench tasks

This mode is for running ACR on existing issue tasks contained in SWE-bench.

#### Set up

##### Install SWE-bench Docker

We use a [fork](https://github.com/nus-apr/SWE-bench-docker) of [SWE-bench docker](https://github.com/aorwall/SWE-bench-docker) to run regression tests (not `FAIL_TO_PASS` tests, but all the tests in the buggy programs). To install this, run

```
conda activate auto-code-rover
git submodule update --init --recursive
cd SWE-bench-docker
pip install .
```

##### Setting up Testbed

For SWE-bench mode, we recommend setting up ACR on a host machine, instead of running it in docker mode.

Firstly, set up the SWE-bench task instances locally.

1. Clone [this SWE-bench fork](https://github.com/yuntongzhang/SWE-bench) and follow the [installation instruction](https://github.com/yuntongzhang/SWE-bench?tab=readme-ov-file#to-install) to install dependencies.

2. Put the tasks to be run into a file, one per line:

```
cd <SWE-bench-path>
echo django__django-11133 > tasks.txt
```

Or if running on arm64 (e.g. Apple silicon), try this one which doesn't depend on Python 3.6 (which isn't supported in this env):

```
echo django__django-16041 > tasks.txt
```

Then, set up these tasks by running:
3. Set up these tasks in the file by running:

```
cd <SWE-bench-path>
conda activate swe-bench
python harness/run_setup.py --log_dir logs --testbed testbed --result_dir setup_result --subset_file tasks.txt
```

Once the setup for this task is completed, the following two lines will be printed:

```
setup_map is saved to setup_result/setup_map.json
tasks_map is saved to setup_result/tasks_map.json
```

The `testbed` directory will now contain the cloned source code of the target project.
A conda environment will also be created for this task instance.

_If you want to set up multiple tasks together, put multiple ids in `tasks.txt` and follow the same steps._

#### Run a single task in SWE-bench

Before running the task (`django__django-11133` here), make sure it has been set up as mentioned [above](#set-up-one-or-more-tasks-in-swe-bench).

```
cd <AutoCodeRover-path>
conda activate auto-code-rover
PYTHONPATH=. python app/main.py swe-bench --model gpt-4o-2024-05-13 --setup-map <SWE-bench-path>/setup_result/setup_map.json --tasks-map <SWE-bench-path>/setup_result/tasks_map.json --output-dir output --task django__django-11133
```

The output for a run (e.g. for `django__django-11133`) can be found at a location like this: `output/applicable_patch/django__django-11133_yyyy-MM-dd_HH-mm-ss/` (the date-time field in the directory name will be different depending on when the experiment was run).

Path to the final generated patch is written in a file named `selected_patch.json` in the output directory.

#### Run multiple tasks in SWE-bench

First, put the id's of all tasks to run in a file, one per line. Suppose this file is `tasks.txt`, the tasks can be run with

```
cd <AutoCodeRover-path>
conda activate auto-code-rover
PYTHONPATH=. python app/main.py swe-bench --model gpt-4o-2024-05-13 --setup-map <SWE-bench-path>/setup_result/setup_map.json --tasks-map <SWE-bench-path>/setup_result/tasks_map.json --output-dir output --task-list-file <SWE-bench-path>/tasks.txt
```

**NOTE**: make sure that the tasks in `tasks.txt` have all been set up in SWE-bench. See the steps [above](#set-up-one-or-more-tasks-in-swe-bench).

#### Using a config file

Alternatively, a config file can be used to specify all parameters and tasks to run. See `conf/example.conf` for an example.
Also see [EXPERIMENT.md](EXPERIMENT.md) for the details of the items in a conf file.
A config file can be used by:

```
python scripts/run.py conf/example.conf
```

### Using a different model

AutoCodeRover works with different foundation models. You can set the foundation model to be used with the `--model` command line argument.

The current list of supported models:

|  | Model | AutoCodeRover cmd line argument |
|:--------------:|---------------|--------------|
| OpenAI         | gpt-4o-2024-08-06      | --model gpt-4o-2024-08-06 |
|                | gpt-4o-2024-05-13      | --model gpt-4o-2024-05-13 |
|                | gpt-4-turbo-2024-04-09 | --model gpt-4-turbo-2024-04-09 |
|                | gpt-4-0125-preview     | --model gpt-4-0125-preview |
|                | gpt-4-1106-preview     | --model gpt-4-1106-preview |
|                | gpt-3.5-turbo-0125     | --model gpt-3.5-turbo-0125 |
|                | gpt-3.5-turbo-1106     | --model gpt-3.5-turbo-1106 |
|                | gpt-3.5-turbo-16k-0613 | --model gpt-3.5-turbo-16k-0613 |
|                | gpt-3.5-turbo-0613     | --model gpt-3.5-turbo-0613 |
|                | gpt-4-0613             | --model gpt-4-0613 |
| Anthropic      | Claude 3.5 Sonnet v2   | --model claude-3-5-sonnet-20241022 |
|                | Claude 3.5 Sonnet      | --model claude-3-5-sonnet-20240620 |
|                | Claude 3 Opus          | --model claude-3-opus-20240229 |
|                | Claude 3 Sonnet        | --model claude-3-sonnet-20240229 |
|                | Claude 3 Haiku         | --model claude-3-haiku-20240307 |
| Meta           | Llama 3 70B            | --model llama3:70b |
|                | Llama 3 8B             | --model llama3     |
| AWS Bedrock    | Claude 3 Opus          | --model bedrock/anthropic.claude-3-opus-20240229-v1:0 |
|                | Claude 3 Sonnet        | --model bedrock/anthropic.claude-3-sonnet-20240229-v1:0 |
|                | Claude 3 Haiku         | --model bedrock/anthropic.claude-3-haiku-20240307-v1:0 |
|                | Claude 3.5 Sonnet      | --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 |
|                | Nova Pro               | --model bedrock/us.amazon.nova-pro-v1:0 |
|                | Nova Lite              | --model bedrock/us.amazon.nova-lite-v1:0 |
|                | Nova Micro             | --model bedrock/us.amazon.nova-micro-v1:0 |
| LiteLLM        | Any LiteLLM model      | --model litellm-generic-<MODEL_NAME_HERE> |
| Groq           | Llama 3 8B             | --model groq/llama3-8b-8192 |
|                | Llama 3 70B            | --model groq/llama3-70b-8192 |
|                | Llama 2 70B            | --model groq/llama2-70b-4096 |
|                | Mixtral 8x7B           | --model groq/mixtral-8x7b-32768 |
|                | Gemma 7B               | --model groq/gemma-7b-it |


> [!NOTE]
> Using the Groq models on a free plan can cause the context limit to be exceeded, even on simple issues.

> [!NOTE]
> Some notes on running ACR with local models such as llama3:
> 1. Before using the llama3 models, please [install ollama](https://ollama.com/download/linux) and download the corresponding models with ollama (e.g. `ollama pull llama3`).
> 2. You can run ollama server on the host machine, and ACR in its container. ACR will attempt to communicate to the ollama server on host.
> 3. If your setup is ollama in host + ACR in its container, we recommend installing [Docker Desktop](https://docs.docker.com/desktop/) on the host, in addition to the [Docker Engine](https://docs.docker.com/engine/).
>     - Docker Desktop contains Docker Engine, and also has a virtual machine which makes it easier to access the host ports from within a container. With Docker Desktop, this setup will work without additional effort.
>     - When the docker installation is only Docker Engine, you may need to add either `--net=host` or `--add-host host.docker.internal=host-gateway` to the `docker run` command when starting the ACR container, so that ACR can communicate with the ollama server on the host machine.
> If you encounter any issue in the tool or experiment, you can contact us via email at ridwan.shariffdeen@sonarsource.com, or through our [discord server](https://discord.com/invite/ScXsdE49JY).

## Experiment Replication

Please refer to [EXPERIMENT.md](EXPERIMENT.md) for information on experiment replication.

## 🪵 Logging System

AutoCodeRover employs a comprehensive logging system:

*   **Backend:**
    *   Uses `loguru` for flexible and powerful logging.
    *   For each task run via `app/main.py` CLI, two main log files are generated in the task-specific output directory (e.g., `output_dir/[task_id_timestamp]/`):
        *   `info.log`: Text-based log with detailed debug information and formatted messages.
        *   `json_info.log`: Structured JSON log containing all log records with levels, timestamps, messages, and bound contextual data (e.g., `task_id`, `agent_name`, `component`). This is useful for programmatic analysis of agent behavior.
    *   Console output is also provided, often using `rich` for better readability, and its verbosity can be controlled. Custom panel prints (e.g., for ACR messages, retrieval results) also route a structured version of their content to the file logs.

*   **Frontend:**
    *   User interactions, stream processing events, and errors in the frontend (`demo_vis/front/app/page.tsx`) are logged.
    *   A `logToBackend` utility function sends these logs (level, message, component, function, context, frontend_timestamp) to the `/api/log_frontend_event` endpoint on the backend.
    *   The backend then records these frontend events into its standard logging system (including the per-task `json_info.log`), allowing for centralized analysis of both frontend and backend activity.

##  주요 모듈 (Key Modules - Backend)

*   **`app/main.py`**: Entry point for CLI operations and definition of the FastAPI application for API services.
*   **`app/config.py`**: Global configuration variables.
*   **`app/log.py`**: Logging utilities and rich console printing functions.
*   **`app/data_structures.py`**: Core data classes used across the application.
*   **`app/agents/`**: Directory containing the logic for different specialized agents (e.g., `PatchAgent`, `TestAgent`, `ReviewAgent`).
*   **`app/api/`**: Modules related to backend API logic, evaluation helpers, and validation.
*   **`app/inference.py`**: Manages the overall workflow for running a task with agents.
*   **`demo_vis/`**: Contains the frontend application (`demo_vis/front`) and potentially backend scripts specific to running the demo.

## Testing

Please refer to [TESTING.md](TESTING.md) for information on setting up and running tests.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
You can also join our [Discord server](https://discord.gg/ScXsdE49JY) for discussions.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✉️ Contacts

For any queries, you are welcome to open an issue.
Alternatively, contact us at acr@sonarsource.com


## Acknowledgements

This work was partially supported by a Singapore Ministry of Education (MoE) Tier 3 grant "Automated Program Repair", MOE-MOET32021-0001.
