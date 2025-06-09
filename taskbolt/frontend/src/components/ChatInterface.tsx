'use client'; // For client-side interactions

import React, { useState, useEffect, FormEvent } from 'react';

interface LogEntry {
  message: string;
}

interface TaskStatus {
  task_id: string;
  status: string;
  details?: string;
}

interface TaskLog {
  task_id: string;
  logs: string[]; // Assuming logs are strings as per API mock
}

const ChatInterface: React.FC = () => {
  const [repoUrl, setRepoUrl] = useState<string>('https://github.com/example/repo.git');
  const [taskDescription, setTaskDescription] = useState<string>('Fix a critical bug in the login module.');

  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string>('Idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const API_BASE_URL = 'http://localhost:8000'; // Assuming API runs on port 8000

  useEffect(() => {
    if (!currentTaskId) return;

    let isMounted = true;
    let pollingTimeoutId: NodeJS.Timeout | undefined = undefined;

    const fetchStatusAndLogs = async () => {
      if (!isMounted) return;
      try {
        const statusRes = await fetch(`${API_BASE_URL}/api/agent/status/${currentTaskId}`);
        if (!statusRes.ok) {
          const errorData = await statusRes.json().catch(() => ({ detail: "Error fetching status, invalid JSON response." }));
          throw new Error(errorData.detail || `Error fetching status: ${statusRes.status}`);
        }
        const statusData: TaskStatus = await statusRes.json();
        if (isMounted) setCurrentStatus(statusData.status + (statusData.details ? ` (${statusData.details})` : ''));

        const logsRes = await fetch(`${API_BASE_URL}/api/agent/logs/${currentTaskId}`);
         if (!logsRes.ok) {
          const errorData = await logsRes.json().catch(() => ({ detail: "Error fetching logs, invalid JSON response." }));
          throw new Error(errorData.detail || `Error fetching logs: ${logsRes.status}`);
        }
        const logsData: TaskLog = await logsRes.json();
        if (isMounted) setLogs(logsData.logs);

        if (isMounted && (statusData.status === 'processing' || statusData.status === 'queued')) {
          pollingTimeoutId = setTimeout(fetchStatusAndLogs, 3000);
        }
      } catch (err: any) {
        console.error("Polling error:", err);
        if (isMounted) {
          setError(err.message);
          setCurrentStatus('Error polling data');
        }
      }
    };

    pollingTimeoutId = setTimeout(fetchStatusAndLogs, 100);

    return () => {
      isMounted = false;
      if (pollingTimeoutId) {
        clearTimeout(pollingTimeoutId);
      }
    };
  }, [currentTaskId]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setLogs([]);
    setCurrentStatus('Submitting...');
    setIsLoading(true);
    setCurrentTaskId(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repository_link: repoUrl, task_description: taskDescription }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Error submitting task, invalid JSON response." }));
        throw new Error(errorData.detail || `Error submitting task: ${response.status}`);
      }

      const data = await response.json();
      if (data.task_id) {
        setCurrentTaskId(data.task_id);
        setCurrentStatus(data.status || 'Task Submitted');
      } else {
        throw new Error("Task ID not received from server.");
      }
    } catch (err: any) {
      console.error("Submission error:", err);
      setError(err.message);
      setCurrentStatus('Submission Failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6 h-full flex flex-col">
      <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Chat & Task Submission</h2>
      <form onSubmit={handleSubmit} className="mb-4">
        {/* Inputs and button as before */}
        <div className="mb-2">
          <label htmlFor="repoUrl" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Repository URL:</label>
          <input type="text" id="repoUrl" value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)} className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white" required disabled={isLoading} />
        </div>
        <div className="mb-3">
          <label htmlFor="taskDescription" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Task Description:</label>
          <textarea id="taskDescription" value={taskDescription} onChange={(e) => setTaskDescription(e.target.value)} rows={3} className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white" required disabled={isLoading} />
        </div>
        <button type="submit" className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50" disabled={isLoading}>
          {isLoading ? 'Submitting...' : 'Submit Task'}
        </button>
      </form>
      {error && <p className="text-red-500 text-sm mb-2">Error: {error}</p>}
      <div className="mb-2">
        <p className="text-sm text-gray-600 dark:text-gray-400">Task ID: <span className="font-semibold text-gray-800 dark:text-white">{currentTaskId || 'N/A'}</span></p>
        <p className="text-sm text-gray-600 dark:text-gray-400">Status: <span className="font-semibold text-gray-800 dark:text-white">{currentStatus}</span></p>
      </div>
      <div className="flex-grow border rounded-md p-3 bg-gray-50 dark:bg-gray-700 overflow-y-auto text-sm">
        <h3 className="text-md font-semibold mb-1 text-gray-700 dark:text-gray-300">Logs:</h3>
        {logs.length > 0 ? logs.map((log, index) => (<p key={index} className="text-gray-600 dark:text-gray-400 font-mono text-xs">{log}</p>)) : (<p className="text-gray-400 dark:text-gray-500">No logs yet.</p>)}
      </div>
    </div>
  );
};
export default ChatInterface;
