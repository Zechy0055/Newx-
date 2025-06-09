import React from 'react';

const TaskHistoryPanel: React.FC = () => {
  return (
    <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Task History & Logs</h2>
      <div className="border rounded-md p-2 h-64 bg-gray-50 dark:bg-gray-700 overflow-y-auto">
        {/* Task history items and logs will go here */}
        <p className="text-sm text-gray-500 dark:text-gray-400">Task history placeholder</p>
      </div>
    </div>
  );
};

export default TaskHistoryPanel;
