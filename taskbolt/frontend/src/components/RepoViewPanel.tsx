import React from 'react';

const RepoViewPanel: React.FC = () => {
  return (
    <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Repository View</h2>
      <div className="border rounded-md p-2 h-64 bg-gray-50 dark:bg-gray-700 overflow-y-auto">
        {/* File tree and file content will go here */}
        <p className="text-sm text-gray-500 dark:text-gray-400">File tree placeholder</p>
      </div>
    </div>
  );
};

export default RepoViewPanel;
