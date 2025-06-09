import React from 'react';
import HamburgerMenu from './HamburgerMenu';
import SettingsDropdown from './SettingsDropdown';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <header className="bg-blue-600 dark:bg-blue-800 text-white p-4 shadow-md sticky top-0 z-50">
        <div className="container mx-auto flex justify-between items-center">
          <div className="text-xl font-semibold">Taskbolt Agent UI</div>
          <div className="flex items-center space-x-4">
            <div className="hidden md:block">
              <SettingsDropdown />
            </div>
            <HamburgerMenu />
          </div>
        </div>
      </header>
      <main className="flex-grow container mx-auto p-4">
        {children}
      </main>
      <footer className="bg-gray-200 dark:bg-gray-700 p-2 text-center text-sm sticky bottom-0 z-40">
        Status: <span className="font-semibold">Idle (Overall System Placeholder)</span>
      </footer>
    </div>
  );
};
export default Layout;
