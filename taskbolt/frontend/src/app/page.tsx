import Layout from '@/components/Layout'; // Using import alias
import ChatInterface from '@/components/ChatInterface';
import RepoViewPanel from '@/components/RepoViewPanel';
import TaskHistoryPanel from '@/components/TaskHistoryPanel';

export default function HomePage() {
  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-[calc(100vh-150px)]"> {/* Adjusted height */}
        {/* Left Column: RepoView and TaskHistory */}
        <div className="lg:col-span-1 flex flex-col space-y-4">
          <div className="flex-1">
            <RepoViewPanel />
          </div>
          <div className="flex-1">
            <TaskHistoryPanel />
          </div>
        </div>

        {/* Right Column: ChatInterface (takes more space) */}
        <div className="lg:col-span-2 h-full">
          <ChatInterface />
        </div>
      </div>
    </Layout>
  );
}
