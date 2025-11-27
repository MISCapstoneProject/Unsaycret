import { useState } from 'react';
import Layout from './components/Layout';
import SilentMode from './components/SilentMode';
import UsersMode from './components/UsersMode';
import SessionsMode from './components/SessionsMode';
import { RecordingProvider } from './contexts/RecordingContext';

type AppMode = 'silent' | 'users' | 'sessions';

function App() {
  const [currentMode, setCurrentMode] = useState<AppMode>('silent');

  const renderCurrentMode = () => {
    switch (currentMode) {
      case 'silent':
        return <SilentMode />;
      case 'users':
        return <UsersMode />;
      case 'sessions':
        return <SessionsMode />;
      default:
        return <SilentMode />;
    }
  };

  return (
    <RecordingProvider>
      <Layout currentMode={currentMode} onModeChange={setCurrentMode as any}>
        {renderCurrentMode()}
      </Layout>
    </RecordingProvider>
  );
}

export default App;