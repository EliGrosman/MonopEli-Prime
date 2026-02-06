import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { HomePage } from '@/components/home/HomePage';
import { LobbyPage } from '@/components/lobby/LobbyPage';
import { GamePage } from '@/components/game/GamePage';
import { ToastContainer, GlobalLoading } from '@/components/common';
import { useUIStore } from '@/store';

function AppContent() {
  const { toasts, removeToast, globalLoading, loadingMessage } = useUIStore();

  return (
    <>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lobby" element={<LobbyPage />} />
          <Route path="/lobby/create" element={<LobbyPage />} />
          <Route path="/lobby/join" element={<LobbyPage />} />
          <Route path="/lobby/:lobbyId" element={<LobbyPage />} />
          <Route path="/game/:gameId" element={<GamePage />} />
          <Route path="/game" element={<GamePage />} />
        </Routes>
      </Layout>
      <ToastContainer toasts={toasts} onClose={removeToast} />
      <GlobalLoading isLoading={globalLoading} message={loadingMessage} />
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
