import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { HomePage } from '@/components/home/HomePage';
import { LobbyPage } from '@/components/lobby/LobbyPage';
import { GamePage } from '@/components/game/GamePage';

function App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}

export default App;
