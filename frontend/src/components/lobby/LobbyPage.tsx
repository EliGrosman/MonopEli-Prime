import { useState, useCallback, useEffect } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { useLobbyState } from '@/hooks';
import { useSession } from '@/hooks';
import { LobbyList } from './LobbyList';
import { LobbyRoom } from './LobbyRoom';
import type { CreateLobbyRequest } from '@/api/lobbies';

// --- Login Form Component ---

interface LoginFormProps {
  onLogin: (name: string) => Promise<unknown>;
}

function LoginForm({ onLogin }: LoginFormProps) {
  const [name, setName] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || isLoading) return;
    setIsLoading(true);
    try {
      await onLogin(name.trim());
    } catch {
      // Error handled by useSession hook (shows toast)
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6 mb-8">
      <h2 className="text-lg font-semibold text-gray-900 mb-2">Welcome to MonopEli!</h2>
      <p className="text-gray-700 text-sm mb-4">Enter your name to start playing</p>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your display name"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent"
          maxLength={20}
          autoFocus
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!name.trim() || isLoading}
          className={`
            w-full py-2 rounded-lg font-semibold transition-colors
            ${
              name.trim() && !isLoading
                ? 'bg-board-border text-white hover:bg-green-800'
                : 'bg-gray-300 text-gray-700 cursor-not-allowed'
            }
          `}
        >
          {isLoading ? 'Joining...' : 'Join Game Lobby'}
        </button>
      </form>
    </div>
  );
}

/**
 * Lobby page - handles lobby list, creation, joining, and room views.
 */
export function LobbyPage() {
  const { lobbyId } = useParams<{ lobbyId: string }>();
  const location = useLocation();
  const { sessionId, isLoggedIn, displayName, login } = useSession();

  const {
    currentLobby,
    lobbyList,
    isLoading,
    connectionState,
    isHost,
    fetchLobbyList,
    create,
    join,
    leave,
    toggleReady,
    addAi,
    removeAi,
    changeSettings,
    start,
  } = useLobbyState(lobbyId);

  // Determine which view to show based on route
  const isCreateRoute = location.pathname === '/lobby/create';
  const isJoinRoute = location.pathname === '/lobby/join';
  const isLobbyRoute = !!lobbyId;
  const isListRoute = !isCreateRoute && !isJoinRoute && !isLobbyRoute;

  // Load lobby list when on list route
  useEffect(() => {
    if (isListRoute) {
      fetchLobbyList();
    }
  }, [isListRoute, fetchLobbyList]);

  // If in a specific lobby, show the room view
  if (isLobbyRoute && currentLobby) {
    return (
      <div className="container mx-auto p-4 md:p-8">
        <LobbyRoom
          lobby={currentLobby}
          sessionId={sessionId || ''}
          isHost={isHost}
          isLoading={isLoading}
          connectionState={connectionState}
          onLeave={leave}
          onToggleReady={toggleReady}
          onAddAi={addAi}
          onRemoveAi={removeAi}
          onChangeSettings={changeSettings}
          onStart={start}
        />
      </div>
    );
  }

  // Loading state for specific lobby
  if (isLobbyRoute && isLoading) {
    return (
      <div className="container mx-auto p-4 md:p-8">
        <div className="max-w-2xl mx-auto text-center py-12">
          <div className="animate-spin w-8 h-8 border-4 border-board-border border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-600">Loading lobby...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 md:p-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
            {isCreateRoute ? 'Create Game' : isJoinRoute ? 'Join Game' : 'Game Lobby'}
          </h1>
          {isLoggedIn && (
            <p className="text-sm text-gray-800 mt-1">Playing as <span className="font-medium">{displayName}</span></p>
          )}
        </div>
        {isListRoute && isLoggedIn && (
          <Link
            to="/lobby/create"
            className="px-6 py-2 bg-board-border text-white rounded-lg font-semibold hover:bg-green-800 transition-colors"
          >
            Create New Game
          </Link>
        )}
        {(isCreateRoute || isJoinRoute) && (
          <Link
            to="/lobby"
            className="text-sm text-gray-600 hover:text-gray-700 flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to lobby
          </Link>
        )}
      </div>

      {/* Login form - show when not logged in (except on create route which handles its own login) */}
      {!isLoggedIn && !isCreateRoute && <LoginForm onLogin={login} />}

      {/* Create form - includes name entry, so no separate login needed */}
      {isCreateRoute && (
        <CreateLobbyForm
          displayName={displayName}
          onCreate={create}
          isLoading={isLoading}
        />
      )}

      {/* Join form */}
      {isJoinRoute && (
        <JoinLobbyForm
          isLoggedIn={isLoggedIn}
          onJoin={join}
          isLoading={isLoading}
        />
      )}

      {/* Lobby list */}
      {isListRoute && (
        <div className="bg-white rounded-lg shadow-md p-6">
          {/* Quick join by code */}
          <JoinByCode onJoin={join} isLoading={isLoading} isLoggedIn={isLoggedIn} />

          {/* Available games */}
          <div className="border-t pt-6 mt-6">
            <h2 className="text-lg font-semibold mb-4">Available Games</h2>
            <LobbyList
              lobbies={lobbyList}
              isLoading={isLoading}
              onJoin={(id) => join(id)}
              onRefresh={fetchLobbyList}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// --- Sub-components ---

interface CreateLobbyFormProps {
  displayName: string;
  onCreate: (request: CreateLobbyRequest) => Promise<unknown>;
  isLoading: boolean;
}

function CreateLobbyForm({ displayName, onCreate, isLoading }: CreateLobbyFormProps) {
  const [hostName, setHostName] = useState(displayName || '');
  const [gameName, setGameName] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [maxPlayers, setMaxPlayers] = useState(4);

  // Update default game name when host name changes
  const effectiveGameName = gameName.trim() || `${hostName || 'My'}'s Game`;

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!hostName.trim()) return;
      onCreate({
        name: effectiveGameName,
        host_name: hostName.trim(),
        settings: {
          private: isPrivate,
          max_players: maxPlayers,
        },
      });
    },
    [effectiveGameName, hostName, isPrivate, maxPlayers, onCreate]
  );

  const canSubmit = hostName.trim() && !isLoading;

  return (
    <div className="max-w-lg mx-auto bg-white rounded-lg shadow-md p-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Host name */}
        <div>
          <label htmlFor="hostName" className="block text-sm font-medium text-gray-700 mb-2">
            Your Display Name
          </label>
          <input
            type="text"
            id="hostName"
            value={hostName}
            onChange={(e) => setHostName(e.target.value)}
            placeholder="Enter your name"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent"
            maxLength={20}
            disabled={isLoading}
            autoFocus
          />
        </div>

        {/* Game name */}
        <div>
          <label htmlFor="gameName" className="block text-sm font-medium text-gray-700 mb-2">
            Game Name
          </label>
          <input
            type="text"
            id="gameName"
            value={gameName}
            onChange={(e) => setGameName(e.target.value)}
            placeholder={effectiveGameName}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent"
            disabled={isLoading}
          />
        </div>

        {/* Max players */}
        <div>
          <label htmlFor="maxPlayers" className="block text-sm font-medium text-gray-700 mb-2">
            Max Players
          </label>
          <select
            id="maxPlayers"
            value={maxPlayers}
            onChange={(e) => setMaxPlayers(parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent"
            disabled={isLoading}
          >
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n} value={n}>
                {n} Players
              </option>
            ))}
          </select>
        </div>

        {/* Private toggle */}
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            id="isPrivate"
            checked={isPrivate}
            onChange={(e) => setIsPrivate(e.target.checked)}
            className="w-4 h-4 text-board-border border-gray-300 rounded focus:ring-board-border"
            disabled={isLoading}
          />
          <div>
            <label htmlFor="isPrivate" className="text-sm font-medium text-gray-700">
              Private Game
            </label>
            <p className="text-xs text-gray-600">Only players with the code can join</p>
          </div>
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!canSubmit}
          className={`
            w-full py-3 rounded-lg font-semibold text-lg transition-colors
            ${
              canSubmit
                ? 'bg-board-border text-white hover:bg-green-800'
                : 'bg-gray-300 text-gray-700 cursor-not-allowed'
            }
          `}
        >
          {isLoading ? 'Creating...' : 'Create Game'}
        </button>
      </form>
    </div>
  );
}

interface JoinLobbyFormProps {
  isLoggedIn: boolean;
  onJoin: (id: string, code?: string) => Promise<boolean>;
  isLoading: boolean;
}

function JoinLobbyForm({ isLoggedIn, onJoin, isLoading }: JoinLobbyFormProps) {
  const [code, setCode] = useState('');

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!isLoggedIn || !code.trim()) return;

      // For private lobbies, the code is both the lobby ID and the invite code
      // The backend will handle this
      const success = await onJoin(code.trim().toUpperCase(), code.trim().toUpperCase());
      if (!success) {
        // Stay on page to show error
      }
    },
    [code, isLoggedIn, onJoin]
  );

  return (
    <div className="max-w-lg mx-auto bg-white rounded-lg shadow-md p-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="joinCode" className="block text-sm font-medium text-gray-700 mb-2">
            Game Code
          </label>
          <input
            type="text"
            id="joinCode"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            placeholder="Enter 6-character code"
            className="w-full px-4 py-3 text-center text-2xl font-mono tracking-widest border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent uppercase"
            maxLength={36}
            disabled={!isLoggedIn || isLoading}
          />
        </div>

        <button
          type="submit"
          disabled={!isLoggedIn || !code.trim() || isLoading}
          className={`
            w-full py-3 rounded-lg font-semibold text-lg transition-colors
            ${
              isLoggedIn && code.trim() && !isLoading
                ? 'bg-board-border text-white hover:bg-green-800'
                : 'bg-gray-300 text-gray-700 cursor-not-allowed'
            }
          `}
        >
          {isLoading ? 'Joining...' : 'Join Game'}
        </button>
      </form>
    </div>
  );
}

interface JoinByCodeProps {
  onJoin: (id: string, code?: string) => Promise<boolean>;
  isLoading: boolean;
  isLoggedIn: boolean;
}

function JoinByCode({ onJoin, isLoading, isLoggedIn }: JoinByCodeProps) {
  const [code, setCode] = useState('');

  const handleJoin = useCallback(() => {
    if (!code.trim() || !isLoggedIn) return;
    onJoin(code.trim().toUpperCase(), code.trim().toUpperCase());
  }, [code, isLoggedIn, onJoin]);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Join a Game</h2>
      <div className="flex gap-2">
        <input
          type="text"
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase())}
          placeholder="Enter game code"
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border text-gray-900 bg-white focus:border-transparent uppercase font-mono"
          maxLength={36}
          disabled={!isLoggedIn || isLoading}
        />
        <button
          onClick={handleJoin}
          disabled={!isLoggedIn || !code.trim() || isLoading}
          className={`
            px-6 py-2 rounded-lg font-semibold transition-colors
            ${
              isLoggedIn && code.trim() && !isLoading
                ? 'bg-board-border text-white hover:bg-green-800'
                : 'bg-gray-300 text-gray-700 cursor-not-allowed'
            }
          `}
        >
          {isLoading ? '...' : 'Join'}
        </button>
      </div>
    </div>
  );
}

export default LobbyPage;
