import { useParams } from 'react-router-dom';
import { useEffect } from 'react';
import { Board } from '@/components/board/Board';
import { ConnectionStatus, Loading } from '@/components/common';
import { useGameState } from '@/hooks/useGameState';
import { useSessionStore } from '@/store/sessionStore';
import { useUIStore } from '@/store';

/**
 * Main game page - displays the game board and controls.
 */
export function GamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const { setCurrentGame, playerId } = useSessionStore();
  const { addToast } = useUIStore();
  const {
    gameState,
    isLoading,
    error,
    connectionState,
    reconnect,
    isMyTurn,
    myPlayer,
    currentPlayer,
  } = useGameState();

  // Set game ID from URL params
  useEffect(() => {
    if (gameId) {
      // Keep existing playerId when navigating to game
      setCurrentGame(gameId, playerId);
    }
  }, [gameId, playerId, setCurrentGame]);

  // Show error toasts
  useEffect(() => {
    if (error) {
      addToast(error, 'error');
    }
  }, [error, addToast]);

  // Show loading state while connecting
  if (isLoading || connectionState === 'connecting') {
    return (
      <div className="container mx-auto p-4 flex items-center justify-center min-h-[60vh]">
        <Loading message="Connecting to game..." />
      </div>
    );
  }

  // Show reconnecting state
  if (connectionState === 'reconnecting') {
    return (
      <div className="container mx-auto p-4 flex items-center justify-center min-h-[60vh]">
        <Loading message="Reconnecting..." />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      {/* Connection Status Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">
            {gameId ? `Game: ${gameId.slice(0, 8)}...` : 'Monopoly'}
          </h1>
          <ConnectionStatus state={connectionState} onReconnect={reconnect} />
        </div>
        {isMyTurn && (
          <div className="px-3 py-1 bg-green-500 text-white rounded-full text-sm font-medium animate-pulse">
            Your Turn!
          </div>
        )}
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Board section */}
        <div className="flex-1 flex justify-center items-start">
          <Board />
        </div>

        {/* Sidebar */}
        <div className="lg:w-80 space-y-4">
          {/* Player Panel */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Players</h2>
            {gameState?.players && gameState.players.length > 0 ? (
              <div className="space-y-2">
                {gameState.players.map((player) => (
                  <div
                    key={player.id}
                    className={`p-2 rounded-lg flex items-center justify-between ${
                      player.id === currentPlayer?.id
                        ? 'bg-blue-100 border-2 border-blue-400'
                        : 'bg-gray-50'
                    } ${player.id === myPlayer?.id ? 'ring-2 ring-green-500' : ''}`}
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: getPlayerColor(player.id) }}
                      />
                      <span className="font-medium">
                        {player.name || `Player ${player.id + 1}`}
                        {player.id === myPlayer?.id && ' (You)'}
                      </span>
                    </div>
                    <span className="text-green-600 font-mono">${player.money}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">Waiting for players...</p>
            )}
          </div>

          {/* Action Panel placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Actions</h2>
            {isMyTurn ? (
              <p className="text-green-600 font-medium">
                It&apos;s your turn! Actions coming soon...
              </p>
            ) : (
              <p className="text-gray-500">
                {currentPlayer
                  ? `Waiting for ${currentPlayer.name || `Player ${currentPlayer.id + 1}`}...`
                  : 'Actions will appear here...'}
              </p>
            )}
          </div>

          {/* Event Log placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Event Log</h2>
            <p className="text-gray-500">Game events will appear here...</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper to get player color
function getPlayerColor(playerId: number): string {
  const colors = [
    '#E53935', // Red
    '#1E88E5', // Blue
    '#43A047', // Green
    '#FDD835', // Yellow
    '#8E24AA', // Purple
    '#FB8C00', // Orange
    '#00ACC1', // Cyan
    '#6D4C41', // Brown
  ];
  return colors[playerId % colors.length];
}
