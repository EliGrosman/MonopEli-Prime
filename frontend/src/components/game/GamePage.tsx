import { useParams } from 'react-router-dom';
import { useEffect, useState, useMemo } from 'react';
import { Board } from '@/components/board/Board';
import { ConnectionStatus, Loading } from '@/components/common';
import { PropertyModal } from '@/components/property';
import { PlayerPanel } from './PlayerPanel';
import { ActionPanel } from './ActionPanel';
import { useGameState } from '@/hooks/useGameState';
import { useSessionStore } from '@/store/sessionStore';
import { useGameStore } from '@/store/gameStore';
import { useUIStore } from '@/store';

/**
 * Main game page - displays the game board and controls.
 */
export function GamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const { setCurrentGame, playerId } = useSessionStore();
  const { addToast } = useUIStore();
  const { gameState } = useGameStore();
  const {
    isLoading,
    error,
    connectionState,
    reconnect,
    isMyTurn,
    send,
  } = useGameState();

  // Property modal state
  const [selectedProperty, setSelectedProperty] = useState<number | null>(null);
  const isPropertyModalOpen = selectedProperty !== null;

  // Generate CSS variables for player colors
  const players = gameState?.players;
  const playerColorStyle = useMemo(() => {
    if (!players) return {};
    const styles: Record<string, string> = {};
    players.forEach((player) => {
      styles[`--player-${player.id}-color`] = player.color;
    });
    return styles;
  }, [players]);

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

  // Handle property space click
  const handleSpaceClick = (position: number) => {
    // Only open modal for ownable spaces (properties, railroads, utilities)
    const property = gameState?.properties[position];
    if (property !== undefined) {
      setSelectedProperty(position);
    }
  };

  return (
    <div className="container mx-auto p-4" style={playerColorStyle}>
      {/* Connection Status Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-gray-900">
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
          <Board
            players={gameState?.players}
            properties={gameState?.properties}
            onSpaceClick={handleSpaceClick}
          />
        </div>

        {/* Sidebar */}
        <div className="lg:w-96 space-y-4">
          {/* Player Panel */}
          <PlayerPanel />

          {/* Action Panel */}
          <ActionPanel send={send} />

          {/* Event Log placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4 text-gray-900">Event Log</h2>
            <p className="text-gray-500">Game events will appear here...</p>
          </div>
        </div>
      </div>

      {/* Property Modal */}
      <PropertyModal
        position={selectedProperty}
        isOpen={isPropertyModalOpen}
        onClose={() => setSelectedProperty(null)}
        send={send}
      />
    </div>
  );
}
