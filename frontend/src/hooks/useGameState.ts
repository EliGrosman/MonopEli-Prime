import { useCallback, useMemo } from 'react';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useWebSocket } from './useWebSocket';
import type { WSMessage, WSMessageType } from '@/types';

/**
 * Game state hook that combines store and WebSocket.
 *
 * Provides access to game state, connection status, and computed values.
 */
export function useGameState() {
  const { currentGameId, playerId } = useSessionStore();
  const {
    gameState,
    isConnected,
    isLoading,
    error,
    getCurrentPlayer,
    getPlayer,
    getPropertyOwner,
    getProperty,
    canAfford,
    hasMonopoly,
    getPlayerProperties,
  } = useGameStore();

  const handleMessage = useCallback((message: WSMessage) => {
    // Handle game-specific messages
    switch (message.type as WSMessageType) {
      case 'game_over':
        // Handle game over - could show modal, etc.
        console.log('Game over:', message.data);
        break;
      case 'player_joined':
      case 'player_left':
        // Handle player changes
        console.log('Player change:', message.type, message.data);
        break;
    }
  }, []);

  const { connectionState, send, reconnect } = useWebSocket({
    gameId: currentGameId ?? '',
    playerId: playerId ?? undefined,
    onMessage: handleMessage,
    autoConnect: !!currentGameId,
  });

  // Computed values
  const currentPlayer = getCurrentPlayer();
  const isMyTurn = useMemo(
    () => gameState?.decision_player === playerId,
    [gameState?.decision_player, playerId]
  );
  const myPlayer = playerId !== null ? getPlayer(playerId) : null;

  return {
    // State
    gameState,
    isConnected,
    isLoading,
    error,
    connectionState,

    // Computed values
    currentPlayer,
    isMyTurn,
    myPlayer,
    playerId,
    gameId: currentGameId,

    // Methods
    reconnect,
    send,

    // Selectors
    getPlayer,
    getPropertyOwner,
    getProperty,
    canAfford,
    hasMonopoly,
    getPlayerProperties,
  };
}
