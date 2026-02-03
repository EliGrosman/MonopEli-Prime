import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { GameState, PlayerState, PropertyState } from '@/types';
import { getColorGroupPositions } from '@/utils/board';

interface GameStore {
  // State
  gameId: string | null;
  gameState: GameState | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setGameId: (id: string | null) => void;
  updateGameState: (state: GameState) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;

  // Selectors (computed state)
  getCurrentPlayer: () => PlayerState | null;
  getPlayer: (playerId: number) => PlayerState | null;
  getPropertyOwner: (position: number) => PlayerState | null;
  getProperty: (position: number) => PropertyState | null;
  canAfford: (playerId: number, amount: number) => boolean;
  hasMonopoly: (playerId: number, colorGroup: string) => boolean;
  getPlayerProperties: (playerId: number) => PropertyState[];
}

const initialState = {
  gameId: null,
  gameState: null,
  isConnected: false,
  isLoading: false,
  error: null,
};

export const useGameStore = create<GameStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setGameId: (id) => set({ gameId: id }),

      updateGameState: (state) =>
        set({
          gameState: state,
          error: null,
        }),

      setConnected: (connected) => set({ isConnected: connected }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      reset: () => set(initialState),

      // Selectors
      getCurrentPlayer: () => {
        const { gameState } = get();
        if (!gameState) return null;
        return gameState.players[gameState.currentPlayer] ?? null;
      },

      getPlayer: (playerId) => {
        const { gameState } = get();
        if (!gameState) return null;
        return gameState.players.find((p) => p.id === playerId) ?? null;
      },

      getPropertyOwner: (position) => {
        const { gameState } = get();
        if (!gameState) return null;
        const prop = gameState.properties[position];
        if (!prop?.owner && prop?.owner !== 0) return null;
        return gameState.players.find((p) => p.id === prop.owner) ?? null;
      },

      getProperty: (position) => {
        const { gameState } = get();
        if (!gameState) return null;
        return gameState.properties[position] ?? null;
      },

      canAfford: (playerId, amount) => {
        const { gameState } = get();
        if (!gameState) return false;
        const player = gameState.players.find((p) => p.id === playerId);
        return player ? player.money >= amount : false;
      },

      hasMonopoly: (playerId, colorGroup) => {
        const { gameState } = get();
        if (!gameState) return false;

        const colorPositions = getColorGroupPositions(
          colorGroup as 'brown' | 'lightblue' | 'magenta' | 'orange' | 'red' | 'yellow' | 'green' | 'blue'
        );
        return colorPositions.every((pos) => {
          const prop = gameState.properties[pos];
          return prop?.owner === playerId;
        });
      },

      getPlayerProperties: (playerId) => {
        const { gameState } = get();
        if (!gameState) return [];
        return Object.values(gameState.properties).filter((p) => p.owner === playerId);
      },
    }),
    { name: 'game-store' }
  )
);
