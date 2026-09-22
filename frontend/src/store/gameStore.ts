import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { GameState, GameEvent, PlayerState, PropertyState } from '@/types';
import { getColorGroupPositions } from '@/utils/board';

// Default player colors
const PLAYER_COLORS = [
  '#e74c3c', // Red
  '#3498db', // Blue
  '#2ecc71', // Green
  '#f39c12', // Orange
  '#9b59b6', // Purple
  '#1abc9c', // Teal
  '#e91e63', // Pink
  '#795548', // Brown
];

function getPlayerColor(playerId: number): string {
  return PLAYER_COLORS[playerId % PLAYER_COLORS.length];
}

interface GameStore {
  // State
  gameId: string | null;
  gameState: GameState | null;
  events: GameEvent[];
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  pendingRequestId: string | null;

  // Actions
  setGameId: (id: string | null) => void;
  updateGameState: (state: Record<string, unknown>) => void;
  addEvent: (event: GameEvent) => void;
  clearEvents: () => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setPendingRequest: (requestId: string | null) => void;
  reset: () => void;

  // Selectors (computed state)
  getCurrentPlayer: () => PlayerState | null;
  getPlayer: (playerId: number) => PlayerState | null;
  getPropertyOwner: (position: number) => PlayerState | null;
  getProperty: (position: number) => PropertyState | null;
  canAfford: (playerId: number, amount: number) => boolean;
  hasMonopoly: (playerId: number, colorGroup: string) => boolean;
  getPlayerProperties: (playerId: number) => PropertyState[];
  getRecentEvents: (count?: number) => GameEvent[];
}

const MAX_EVENTS = 100;

const initialState = {
  gameId: null,
  gameState: null,
  events: [] as GameEvent[],
  isConnected: false,
  isLoading: false,
  error: null,
  pendingRequestId: null,
};

export const useGameStore = create<GameStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setGameId: (id) => set({ gameId: id }),

      updateGameState: (rawState) => {
        // Transform snake_case from backend to camelCase for frontend
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const raw = rawState as any;
        const currentRevision = get().gameState?.revision ?? -1;
        if (typeof raw.revision === 'number' && raw.revision < currentRevision) return;

        // Transform players
        const players = (raw.players || []).map((p: Record<string, unknown>) => ({
          id: p.id,
          name: p.name,
          money: p.money,
          position: p.position,
          inJail: p.in_jail,
          jailTurns: p.jail_turns,
          jailCards: p.jail_cards,
          bankrupt: p.bankrupt,
          isAi: p.is_ai ?? false,
          color: p.color ?? getPlayerColor(p.id as number),
          properties: p.properties ?? [],
        }));

        // Transform properties (keys are strings from JSON)
        const properties: Record<number, PropertyState> = {};
        for (const [key, value] of Object.entries(raw.properties || {})) {
          const prop = value as Record<string, unknown>;
          properties[Number(key)] = {
            position: prop.position as number,
            owner: prop.owner as number | null,
            houses: prop.houses as number,
            mortgaged: prop.mortgaged as boolean,
          };
        }

        // Determine if doubles were rolled on the last roll
        const rolledDoubles = raw.last_roll && raw.last_roll[0] === raw.last_roll[1];
        // doubles_count > 0 means we rolled doubles and must roll again
        const doublesCount = raw.doubles_count ?? 0;

        // Compute game phase based on state
        // Phase is always post_roll after any roll - buy/pay/etc happens in post_roll
        // The rolledDoubles flag indicates if player must roll again
        let gamePhase: string;
        if (raw.game_over) {
          gamePhase = 'game_over';
        } else if (raw.game_phase) {
          // Use backend's phase if provided
          gamePhase = raw.game_phase;
        } else {
          // post_roll if we've rolled, pre_roll otherwise
          gamePhase = raw.last_roll ? 'post_roll' : 'pre_roll';
        }

        const state: GameState = {
          ...raw,
          players,
          properties,
          // Map snake_case to camelCase aliases
          currentPlayer: raw.current_player,
          turnNumber: raw.turn_number,
          housesRemaining: raw.houses_remaining,
          hotelsRemaining: raw.hotels_remaining,
          gameOver: raw.game_over,
          revision: raw.revision ?? 0,
          lastRoll: raw.last_roll
            ? {
                die1: raw.last_roll[0],
                die2: raw.last_roll[1],
                isDoubles: rolledDoubles,
              }
            : null,
          gamePhase,
          rolledDoubles: rolledDoubles ?? false,
          doublesCount,
        };
        set({
          gameState: state,
          error: null,
        });
      },

      setConnected: (connected) => set({ isConnected: connected }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      setPendingRequest: (pendingRequestId) => set({ pendingRequestId }),

      addEvent: (event) =>
        set((state) => ({
          events: [...state.events, event].slice(-MAX_EVENTS),
        })),

      clearEvents: () => set({ events: [] }),

      reset: () => set(initialState),

      // Selectors
      getCurrentPlayer: () => {
        const { gameState } = get();
        if (!gameState || gameState.currentPlayer == null) return null;
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
          colorGroup as
            | 'brown'
            | 'lightblue'
            | 'magenta'
            | 'orange'
            | 'red'
            | 'yellow'
            | 'green'
            | 'blue'
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

      getRecentEvents: (count = 20) => {
        const { events } = get();
        return events.slice(-count).reverse();
      },
    }),
    { name: 'game-store' }
  )
);
