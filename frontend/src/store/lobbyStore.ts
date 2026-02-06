import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { LobbyState, LobbyListItem, LobbySettings } from '@/types';

interface LobbyStore {
  // State
  currentLobby: LobbyState | null;
  lobbyList: LobbyListItem[];
  isLoading: boolean;
  error: string | null;

  // Actions
  setCurrentLobby: (lobby: LobbyState | null) => void;
  updateLobby: (updates: Partial<LobbyState>) => void;
  setLobbyList: (lobbies: LobbyListItem[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;

  // Player actions
  addPlayer: (player: LobbyState['players'][0]) => void;
  removePlayer: (sessionId: string) => void;
  removePlayerBySlot: (slotId: number) => void;
  updatePlayerReady: (sessionId: string, isReady: boolean) => void;

  // Settings
  updateSettings: (settings: Partial<LobbySettings>) => void;

  // Computed
  isHost: (sessionId: string) => boolean;
  canStartGame: () => boolean;
  getMySlot: (sessionId: string) => LobbyState['players'][0] | null;
}

const initialState = {
  currentLobby: null,
  lobbyList: [],
  isLoading: false,
  error: null,
};

export const useLobbyStore = create<LobbyStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setCurrentLobby: (lobby) => set({ currentLobby: lobby, error: null }),

      updateLobby: (updates) =>
        set((state) => ({
          currentLobby: state.currentLobby
            ? { ...state.currentLobby, ...updates }
            : null,
        })),

      setLobbyList: (lobbies) => set({ lobbyList: lobbies }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      reset: () => set(initialState),

      // Player actions
      addPlayer: (player) =>
        set((state) => {
          if (!state.currentLobby) return state;
          return {
            currentLobby: {
              ...state.currentLobby,
              players: [...state.currentLobby.players, player],
            },
          };
        }),

      removePlayer: (sessionId) =>
        set((state) => {
          if (!state.currentLobby) return state;
          return {
            currentLobby: {
              ...state.currentLobby,
              players: state.currentLobby.players.filter(
                (p) => p.session_id !== sessionId
              ),
            },
          };
        }),

      removePlayerBySlot: (slotId) =>
        set((state) => {
          if (!state.currentLobby) return state;
          return {
            currentLobby: {
              ...state.currentLobby,
              players: state.currentLobby.players.filter(
                (p) => p.slot_id !== slotId
              ),
            },
          };
        }),

      updatePlayerReady: (sessionId, isReady) =>
        set((state) => {
          if (!state.currentLobby) return state;
          return {
            currentLobby: {
              ...state.currentLobby,
              players: state.currentLobby.players.map((p) =>
                p.session_id === sessionId ? { ...p, is_ready: isReady } : p
              ),
            },
          };
        }),

      // Settings
      updateSettings: (settings) =>
        set((state) => {
          if (!state.currentLobby) return state;
          return {
            currentLobby: {
              ...state.currentLobby,
              settings: { ...state.currentLobby.settings, ...settings },
            },
          };
        }),

      // Computed
      isHost: (sessionId) => {
        const { currentLobby } = get();
        return currentLobby?.host_session_id === sessionId;
      },

      canStartGame: () => {
        const { currentLobby } = get();
        if (!currentLobby?.players) return false;
        if (currentLobby.players.length < 2) return false;
        return currentLobby.players.every((p) => p.is_ready || p.is_ai);
      },

      getMySlot: (sessionId) => {
        const { currentLobby } = get();
        if (!currentLobby?.players) return null;
        return currentLobby.players.find((p) => p.session_id === sessionId) ?? null;
      },
    }),
    { name: 'lobby-store' }
  )
);
