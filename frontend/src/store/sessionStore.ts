import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

interface SessionStore {
  // State
  sessionId: string | null;
  displayName: string;
  currentGameId: string | null;
  currentLobbyId: string | null;
  playerId: number | null;

  // Actions
  setSession: (sessionId: string, displayName: string) => void;
  setDisplayName: (name: string) => void;
  setCurrentGame: (gameId: string | null, playerId: number | null) => void;
  setCurrentLobby: (lobbyId: string | null) => void;
  clearSession: () => void;

  // Computed
  isLoggedIn: () => boolean;
  isInGame: () => boolean;
  isInLobby: () => boolean;
}

export const useSessionStore = create<SessionStore>()(
  devtools(
    persist(
      (set, get) => ({
        sessionId: null,
        displayName: 'Player',
        currentGameId: null,
        currentLobbyId: null,
        playerId: null,

        setSession: (sessionId, displayName) =>
          set({
            sessionId,
            displayName,
          }),

        setDisplayName: (displayName) => set({ displayName }),

        setCurrentGame: (currentGameId, playerId) =>
          set({
            currentGameId,
            playerId,
            currentLobbyId: null, // Clear lobby when joining game
          }),

        setCurrentLobby: (currentLobbyId) =>
          set({
            currentLobbyId,
            currentGameId: null, // Clear game when in lobby
            playerId: null,
          }),

        clearSession: () =>
          set({
            sessionId: null,
            displayName: 'Player',
            currentGameId: null,
            currentLobbyId: null,
            playerId: null,
          }),

        // Computed
        isLoggedIn: () => get().sessionId !== null,
        isInGame: () => get().currentGameId !== null,
        isInLobby: () => get().currentLobbyId !== null,
      }),
      {
        name: 'monopoly-session',
        partialize: (state) => ({
          sessionId: state.sessionId,
          displayName: state.displayName,
        }),
      }
    ),
    { name: 'session-store' }
  )
);
