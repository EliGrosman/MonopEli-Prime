import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react';
import { useSessionStore } from './sessionStore';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Reset store between tests
beforeEach(() => {
  localStorageMock.clear();
  act(() => {
    useSessionStore.getState().clearSession();
  });
});

describe('sessionStore', () => {
  describe('initial state', () => {
    it('has correct initial values', () => {
      const state = useSessionStore.getState();
      expect(state.sessionId).toBeNull();
      expect(state.displayName).toBe('Player');
      expect(state.currentGameId).toBeNull();
      expect(state.currentLobbyId).toBeNull();
      expect(state.playerId).toBeNull();
    });
  });

  describe('setSession', () => {
    it('sets sessionId and displayName', () => {
      act(() => {
        useSessionStore.getState().setSession('session-123', 'Alice');
      });

      const state = useSessionStore.getState();
      expect(state.sessionId).toBe('session-123');
      expect(state.displayName).toBe('Alice');
    });
  });

  describe('setDisplayName', () => {
    it('updates displayName only', () => {
      act(() => {
        useSessionStore.getState().setSession('session-123', 'Alice');
        useSessionStore.getState().setDisplayName('Bob');
      });

      const state = useSessionStore.getState();
      expect(state.sessionId).toBe('session-123');
      expect(state.displayName).toBe('Bob');
    });
  });

  describe('setCurrentGame', () => {
    it('sets gameId and playerId', () => {
      act(() => {
        useSessionStore.getState().setCurrentGame('game-abc', 2);
      });

      const state = useSessionStore.getState();
      expect(state.currentGameId).toBe('game-abc');
      expect(state.playerId).toBe(2);
    });

    it('clears lobbyId when joining game', () => {
      act(() => {
        useSessionStore.getState().setCurrentLobby('lobby-xyz');
        useSessionStore.getState().setCurrentGame('game-abc', 1);
      });

      const state = useSessionStore.getState();
      expect(state.currentGameId).toBe('game-abc');
      expect(state.currentLobbyId).toBeNull();
    });

    it('can clear game by setting null', () => {
      act(() => {
        useSessionStore.getState().setCurrentGame('game-abc', 1);
        useSessionStore.getState().setCurrentGame(null, null);
      });

      const state = useSessionStore.getState();
      expect(state.currentGameId).toBeNull();
      expect(state.playerId).toBeNull();
    });
  });

  describe('setCurrentLobby', () => {
    it('sets lobbyId', () => {
      act(() => {
        useSessionStore.getState().setCurrentLobby('lobby-xyz');
      });

      expect(useSessionStore.getState().currentLobbyId).toBe('lobby-xyz');
    });

    it('clears game when entering lobby', () => {
      act(() => {
        useSessionStore.getState().setCurrentGame('game-abc', 1);
        useSessionStore.getState().setCurrentLobby('lobby-xyz');
      });

      const state = useSessionStore.getState();
      expect(state.currentLobbyId).toBe('lobby-xyz');
      expect(state.currentGameId).toBeNull();
      expect(state.playerId).toBeNull();
    });
  });

  describe('clearSession', () => {
    it('resets all state to initial values', () => {
      act(() => {
        useSessionStore.getState().setSession('session-123', 'Alice');
        useSessionStore.getState().setCurrentGame('game-abc', 1);
        useSessionStore.getState().clearSession();
      });

      const state = useSessionStore.getState();
      expect(state.sessionId).toBeNull();
      expect(state.displayName).toBe('Player');
      expect(state.currentGameId).toBeNull();
      expect(state.currentLobbyId).toBeNull();
      expect(state.playerId).toBeNull();
    });
  });

  describe('computed properties', () => {
    it('isLoggedIn returns true when sessionId exists', () => {
      expect(useSessionStore.getState().isLoggedIn()).toBe(false);

      act(() => {
        useSessionStore.getState().setSession('session-123', 'Alice');
      });

      expect(useSessionStore.getState().isLoggedIn()).toBe(true);
    });

    it('isInGame returns true when gameId exists', () => {
      expect(useSessionStore.getState().isInGame()).toBe(false);

      act(() => {
        useSessionStore.getState().setCurrentGame('game-abc', 1);
      });

      expect(useSessionStore.getState().isInGame()).toBe(true);
    });

    it('isInLobby returns true when lobbyId exists', () => {
      expect(useSessionStore.getState().isInLobby()).toBe(false);

      act(() => {
        useSessionStore.getState().setCurrentLobby('lobby-xyz');
      });

      expect(useSessionStore.getState().isInLobby()).toBe(true);
    });
  });
});
