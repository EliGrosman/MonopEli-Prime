import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import { useLobbyStore } from '@/store/lobbyStore';
import type { LobbyState, LobbyPlayer } from '@/types';

function createMockPlayer(overrides: Partial<LobbyPlayer> = {}): LobbyPlayer {
  return {
    session_id: 'session-1',
    name: 'TestPlayer',
    is_host: false,
    is_ready: false,
    is_ai: false,
    slot_id: 0,
    joined_at: new Date().toISOString(),
    ...overrides,
  };
}

function createMockLobby(overrides: Partial<LobbyState> = {}): LobbyState {
  return {
    id: 'lobby-1',
    name: 'Test Lobby',
    host_session_id: 'host-session',
    status: 'waiting',
    players: [createMockPlayer()],
    settings: {
      max_players: 4,
      min_players: 2,
      starting_money: 1500,
      go_salary: 200,
      allow_spectators: false,
      private: false,
      rules_id: 'foundation-v1',
    },
    spectator_count: 0,
    game_id: null,
    invite_code: 'ABC123',
    created_at: new Date().toISOString(),
    ...overrides,
  };
}

describe('lobbyStore', () => {
  beforeEach(() => {
    act(() => {
      useLobbyStore.getState().reset();
    });
  });

  describe('initial state', () => {
    it('has correct initial values', () => {
      const state = useLobbyStore.getState();
      expect(state.currentLobby).toBeNull();
      expect(state.lobbyList).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('setCurrentLobby', () => {
    it('sets current lobby', () => {
      const lobby = createMockLobby();
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().currentLobby).toEqual(lobby);
    });

    it('clears error when setting lobby', () => {
      act(() => {
        useLobbyStore.getState().setError('Previous error');
        useLobbyStore.getState().setCurrentLobby(createMockLobby());
      });
      expect(useLobbyStore.getState().error).toBeNull();
    });
  });

  describe('updateLobby', () => {
    it('updates lobby properties', () => {
      act(() => {
        useLobbyStore.getState().setCurrentLobby(createMockLobby());
        useLobbyStore.getState().updateLobby({ name: 'Updated Name' });
      });
      expect(useLobbyStore.getState().currentLobby?.name).toBe('Updated Name');
    });

    it('does nothing when no current lobby', () => {
      act(() => {
        useLobbyStore.getState().updateLobby({ name: 'Updated Name' });
      });
      expect(useLobbyStore.getState().currentLobby).toBeNull();
    });
  });

  describe('setLobbyList', () => {
    it('sets lobby list', () => {
      const lobbies = [
        {
          id: '1',
          name: 'Game 1',
          host_name: 'Host',
          status: 'waiting',
          current_players: 2,
          max_players: 4,
          created_at: new Date().toISOString(),
        },
        {
          id: '2',
          name: 'Game 2',
          host_name: 'Host2',
          status: 'waiting',
          current_players: 1,
          max_players: 4,
          created_at: new Date().toISOString(),
        },
      ];
      act(() => {
        useLobbyStore.getState().setLobbyList(lobbies);
      });
      expect(useLobbyStore.getState().lobbyList).toEqual(lobbies);
    });
  });

  describe('addPlayer', () => {
    it('adds player to lobby', () => {
      const lobby = createMockLobby({ players: [] });
      const player = createMockPlayer({ session_id: 'new-player' });

      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
        useLobbyStore.getState().addPlayer(player);
      });

      expect(useLobbyStore.getState().currentLobby?.players).toHaveLength(1);
      expect(useLobbyStore.getState().currentLobby?.players[0].session_id).toBe('new-player');
    });

    it('does nothing when no current lobby', () => {
      act(() => {
        useLobbyStore.getState().addPlayer(createMockPlayer());
      });
      expect(useLobbyStore.getState().currentLobby).toBeNull();
    });
  });

  describe('removePlayer', () => {
    it('removes player from lobby', () => {
      const lobby = createMockLobby({
        players: [
          createMockPlayer({ session_id: 'player-1' }),
          createMockPlayer({ session_id: 'player-2' }),
        ],
      });

      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
        useLobbyStore.getState().removePlayer('player-1');
      });

      expect(useLobbyStore.getState().currentLobby?.players).toHaveLength(1);
      expect(useLobbyStore.getState().currentLobby?.players[0].session_id).toBe('player-2');
    });
  });

  describe('updatePlayerReady', () => {
    it('updates player ready status', () => {
      const lobby = createMockLobby({
        players: [createMockPlayer({ session_id: 'player-1', is_ready: false })],
      });

      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
        useLobbyStore.getState().updatePlayerReady('player-1', true);
      });

      expect(useLobbyStore.getState().currentLobby?.players[0].is_ready).toBe(true);
    });
  });

  describe('updateSettings', () => {
    it('updates lobby settings', () => {
      const lobby = createMockLobby();

      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
        useLobbyStore.getState().updateSettings({ starting_money: 2000 });
      });

      expect(useLobbyStore.getState().currentLobby?.settings.starting_money).toBe(2000);
    });

    it('preserves other settings', () => {
      const lobby = createMockLobby();

      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
        useLobbyStore.getState().updateSettings({ starting_money: 2000 });
      });

      expect(useLobbyStore.getState().currentLobby?.settings.max_players).toBe(4);
    });
  });

  describe('isHost', () => {
    it('returns true for host session', () => {
      const lobby = createMockLobby({ host_session_id: 'host-session' });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().isHost('host-session')).toBe(true);
    });

    it('returns false for non-host session', () => {
      const lobby = createMockLobby({ host_session_id: 'host-session' });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().isHost('other-session')).toBe(false);
    });

    it('returns false when no lobby', () => {
      expect(useLobbyStore.getState().isHost('any-session')).toBe(false);
    });
  });

  describe('canStartGame', () => {
    it('returns true when 2+ players and all ready', () => {
      const lobby = createMockLobby({
        players: [
          createMockPlayer({ session_id: 'p1', is_ready: true }),
          createMockPlayer({ session_id: 'p2', is_ready: true }),
        ],
      });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().canStartGame()).toBe(true);
    });

    it('returns true when players are ready or AI', () => {
      const lobby = createMockLobby({
        players: [
          createMockPlayer({ session_id: 'p1', is_ready: true }),
          createMockPlayer({ session_id: 'ai', is_ai: true, is_ready: false }),
        ],
      });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().canStartGame()).toBe(true);
    });

    it('returns false when less than 2 players', () => {
      const lobby = createMockLobby({
        players: [createMockPlayer({ is_ready: true })],
      });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().canStartGame()).toBe(false);
    });

    it('returns false when not all ready', () => {
      const lobby = createMockLobby({
        players: [
          createMockPlayer({ session_id: 'p1', is_ready: true }),
          createMockPlayer({ session_id: 'p2', is_ready: false }),
        ],
      });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().canStartGame()).toBe(false);
    });

    it('returns false when no lobby', () => {
      expect(useLobbyStore.getState().canStartGame()).toBe(false);
    });
  });

  describe('getMySlot', () => {
    it('returns player matching session ID', () => {
      const lobby = createMockLobby({
        players: [
          createMockPlayer({ session_id: 'p1', name: 'Player 1' }),
          createMockPlayer({ session_id: 'p2', name: 'Player 2' }),
        ],
      });
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().getMySlot('p2')?.name).toBe('Player 2');
    });

    it('returns null when no matching player', () => {
      const lobby = createMockLobby();
      act(() => {
        useLobbyStore.getState().setCurrentLobby(lobby);
      });
      expect(useLobbyStore.getState().getMySlot('unknown')).toBeNull();
    });

    it('returns null when no lobby', () => {
      expect(useLobbyStore.getState().getMySlot('any')).toBeNull();
    });
  });

  describe('reset', () => {
    it('resets to initial state', () => {
      act(() => {
        useLobbyStore.getState().setCurrentLobby(createMockLobby());
        useLobbyStore.getState().setLoading(true);
        useLobbyStore.getState().setError('error');
        useLobbyStore.getState().reset();
      });

      const state = useLobbyStore.getState();
      expect(state.currentLobby).toBeNull();
      expect(state.lobbyList).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });
});
