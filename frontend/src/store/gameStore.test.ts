import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import { useGameStore } from './gameStore';

// Reset store between tests
beforeEach(() => {
  act(() => {
    useGameStore.getState().reset();
  });
});

// Create a mock raw backend state (snake_case) for updateGameState
function createMockRawState() {
  return {
    current_player: 0,
    doubles_count: 0,
    last_roll: null as [number, number] | null,
    players: [
      {
        id: 0,
        name: 'Alice',
        money: 1500,
        position: 0,
        in_jail: false,
        jail_turns: 0,
        jail_cards: 0,
        bankrupt: false,
        is_ai: false,
        color: '#E53935',
      },
      {
        id: 1,
        name: 'Bob',
        money: 1200,
        position: 5,
        in_jail: false,
        jail_turns: 0,
        jail_cards: 0,
        bankrupt: false,
        is_ai: false,
        color: '#1E88E5',
      },
    ],
    properties: {
      1: { position: 1, owner: 0, houses: 2, mortgaged: false },
      3: { position: 3, owner: 0, houses: 0, mortgaged: false },
      5: { position: 5, owner: 1, houses: 0, mortgaged: true },
    },
    houses_remaining: 30,
    hotels_remaining: 12,
    game_over: false,
    winner: null,
  };
}

describe('gameStore', () => {
  describe('initial state', () => {
    it('has correct initial values', () => {
      const state = useGameStore.getState();
      expect(state.gameId).toBeNull();
      expect(state.gameState).toBeNull();
      expect(state.isConnected).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('actions', () => {
    it('setGameId updates gameId', () => {
      act(() => {
        useGameStore.getState().setGameId('game-123');
      });
      expect(useGameStore.getState().gameId).toBe('game-123');
    });

    it('setGameId can clear gameId', () => {
      act(() => {
        useGameStore.getState().setGameId('game-123');
        useGameStore.getState().setGameId(null);
      });
      expect(useGameStore.getState().gameId).toBeNull();
    });

    it('updateGameState sets game state and clears error', () => {
      const rawState = createMockRawState();

      act(() => {
        useGameStore.getState().setError('Some error');
        useGameStore.getState().updateGameState(rawState);
      });

      const gameState = useGameStore.getState().gameState;
      expect(gameState).not.toBeNull();
      expect(gameState!.currentPlayer).toBe(0);
      expect(gameState!.players).toHaveLength(2);
      expect(gameState!.players[0].name).toBe('Alice');
      expect(gameState!.players[1].name).toBe('Bob');
      expect(gameState!.housesRemaining).toBe(30);
      expect(gameState!.hotelsRemaining).toBe(12);
      expect(gameState!.gameOver).toBe(false);
      expect(gameState!.gamePhase).toBe('pre_roll');
      expect(gameState!.lastRoll).toBeNull();
      expect(useGameStore.getState().error).toBeNull();
    });

    it('setConnected updates connection status', () => {
      act(() => {
        useGameStore.getState().setConnected(true);
      });
      expect(useGameStore.getState().isConnected).toBe(true);

      act(() => {
        useGameStore.getState().setConnected(false);
      });
      expect(useGameStore.getState().isConnected).toBe(false);
    });

    it('setLoading updates loading state', () => {
      act(() => {
        useGameStore.getState().setLoading(true);
      });
      expect(useGameStore.getState().isLoading).toBe(true);
    });

    it('setError updates error', () => {
      act(() => {
        useGameStore.getState().setError('Connection failed');
      });
      expect(useGameStore.getState().error).toBe('Connection failed');
    });

    it('reset restores initial state', () => {
      act(() => {
        useGameStore.getState().setGameId('game-123');
        useGameStore.getState().updateGameState(createMockRawState());
        useGameStore.getState().setConnected(true);
        useGameStore.getState().setLoading(true);
        useGameStore.getState().setError('error');
        useGameStore.getState().reset();
      });

      const state = useGameStore.getState();
      expect(state.gameId).toBeNull();
      expect(state.gameState).toBeNull();
      expect(state.isConnected).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('selectors', () => {
    beforeEach(() => {
      act(() => {
        useGameStore.getState().updateGameState(createMockRawState());
      });
    });

    it('getCurrentPlayer returns the current player', () => {
      const currentPlayer = useGameStore.getState().getCurrentPlayer();
      expect(currentPlayer?.id).toBe(0);
      expect(currentPlayer?.name).toBe('Alice');
    });

    it('getCurrentPlayer returns null with no game state', () => {
      act(() => {
        useGameStore.getState().reset();
      });
      expect(useGameStore.getState().getCurrentPlayer()).toBeNull();
    });

    it('getPlayer returns player by id', () => {
      const player = useGameStore.getState().getPlayer(1);
      expect(player?.id).toBe(1);
      expect(player?.name).toBe('Bob');
    });

    it('getPlayer returns null for invalid id', () => {
      expect(useGameStore.getState().getPlayer(99)).toBeNull();
    });

    it('getPropertyOwner returns owner of property', () => {
      const owner = useGameStore.getState().getPropertyOwner(1);
      expect(owner?.id).toBe(0);
      expect(owner?.name).toBe('Alice');
    });

    it('getPropertyOwner returns null for unowned property', () => {
      expect(useGameStore.getState().getPropertyOwner(10)).toBeNull();
    });

    it('getProperty returns property state', () => {
      const prop = useGameStore.getState().getProperty(1);
      expect(prop?.position).toBe(1);
      expect(prop?.owner).toBe(0);
      expect(prop?.houses).toBe(2);
    });

    it('getProperty returns null for non-existent property', () => {
      expect(useGameStore.getState().getProperty(99)).toBeNull();
    });

    it('canAfford checks if player has enough money', () => {
      expect(useGameStore.getState().canAfford(0, 1000)).toBe(true);
      expect(useGameStore.getState().canAfford(0, 1500)).toBe(true);
      expect(useGameStore.getState().canAfford(0, 2000)).toBe(false);
      expect(useGameStore.getState().canAfford(1, 1200)).toBe(true);
      expect(useGameStore.getState().canAfford(1, 1300)).toBe(false);
    });

    it('canAfford returns false for invalid player', () => {
      expect(useGameStore.getState().canAfford(99, 100)).toBe(false);
    });

    it('getPlayerProperties returns all properties owned by player', () => {
      const props = useGameStore.getState().getPlayerProperties(0);
      expect(props).toHaveLength(2);
      expect(props.map((p) => p.position).sort()).toEqual([1, 3]);
    });

    it('getPlayerProperties returns empty array for player with no properties', () => {
      // Player 1 owns position 5
      const props = useGameStore.getState().getPlayerProperties(99);
      expect(props).toHaveLength(0);
    });

    it('hasMonopoly checks if player owns all properties in color group', () => {
      // Alice owns positions 1 and 3 (brown properties)
      expect(useGameStore.getState().hasMonopoly(0, 'brown')).toBe(true);
      // Alice doesn't own all light blue properties
      expect(useGameStore.getState().hasMonopoly(0, 'lightblue')).toBe(false);
    });
  });
});
