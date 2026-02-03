import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useActions } from './useActions';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import type { GameState } from '@/types';

// Mock game state
const mockGameState: GameState = {
  gameId: 'test-game',
  players: [
    {
      id: 0,
      name: 'Alice',
      money: 1500,
      position: 5,
      inJail: false,
      jailTurns: 0,
      jailCards: 1,
      bankrupt: false,
      isAi: false,
      color: '#E53935',
    },
    {
      id: 1,
      name: 'Bob',
      money: 1200,
      position: 10,
      inJail: true,
      jailTurns: 1,
      jailCards: 0,
      bankrupt: false,
      isAi: false,
      color: '#1E88E5',
    },
  ],
  properties: {},
  currentPlayer: 0,
  turnNumber: 1,
  lastRoll: null,
  housesRemaining: 32,
  hotelsRemaining: 12,
  gamePhase: 'pre_roll',
  gameOver: false,
  winner: null,
};

describe('useActions', () => {
  const mockSend = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset stores
    act(() => {
      useGameStore.getState().reset();
      useSessionStore.getState().clearSession();
    });
  });

  it('returns action functions', () => {
    const { result } = renderHook(() => useActions({ send: mockSend }));

    expect(result.current.rollDice).toBeDefined();
    expect(result.current.buyProperty).toBeDefined();
    expect(result.current.endTurn).toBeDefined();
    expect(result.current.buildHouse).toBeDefined();
    expect(result.current.mortgageProperty).toBeDefined();
  });

  it('isMyTurn is false when not player turn', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 1); // Player 1, but current is 0
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.isMyTurn).toBe(false);
  });

  it('isMyTurn is true when it is player turn', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0); // Player 0 is current
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.isMyTurn).toBe(true);
  });

  it('canRollDice is true in pre_roll phase', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.canRollDice).toBe(true);
  });

  it('canRollDice is false after rolling', () => {
    const stateWithRoll = {
      ...mockGameState,
      lastRoll: { die1: 3, die2: 4, isDoubles: false },
    };

    act(() => {
      useGameStore.getState().updateGameState(stateWithRoll);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.canRollDice).toBe(false);
  });

  it('canEndTurn is true after rolling', () => {
    const stateWithRoll = {
      ...mockGameState,
      lastRoll: { die1: 3, die2: 4, isDoubles: false },
      gamePhase: 'post_roll' as const,
    };

    act(() => {
      useGameStore.getState().updateGameState(stateWithRoll);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.canEndTurn).toBe(true);
  });

  it('isInJail is true when player is in jail', () => {
    act(() => {
      useGameStore.getState().updateGameState({
        ...mockGameState,
        currentPlayer: 1,
      });
      useSessionStore.getState().setCurrentGame('test-game', 1);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.isInJail).toBe(true);
  });

  it('canPayJailFine is true when in jail with enough money', () => {
    act(() => {
      useGameStore.getState().updateGameState({
        ...mockGameState,
        currentPlayer: 1,
      });
      useSessionStore.getState().setCurrentGame('test-game', 1);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.canPayJailFine).toBe(true);
  });

  it('canUseJailCard is true when player has jail cards', () => {
    // Player 0 has jail cards
    act(() => {
      useGameStore.getState().updateGameState({
        ...mockGameState,
        players: [
          { ...mockGameState.players[0], inJail: true },
          mockGameState.players[1],
        ],
      });
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));
    expect(result.current.canUseJailCard).toBe(true);
  });

  it('rollDice sends correct action', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));

    act(() => {
      result.current.rollDice();
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: 'action',
      data: { action_type: 'roll_dice' },
    });
  });

  it('buyProperty sends correct action with property id', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));

    act(() => {
      result.current.buyProperty(5);
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: 'action',
      data: { action_type: 'buy_property', property_id: 5 },
    });
  });

  it('buildHouse sends correct action', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));

    act(() => {
      result.current.buildHouse(1);
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: 'action',
      data: { action_type: 'build_house', property_id: 1 },
    });
  });

  it('mortgageProperty sends correct action', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));

    act(() => {
      result.current.mortgageProperty(3);
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: 'action',
      data: { action_type: 'mortgage_property', property_id: 3 },
    });
  });

  it('endTurn sends correct action', () => {
    act(() => {
      useGameStore.getState().updateGameState(mockGameState);
      useSessionStore.getState().setCurrentGame('test-game', 0);
    });

    const { result } = renderHook(() => useActions({ send: mockSend }));

    act(() => {
      result.current.endTurn();
    });

    expect(mockSend).toHaveBeenCalledWith({
      type: 'action',
      data: { action_type: 'end_turn' },
    });
  });
});
