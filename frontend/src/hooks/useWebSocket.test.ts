import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useActions } from './useActions';
import { useWebSocket } from './useWebSocket';

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  static instances: MockWebSocket[] = [];
  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  send = vi.fn<(data: string) => void>();

  constructor() {
    MockWebSocket.instances.push(this);
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  message(type: string, data: unknown) {
    this.onmessage?.({ data: JSON.stringify({ type, data }) });
  }

  close(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }
}

const state = {
  game_id: 'game',
  rules_id: 'foundation-trade-v1',
  revision: 7,
  current_player: 0,
  decision_player: 0,
  game_phase: 'debt_resolution',
  players: [],
  properties: {},
};

function setup() {
  const hook = renderHook(() => {
    const socket = useWebSocket({ gameId: 'game', autoConnect: false });
    const actions = useActions({ send: socket.send });
    return { socket, actions };
  });
  act(() => hook.result.current.socket.connect());
  const ws = MockWebSocket.instances.at(-1)!;
  act(() => {
    ws.open();
    ws.message('state_update', state);
  });
  return { ...hook, ws };
}

function actionData(ws: MockWebSocket, index = 0) {
  return JSON.parse(ws.send.mock.calls[index][0]).data;
}

describe('useWebSocket action lifecycle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('WebSocket', MockWebSocket);
    MockWebSocket.instances = [];
    useGameStore.getState().reset();
    useSessionStore.getState().setSession('session', 'Player');
    useSessionStore.getState().setCurrentGame('game', 0);
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it.each(['mortgage_property', 'sell_house', 'sell_hotel'])(
    'adds the contract envelope to directly sent debt action %s',
    (actionType) => {
      const { result, ws } = setup();
      act(() =>
        result.current.socket.send({
          type: 'action',
          data: { action_type: actionType, property_position: 39 },
        })
      );
      expect(ws.send).toHaveBeenCalledOnce();
      expect(actionData(ws)).toEqual({
        action_type: actionType,
        property_position: 39,
        contract_version: 'decision-contract-v1',
        expected_revision: 7,
        request_id: expect.any(String),
      });
      expect(useGameStore.getState().pendingRequestId).toBe(actionData(ws).request_id);
    }
  );

  it('sends hook actions once and suppresses rapid duplicate submissions', () => {
    const { result, ws } = setup();
    act(() => {
      result.current.actions.mortgageProperty(39);
      result.current.actions.mortgageProperty(39);
      result.current.socket.send({ type: 'action', data: { action_type: 'end_turn' } });
    });
    expect(ws.send).toHaveBeenCalledOnce();
    expect(actionData(ws).action_type).toBe('mortgage_property');
    expect(actionData(ws).expected_revision).toBe(7);
    expect(result.current.actions.isActionPending).toBe(true);
  });

  it('preserves ordinary action messages for the non-trading ruleset', () => {
    const { result, ws } = setup();
    act(() => {
      ws.message('state_update', { ...state, rules_id: 'foundation-v1' });
      result.current.actions.rollDice();
    });
    expect(ws.send).toHaveBeenCalledOnce();
    expect(actionData(ws)).toEqual({ action_type: 'roll_dice' });
    expect(useGameStore.getState().pendingRequestId).toBeNull();
  });

  it('requires the matching acknowledgement and resulting state before enabling another action', () => {
    const { result, ws } = setup();
    act(() => result.current.actions.mortgageProperty(39));
    const requestId = actionData(ws).request_id;
    act(() => {
      ws.message('state_update', { ...state, revision: 6 });
      ws.message('state_update', state);
      ws.message('action_result', { success: true, request_id: 'unrelated', revision: 8 });
    });
    expect(useGameStore.getState().pendingRequestId).toBe(requestId);
    act(() => ws.message('action_result', { success: true, request_id: requestId, revision: 8 }));
    expect(useGameStore.getState().pendingRequestId).toBe(requestId);
    act(() => {
      ws.message('state_update', state);
      result.current.actions.endTurn();
    });
    expect(ws.send).toHaveBeenCalledOnce();
    act(() => ws.message('state_update', { ...state, revision: 8 }));
    expect(useGameStore.getState().pendingRequestId).toBeNull();
    act(() => result.current.actions.endTurn());
    expect(actionData(ws, 1).expected_revision).toBe(8);

    const nextRequest = actionData(ws, 1).request_id;
    act(() => ws.message('action_result', { success: true, request_id: requestId, revision: 8 }));
    expect(useGameStore.getState().pendingRequestId).toBe(nextRequest);
  });

  it('does not clear pending for a state broadcast that arrives before its acknowledgement', () => {
    const { result, ws } = setup();
    act(() => result.current.actions.mortgageProperty(39));
    const requestId = actionData(ws).request_id;
    act(() => ws.message('state_update', { ...state, revision: 8 }));
    expect(useGameStore.getState().pendingRequestId).toBe(requestId);
    act(() => ws.message('action_result', { success: true, request_id: requestId, revision: 8 }));
    expect(useGameStore.getState().pendingRequestId).toBeNull();
  });

  it('clears a matching failed request immediately, but ignores an unrelated rejection', () => {
    const { result, ws } = setup();
    act(() => result.current.actions.mortgageProperty(39));
    const requestId = actionData(ws).request_id;
    act(() =>
      ws.message('action_result', { success: false, request_id: 'other', message: 'Old error' })
    );
    expect(useGameStore.getState().pendingRequestId).toBe(requestId);
    expect(useGameStore.getState().error).toBeNull();
    act(() =>
      ws.message('action_result', {
        success: false,
        request_id: requestId,
        revision: 8,
        message: 'Stale decision revision',
      })
    );
    expect(useGameStore.getState().pendingRequestId).toBeNull();
    expect(useGameStore.getState().error).toBe('Stale decision revision');
    act(() => {
      ws.message('state_update', { ...state, revision: 8 });
      result.current.actions.mortgageProperty(39);
    });
    expect(actionData(ws, 1).expected_revision).toBe(8);
  });

  it('recovers from a socket send exception without leaving actions pending', () => {
    const { result, ws } = setup();
    ws.send.mockImplementationOnce(() => {
      throw new Error('Socket failed');
    });
    act(() => result.current.actions.mortgageProperty(39));
    expect(useGameStore.getState().pendingRequestId).toBeNull();
    expect(useGameStore.getState().error).toBe('Socket failed');
    act(() => result.current.actions.mortgageProperty(39));
    expect(ws.send).toHaveBeenCalledTimes(2);
    expect(useGameStore.getState().pendingRequestId).toBe(actionData(ws, 1).request_id);
  });

  it('discards disconnected actions and waits for fresh state on reconnect without replaying a sent action', () => {
    const { result, ws } = setup();
    act(() => result.current.actions.mortgageProperty(39));
    act(() => ws.close(1006));
    expect(useGameStore.getState().pendingRequestId).toBeNull();
    act(() => {
      result.current.actions.mortgageProperty(39);
      vi.advanceTimersByTime(3000);
    });
    const reconnected = MockWebSocket.instances.at(-1)!;
    expect(reconnected).not.toBe(ws);
    act(() => {
      reconnected.open();
      result.current.actions.mortgageProperty(39);
    });
    expect(useGameStore.getState().isConnected).toBe(false);
    expect(reconnected.send).not.toHaveBeenCalled();
    act(() => {
      reconnected.message('state_update', { ...state, revision: 8 });
      result.current.actions.endTurn();
    });
    expect(reconnected.send).toHaveBeenCalledOnce();
    expect(actionData(reconnected).expected_revision).toBe(8);

    // A replaced socket's delayed callbacks cannot clear a new request or state.
    const requestId = actionData(reconnected).request_id;
    act(() => {
      ws.message('state_update', { ...state, revision: 100 });
      ws.close(1006);
    });
    expect(useGameStore.getState().pendingRequestId).toBe(requestId);
    expect(useGameStore.getState().gameState?.revision).toBe(8);
    expect(useGameStore.getState().isConnected).toBe(true);
  });

  it('cancels scheduled reconnects on unmount', () => {
    const { ws, unmount } = setup();
    act(() => ws.close(1006));
    unmount();
    act(() => vi.advanceTimersByTime(3000));
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it('applies agent status without changing revision or pending human requests', () => {
    const { result, ws } = setup();
    act(() => result.current.actions.mortgageProperty(39));
    const pending = useGameStore.getState().pendingRequestId;
    act(() =>
      ws.message('agent_update', {
        player_id: 1,
        sequence: 2,
        basis_revision: 7,
        status: 'thinking',
        short_term_objective: 'Complete orange',
        long_term_objective: 'Develop orange rent income',
        cash_reserve_target: 200,
        latest_summary: 'Considering the current decision',
        fallback_reason: null,
      })
    );
    expect(useGameStore.getState().gameState?.revision).toBe(7);
    expect(useGameStore.getState().pendingRequestId).toBe(pending);
    expect(useGameStore.getState().gameState?.agent_inspections?.[1].status).toBe('thinking');
  });
});
