import { useCallback, useState, useMemo } from 'react';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useUIStore } from '@/store/uiStore';
import type { ClientActionMessage } from '@/types';

interface UseActionsOptions {
  send: (message: ClientActionMessage) => void;
}

/**
 * Game action dispatchers.
 *
 * Sends actions to server via WebSocket.
 * Provides computed states for UI enablement.
 */
export function useActions({ send }: UseActionsOptions) {
  const { setLoading, setError, gameState } = useGameStore();
  const { playerId } = useSessionStore();
  const { addToast } = useUIStore();
  const [isActionPending, setIsActionPending] = useState(false);

  // Current player state
  const currentPlayer = useMemo(
    () => gameState?.players.find((p) => p.id === playerId),
    [gameState?.players, playerId]
  );

  const isMyTurn = gameState?.decision_player === playerId;
  const isInJail = currentPlayer?.inJail ?? false;
  const doublesCount = gameState?.doublesCount ?? 0;
  const legal = (type: string) =>
    isMyTurn && (gameState?.legal_actions ?? []).some((action) => action.type === type);
  const canRollDice = !isInJail && legal('RollDice');
  const canRollInJail = isInJail && legal('RollDice');
  const canEndTurn = legal('EndTurn');
  const canPayJailFine = legal('PayJailFine');
  const canUseJailCard = legal('UseJailCard');

  const sendAction = useCallback(
    (actionType: string, data: Record<string, unknown> = {}) => {
      setIsActionPending(true);
      setLoading(true);

      try {
        send({
          type: 'action',
          data: {
            action_type: actionType,
            ...data,
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Action failed';
        setError(message);
        addToast(message, 'error');
      } finally {
        // Note: Loading state will be cleared when we receive state update
        setIsActionPending(false);
        setLoading(false);
      }
    },
    [send, setLoading, setError, addToast]
  );

  return {
    isActionPending,

    // Computed states for UI
    isMyTurn,
    isInJail,
    canRollDice,
    canRollInJail,
    canEndTurn,
    canPayJailFine,
    canUseJailCard,
    currentPlayer,
    doublesCount,

    // Dice
    rollDice: useCallback(() => sendAction('roll_dice'), [sendAction]),

    // Property
    buyProperty: useCallback(
      (propertyPosition: number) =>
        sendAction('buy_property', { property_position: propertyPosition }),
      [sendAction]
    ),

    // Pass on buying just ends the turn (no specific action needed)
    passBuy: useCallback(() => sendAction('pass_buy'), [sendAction]),

    // Building
    buildHouse: useCallback(
      (propertyPosition: number) =>
        sendAction('build_house', { property_position: propertyPosition }),
      [sendAction]
    ),

    buildHotel: useCallback(
      (propertyPosition: number) =>
        sendAction('build_hotel', { property_position: propertyPosition }),
      [sendAction]
    ),

    sellHouse: useCallback(
      (propertyPosition: number) =>
        sendAction('sell_house', { property_position: propertyPosition }),
      [sendAction]
    ),

    sellHotel: useCallback(
      (propertyPosition: number) =>
        sendAction('sell_hotel', { property_position: propertyPosition }),
      [sendAction]
    ),

    // Mortgage
    mortgageProperty: useCallback(
      (propertyPosition: number) =>
        sendAction('mortgage_property', { property_position: propertyPosition }),
      [sendAction]
    ),

    unmortgageProperty: useCallback(
      (propertyPosition: number) =>
        sendAction('unmortgage_property', { property_position: propertyPosition }),
      [sendAction]
    ),

    // Jail
    payJailFine: useCallback(() => sendAction('pay_jail_fine'), [sendAction]),

    useJailCard: useCallback(() => sendAction('use_jail_card'), [sendAction]),

    // Turn
    endTurn: useCallback(() => sendAction('end_turn'), [sendAction]),

    // Trade (simplified)
    proposeTradeOffer: useCallback(
      (targetPlayerId: number, offer: Record<string, unknown>) =>
        sendAction('propose_trade', { target_player: targetPlayerId, ...offer }),
      [sendAction]
    ),

    acceptTrade: useCallback(
      (tradeId: string) => sendAction('accept_trade', { trade_id: tradeId }),
      [sendAction]
    ),

    rejectTrade: useCallback(
      (tradeId: string) => sendAction('reject_trade', { trade_id: tradeId }),
      [sendAction]
    ),
  };
}
