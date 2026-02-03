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

  // Computed states for UI
  const isMyTurn = gameState?.currentPlayer === playerId;
  const isInJail = currentPlayer?.inJail ?? false;
  const canRollDice = isMyTurn && !gameState?.lastRoll && gameState?.gamePhase === 'pre_roll' && !isInJail;
  const canEndTurn = isMyTurn && (gameState?.gamePhase === 'post_roll' || gameState?.lastRoll !== null);
  const canPayJailFine = isMyTurn && isInJail && (currentPlayer?.money ?? 0) >= 50;
  const canUseJailCard = isMyTurn && isInJail && (currentPlayer?.jailCards ?? 0) > 0;

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
    canEndTurn,
    canPayJailFine,
    canUseJailCard,
    currentPlayer,

    // Dice
    rollDice: useCallback(() => sendAction('roll_dice'), [sendAction]),

    // Property
    buyProperty: useCallback(
      (propertyId: number) => sendAction('buy_property', { property_id: propertyId }),
      [sendAction]
    ),

    passBuy: useCallback(() => sendAction('pass_buy'), [sendAction]),

    // Building
    buildHouse: useCallback(
      (propertyId: number) => sendAction('build_house', { property_id: propertyId }),
      [sendAction]
    ),

    buildHotel: useCallback(
      (propertyId: number) => sendAction('build_hotel', { property_id: propertyId }),
      [sendAction]
    ),

    sellHouse: useCallback(
      (propertyId: number) => sendAction('sell_house', { property_id: propertyId }),
      [sendAction]
    ),

    sellHotel: useCallback(
      (propertyId: number) => sendAction('sell_hotel', { property_id: propertyId }),
      [sendAction]
    ),

    // Mortgage
    mortgageProperty: useCallback(
      (propertyId: number) => sendAction('mortgage_property', { property_id: propertyId }),
      [sendAction]
    ),

    unmortgageProperty: useCallback(
      (propertyId: number) => sendAction('unmortgage_property', { property_id: propertyId }),
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
