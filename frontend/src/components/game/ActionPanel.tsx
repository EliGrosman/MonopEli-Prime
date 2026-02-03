import { useGameStore } from '@/store/gameStore';
import { useActions } from '@/hooks/useActions';
import { DiceRoll } from './DiceRoll';
import { BuildingControls } from './BuildingControls';
import { BOARD_SPACES } from '@/utils/board';
import type { ClientActionMessage } from '@/types';

interface ActionPanelProps {
  send: (message: ClientActionMessage) => void;
}

/**
 * Action panel for the current player's turn.
 *
 * Shows available actions based on game state:
 * - Roll Dice (if hasn't rolled)
 * - Buy Property (if on unowned property)
 * - Build House/Hotel (if has monopoly)
 * - Mortgage/Unmortgage
 * - End Turn
 * - Jail actions (if in jail)
 */
export function ActionPanel({ send }: ActionPanelProps) {
  const { gameState } = useGameStore();
  const {
    isActionPending,
    isMyTurn,
    isInJail,
    canRollDice,
    canEndTurn,
    canPayJailFine,
    canUseJailCard,
    currentPlayer,
    rollDice,
    buyProperty,
    passBuy,
    payJailFine,
    useJailCard,
    endTurn,
  } = useActions({ send });

  if (!gameState) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-lg font-semibold mb-4">Actions</h2>
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }

  // Check if player is bankrupt
  if (currentPlayer?.bankrupt) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-lg font-semibold mb-4">Actions</h2>
        <div className="text-center py-4">
          <p className="text-red-500 font-medium">You are bankrupt</p>
          <p className="text-gray-500 text-sm mt-2">
            Watch the rest of the game unfold!
          </p>
        </div>
      </div>
    );
  }

  // Not my turn
  if (!isMyTurn) {
    const currentTurnPlayer = gameState.players[gameState.currentPlayer];
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-lg font-semibold mb-4">Actions</h2>
        <div className="text-center py-4">
          <p className="text-gray-600">
            Waiting for{' '}
            <span className="font-medium" style={{ color: currentTurnPlayer?.color }}>
              {currentTurnPlayer?.name || 'opponent'}
            </span>
            ...
          </p>
          {currentTurnPlayer?.isAi && (
            <p className="text-gray-400 text-sm mt-2">AI is thinking...</p>
          )}
        </div>
      </div>
    );
  }

  // Get current position info
  const currentPosition = currentPlayer?.position ?? 0;
  const currentSpace = BOARD_SPACES[currentPosition];
  const propertyAtPosition = gameState.properties[currentPosition];
  const canBuyProperty =
    propertyAtPosition &&
    propertyAtPosition.owner === null &&
    currentSpace?.price &&
    (currentPlayer?.money ?? 0) >= currentSpace.price &&
    gameState.gamePhase === 'post_roll';

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Your Turn</h2>
        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-sm">
          {gameState.gamePhase.replace('_', ' ')}
        </span>
      </div>

      {/* Dice section */}
      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <DiceRoll
          roll={gameState.lastRoll}
          onRoll={rollDice}
          canRoll={canRollDice}
          isRolling={isActionPending}
        />
      </div>

      {/* Jail section */}
      {isInJail && (
        <div className="mb-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
          <h3 className="font-medium text-orange-800 mb-2">
            In Jail (Turn {(currentPlayer?.jailTurns ?? 0) + 1}/3)
          </h3>
          <div className="space-y-2">
            {canUseJailCard && (
              <button
                onClick={useJailCard}
                disabled={isActionPending}
                className="w-full px-3 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50 transition-colors"
              >
                Use Get Out of Jail Free Card ({currentPlayer?.jailCards})
              </button>
            )}
            {canPayJailFine && (
              <button
                onClick={payJailFine}
                disabled={isActionPending}
                className="w-full px-3 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50 transition-colors"
              >
                Pay $50 Fine
              </button>
            )}
            <button
              onClick={rollDice}
              disabled={isActionPending || !canRollDice}
              className="w-full px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 transition-colors"
            >
              Try for Doubles
            </button>
          </div>
        </div>
      )}

      {/* Buy property section */}
      {canBuyProperty && currentSpace && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <h3 className="font-medium text-green-800 mb-2">Buy Property?</h3>
          <p className="text-sm text-gray-600 mb-3">
            <span className="font-medium">{currentSpace.name}</span>
            {' - '}${currentSpace.price?.toLocaleString()}
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => buyProperty(currentPosition)}
              disabled={isActionPending}
              className="flex-1 px-3 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 transition-colors"
            >
              Buy (${currentSpace.price?.toLocaleString()})
            </button>
            <button
              onClick={passBuy}
              disabled={isActionPending}
              className="flex-1 px-3 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 transition-colors"
            >
              Pass
            </button>
          </div>
        </div>
      )}

      {/* Building controls */}
      <div className="mb-4">
        <BuildingControls send={send} />
      </div>

      {/* End turn */}
      {canEndTurn && (
        <button
          onClick={endTurn}
          disabled={isActionPending}
          className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg font-medium hover:bg-gray-800 disabled:opacity-50 transition-colors"
        >
          End Turn
        </button>
      )}

      {/* Current position info */}
      <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-500">
        <p>
          Position: <span className="font-medium text-gray-700">{currentSpace?.name}</span>
        </p>
        <p>
          Cash: <span className="font-medium text-green-600">${currentPlayer?.money?.toLocaleString()}</span>
        </p>
      </div>
    </div>
  );
}
