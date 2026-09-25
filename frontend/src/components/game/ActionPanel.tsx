import type { ReactNode } from 'react';
import { useGameStore } from '@/store/gameStore';
import { useActions } from '@/hooks/useActions';
import { DiceRoll } from './DiceRoll';
import { BuildingControls } from './BuildingControls';
import { PROPERTY_INFO } from '@/utils/board';
import type { ClientActionMessage } from '@/types';

interface ActionPanelProps {
  send: (message: ClientActionMessage) => void;
  variant?: 'default' | 'board';
  headerAction?: ReactNode;
  onManageProperties?: () => void;
}

export function ActionPanel({
  send,
  variant = 'default',
  headerAction,
  onManageProperties,
}: ActionPanelProps) {
  const gameState = useGameStore((state) => state.gameState);
  const actions = useActions({ send });
  const { currentPlayer, isMyTurn, isActionPending, isInJail } = actions;
  const isBoard = variant === 'board';
  const canPass = gameState?.legal_actions?.some((a) => a.type === 'PassBuy') && isMyTurn;
  const canBuy = gameState?.legal_actions?.some((a) => a.type === 'BuyProperty') && isMyTurn;
  const position = currentPlayer?.position ?? 0;
  const property = PROPERTY_INFO[position];
  const assetActions = [
    'BuildHouse',
    'BuildHotel',
    'SellHouse',
    'SellHotel',
    'SellBuildingGroup',
    'MortgageProperty',
    'UnmortgageProperty',
  ];
  const canManage =
    isMyTurn && gameState?.legal_actions?.some((a) => assetActions.includes(a.type));

  return (
    <section
      className={isBoard ? 'center-actions' : 'rounded-lg bg-white p-4 text-gray-900 shadow-md'}
      aria-label="Game actions"
      aria-busy={!gameState || isActionPending}
    >
      {!isBoard && <h2 className="mb-3 font-semibold">{isMyTurn ? 'Your Turn' : 'Actions'}</h2>}
      {headerAction}
      {!gameState ? (
        <p>Loading...</p>
      ) : gameState.gameOver ? (
        <p className="font-semibold">
          {gameState.players.find((p) => p.id === gameState.winner)?.name ?? 'A player'} won!
        </p>
      ) : currentPlayer?.bankrupt ? (
        <p className="text-sm text-gray-700">
          You are bankrupt. Follow the remaining players below.
        </p>
      ) : !isMyTurn ? (
        <p className="waiting-hint text-sm text-gray-600">
          Follow the action below while the other player decides.
        </p>
      ) : (
        <>
          {canPass && property ? (
            <div className="primary-decision" role="group" aria-label="Property purchase options">
              <p className="mb-1 text-sm font-medium">
                Buy {property.name} for ${property.price.toLocaleString()}?
              </p>
              <div className="flex gap-2">
                <button
                  className="game-button game-button-primary flex-1"
                  disabled={!canBuy || isActionPending}
                  onClick={() => actions.buyProperty(position)}
                  aria-label={`Buy ${property.name} for ${property.price} dollars`}
                >
                  Buy (${property.price})
                </button>
                <button
                  className="game-button flex-1"
                  disabled={isActionPending}
                  onClick={actions.passBuy}
                  aria-label={`Pass on buying ${property.name}`}
                >
                  Pass
                </button>
              </div>
            </div>
          ) : isInJail && gameState.gamePhase === 'jail_decision' ? (
            <div className="primary-decision jail-decisions" role="group" aria-label="Jail options">
              {actions.canUseJailCard && (
                <button
                  className="game-button"
                  disabled={isActionPending}
                  onClick={actions.useJailCard}
                  aria-label={`Use Get Out of Jail Free Card. You have ${currentPlayer?.jailCards} cards.`}
                >
                  Use jail card
                </button>
              )}
              {actions.canPayJailFine && (
                <button
                  className="game-button"
                  disabled={isActionPending}
                  onClick={actions.payJailFine}
                  aria-label="Pay 50 dollar fine to get out of jail"
                >
                  Pay $50 fine
                </button>
              )}
              <button
                className="game-button game-button-primary"
                disabled={isActionPending || !actions.canRollInJail}
                onClick={actions.rollDice}
                aria-label="Roll dice to try for doubles and escape jail"
              >
                Try for doubles
              </button>
            </div>
          ) : actions.canRollDice ? (
            <div className="primary-decision center-dice" role="group" aria-label="Dice controls">
              <DiceRoll
                roll={gameState.lastRoll}
                onRoll={actions.rollDice}
                canRoll
                isRolling={isActionPending}
                compact={isBoard}
              />
            </div>
          ) : (
            gameState.lastRoll && (
              <p className="last-roll-summary text-sm text-gray-600">
                Rolled {gameState.lastRoll.die1} + {gameState.lastRoll.die2}
                {gameState.lastRoll.isDoubles ? ' · Doubles' : ''}
              </p>
            )
          )}
          <div className="center-action-buttons">
            {canManage && onManageProperties && (
              <button className="game-button" onClick={onManageProperties}>
                Manage properties
              </button>
            )}
            {actions.canEndTurn && (
              <button
                className="game-button game-button-primary"
                disabled={isActionPending}
                onClick={actions.endTurn}
                aria-label="End your turn"
              >
                End Turn
              </button>
            )}
          </div>
          {!isBoard && !onManageProperties && <BuildingControls send={send} />}
        </>
      )}
    </section>
  );
}
