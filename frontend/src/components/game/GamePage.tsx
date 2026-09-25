import { useParams } from 'react-router-dom';
import { useEffect, useState, useMemo } from 'react';
import { Board } from '@/components/board/Board';
import { ConnectionStatus, Loading, Modal } from '@/components/common';
import { PropertyModal, PropertyList } from '@/components/property';
import { PlayerPanel } from './PlayerPanel';
import { PlayerCard } from './PlayerCard';
import { ActionPanel } from './ActionPanel';
import { BuildingControls } from './BuildingControls';
import { TradePanel } from './TradePanel';
import { EventLog, LatestActivity } from './EventLog';
import { TurnStatus } from './TurnStatus';
import { useGameState } from '@/hooks/useGameState';
import { useSessionStore } from '@/store/sessionStore';
import { useGameStore } from '@/store/gameStore';
import { useUIStore } from '@/store';
import { PROPERTY_INFO } from '@/utils/board';

type GameDialog =
  | { type: 'property'; position: number }
  | { type: 'player'; id: number }
  | { type: 'assets' | 'history' }
  | { type: 'trade'; key: string };

export function GamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  return <GameSession key={gameId} gameId={gameId} />;
}

function GameSession({ gameId }: { gameId?: string }) {
  const { setCurrentGame, playerId } = useSessionStore();
  const { addToast } = useUIStore();
  const { gameState, isConnected, pendingRequestId } = useGameStore();
  const { isLoading, error, connectionState, reconnect, isMyTurn, send } = useGameState();
  const [dialog, setDialog] = useState<GameDialog | null>(null);
  const [dismissedOffer, setDismissedOffer] = useState<string | null>(null);
  const contract = gameState?.decision_contract;
  const offer = contract?.pending_offer;
  const isTradingRuleset = gameState?.rules_id === 'foundation-trade-v1';
  const canPropose = contract?.trade.can_propose && gameState?.decision_player === playerId;
  const isIncoming = Boolean(
    offer && offer.to_player === playerId && gameState?.decision_player === playerId
  );
  const canTrade = Boolean(
    isTradingRuleset &&
    isConnected &&
    !gameState?.gameOver &&
    (canPropose || (offer && (offer.from_player === playerId || offer.to_player === playerId)))
  );
  const tradeKey = offer ? `offer:${offer.trade_id}` : `draft:${gameState?.revision}`;
  // Deriving priority avoids flashing a property dialog before an incoming offer.
  // A dismissed offer stays dismissed across repeated state broadcasts.
  const activeDialog: GameDialog | null =
    canTrade && isIncoming && dismissedOffer !== tradeKey
      ? { type: 'trade', key: tradeKey }
      : dialog?.type === 'trade' && (!canTrade || dialog.key !== tradeKey)
        ? null
        : dialog;
  const closeDialog = () => {
    if (isIncoming) setDismissedOffer(tradeKey);
    setDialog(null);
  };
  const selectedPlayer =
    activeDialog?.type === 'player'
      ? gameState?.players.find((p) => p.id === activeDialog.id)
      : undefined;
  const players = gameState?.players;
  const playerColorStyle = useMemo(
    () =>
      Object.fromEntries(
        (players ?? []).map((player) => [`--player-${player.id}-color`, player.color])
      ),
    [players]
  );

  useEffect(() => {
    if (gameId) setCurrentGame(gameId, playerId);
  }, [gameId, playerId, setCurrentGame]);
  useEffect(() => {
    if (error) addToast(error, 'error');
  }, [error, addToast]);

  if (isLoading || connectionState === 'connecting' || connectionState === 'reconnecting') {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loading
          message={connectionState === 'reconnecting' ? 'Reconnecting...' : 'Connecting to game...'}
        />
      </div>
    );
  }

  const openProperty = (position: number) => {
    if (gameState?.properties[position] !== undefined) setDialog({ type: 'property', position });
  };
  const openHistory = () => setDialog({ type: 'history' });
  const debt = gameState?.debt;
  const debtor = gameState?.players.find((p) => p.id === debt?.debtor);
  const debtActions: Record<string, string> = {
    SellHouse: 'sell_house',
    SellHotel: 'sell_hotel',
    SellBuildingGroup: 'sell_building_group',
    MortgageProperty: 'mortgage_property',
    DeclareBankruptcy: 'declare_bankruptcy',
  };

  const center = (
    <div className="center-layout">
      <div className="center-status-row">
        <TurnStatus />
        {isTradingRuleset && (
          <button
            className="game-button trade-button"
            disabled={!canTrade}
            onClick={() => setDialog({ type: 'trade', key: tradeKey })}
            aria-label={isIncoming ? 'Open incoming trade offer' : 'Open trading'}
          >
            Trade
            {offer && (
              <span className="trade-dot" aria-label="Trade offer pending">
                1
              </span>
            )}
          </button>
        )}
      </div>
      <div className="center-main">
        {debt ? (
          <section className="debt-controls" aria-label="Debt resolution">
            <h3 className="font-semibold">Debt resolution · ${debt.amount.toLocaleString()}</h3>
            <p className="text-xs">{debtor?.name ?? 'The debtor'} must raise cash to pay.</p>
            <div className="debt-action-list" role="group" aria-label="Debt actions" tabIndex={0}>
              {gameState.decision_player === playerId ? (
                (gameState.legal_actions ?? [])
                  .filter((a) => debtActions[a.type])
                  .map((action) => (
                    <button
                      key={`${action.type}-${action.property_id}`}
                      className="game-button text-left"
                      disabled={!isConnected || pendingRequestId !== null}
                      onClick={() =>
                        send({
                          type: 'action',
                          data: {
                            action_type: debtActions[action.type],
                            ...(action.property_id !== undefined
                              ? { property_position: action.property_id }
                              : {}),
                          },
                        })
                      }
                    >
                      {action.type.replace(/([A-Z])/g, ' $1').trim()}
                      {action.property_id !== undefined
                        ? ` · ${PROPERTY_INFO[action.property_id]?.name ?? action.property_id}`
                        : ''}
                    </button>
                  ))
              ) : (
                <p className="text-sm">Waiting for {debtor?.name ?? 'the debtor'}…</p>
              )}
            </div>
          </section>
        ) : (
          <ActionPanel
            send={send}
            variant="board"
            onManageProperties={() => setDialog({ type: 'assets' })}
          />
        )}
      </div>
      <div className="center-history">
        <EventLog onOpenHistory={openHistory} />
      </div>
      <div className="center-latest">
        <LatestActivity onOpenHistory={openHistory} />
      </div>
    </div>
  );

  return (
    <div className="game-page" style={playerColorStyle}>
      <div className="game-page-header">
        <h1 className="font-semibold">{gameId ? `Game ${gameId.slice(0, 8)}` : 'Monopoly'}</h1>
        <ConnectionStatus state={connectionState} onReconnect={reconnect} />
        {isMyTurn && <span className="your-turn-pill">Your Turn!</span>}
      </div>
      <div className="game-layout">
        <div className="board-stage">
          <Board
            fitToContainer
            players={gameState?.players}
            properties={gameState?.properties}
            onSpaceClick={openProperty}
            centerContent={center}
          />
        </div>
        <PlayerPanel onSelect={(id) => setDialog({ type: 'player', id })} />
      </div>
      <PropertyModal
        position={activeDialog?.type === 'property' ? activeDialog.position : null}
        isOpen={activeDialog?.type === 'property'}
        onClose={closeDialog}
        send={send}
      />
      <Modal
        isOpen={activeDialog !== null && activeDialog.type !== 'property'}
        onClose={closeDialog}
        title={
          activeDialog?.type === 'trade'
            ? isIncoming
              ? 'Incoming trade offer'
              : 'Trade'
            : activeDialog?.type === 'history'
              ? 'Recent activity'
              : activeDialog?.type === 'assets'
                ? 'Manage properties'
                : (selectedPlayer?.name ?? 'Player details')
        }
        size="lg"
      >
        {activeDialog?.type === 'trade' && (
          <TradePanel key={tradeKey} send={send} variant="embedded" />
        )}
        {activeDialog?.type === 'history' && <EventLog className="history-dialog-feed" />}
        {activeDialog?.type === 'assets' && (
          <>
            <p className="mb-3 text-sm text-slate-600">
              Available building and mortgage actions. Your holdings are listed below.
            </p>
            <BuildingControls send={send} />
            {playerId !== null && (
              <PropertyList playerId={playerId} onPropertyClick={openProperty} />
            )}
          </>
        )}
        {selectedPlayer && gameState && (
          <>
            <PlayerCard
              player={selectedPlayer}
              isCurrentTurn={selectedPlayer.id === gameState.decision_player}
              isCurrentUser={selectedPlayer.id === playerId}
              properties={gameState.properties}
              inspection={gameState.agent_inspections?.[selectedPlayer.id]}
            />
            <h3 className="mb-2 mt-4 font-semibold">Holdings</h3>
            <PropertyList playerId={selectedPlayer.id} onPropertyClick={openProperty} />
          </>
        )}
      </Modal>
    </div>
  );
}
