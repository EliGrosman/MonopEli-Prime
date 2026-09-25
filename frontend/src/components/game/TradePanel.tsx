import { useMemo, useState } from 'react';
import { useActions } from '@/hooks/useActions';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { PROPERTY_INFO } from '@/utils/board';
import type { ClientActionMessage } from '@/types';

interface TradePanelProps {
  send: (message: ClientActionMessage) => void;
  variant?: 'card' | 'embedded';
  onActionSubmitted?: () => void;
}

function propertyLabel(position: number, mortgaged: boolean): string {
  const name = PROPERTY_INFO[position]?.name ?? `Property ${position}`;
  return `${name}${mortgaged ? ' (mortgaged)' : ''}`;
}

export function TradePanel({ send, variant = 'card', onActionSubmitted }: TradePanelProps) {
  const gameState = useGameStore((state) => state.gameState);
  const playerId = useSessionStore((state) => state.playerId);
  const { proposeTradeOffer, acceptTrade, rejectTrade, isActionPending } = useActions({ send });
  const [recipient, setRecipient] = useState<number | null>(null);
  const [give, setGive] = useState<number[]>([]);
  const [want, setWant] = useState<number[]>([]);
  const [cash, setCash] = useState(0);
  const [cashDirection, setCashDirection] = useState<'give' | 'want'>('give');
  const [reviewing, setReviewing] = useState(false);
  const contract = gameState?.decision_contract;
  const offer = contract?.pending_offer;
  const mine = useMemo(
    () =>
      Object.values(gameState?.properties ?? {}).filter(
        (property) =>
          property.owner === playerId &&
          contract?.trade.tradeable_properties.includes(property.position)
      ),
    [contract, gameState, playerId]
  );
  const theirs = useMemo(
    () =>
      Object.values(gameState?.properties ?? {}).filter(
        (property) =>
          property.owner === recipient &&
          contract?.trade.tradeable_properties.includes(property.position)
      ),
    [contract, gameState, recipient]
  );

  if (!gameState || gameState.rules_id !== 'foundation-trade-v1' || playerId === null) return null;

  const toggle = (position: number, values: number[], setValues: (values: number[]) => void) => {
    if (values.includes(position)) setValues(values.filter((value) => value !== position));
    else if (values.length < 2) setValues([...values, position].sort((a, b) => a - b));
  };

  if (offer) {
    const responding = offer.to_player === playerId && gameState.decision_player === playerId;
    return (
      <section
        className={`${variant === 'embedded' ? '' : 'rounded-lg border border-blue-200 bg-blue-50 p-4'}`}
        aria-label="Trade offer"
      >
        <h2 className="font-semibold text-blue-900">
          {responding ? 'Your response' : 'Trade response pending'}
        </h2>
        <p className="text-sm mt-2">
          Player {offer.from_player + 1} gives{' '}
          {offer.give_properties
            .map((position) => propertyLabel(position, gameState.properties[position].mortgaged))
            .join(', ') || 'no properties'}
          {offer.give_money ? ` and $${offer.give_money}` : ''}.
        </p>
        <p className="text-sm">
          Player {offer.to_player + 1} gives{' '}
          {offer.want_properties
            .map((position) => propertyLabel(position, gameState.properties[position].mortgaged))
            .join(', ') || 'no properties'}
          {offer.want_money ? ` and $${offer.want_money}` : ''}.
        </p>
        <p className="text-xs text-gray-600 mt-2">
          Mortgaged properties remain mortgaged; this ruleset charges no immediate transfer fee.
        </p>
        {responding && (
          <div className="flex gap-2 mt-3">
            <button
              className="flex-1 p-2 rounded bg-green-600 text-white"
              disabled={isActionPending}
              onClick={() => {
                acceptTrade(offer.trade_id);
                onActionSubmitted?.();
              }}
            >
              Accept
            </button>
            <button
              className="flex-1 p-2 rounded bg-gray-700 text-white"
              disabled={isActionPending}
              onClick={() => {
                rejectTrade(offer.trade_id);
                onActionSubmitted?.();
              }}
            >
              Reject
            </button>
          </div>
        )}
      </section>
    );
  }
  if (!contract?.trade.can_propose || gameState.decision_player !== playerId) return null;

  const submit = () => {
    if (recipient === null || (!give.length && !want.length)) return;
    proposeTradeOffer(recipient, {
      give_properties: give,
      want_properties: want,
      give_money: cashDirection === 'give' ? cash : 0,
      want_money: cashDirection === 'want' ? cash : 0,
    });
    onActionSubmitted?.();
  };

  const propertyDetails = (position: number) =>
    contract.properties.find((property) => property.position === position);
  const playerMoney = gameState.players[playerId].money;
  const recipientMoney = recipient === null ? 0 : gameState.players[recipient].money;
  const giveMoney = cashDirection === 'give' ? cash : 0;
  const wantMoney = cashDirection === 'want' ? cash : 0;

  if (reviewing && recipient !== null) {
    return (
      <section
        className={`${variant === 'embedded' ? '' : 'rounded-lg border border-blue-200 bg-blue-50 p-4'}`}
        aria-label="Review trade"
      >
        <h2 className="font-semibold text-blue-900">Review offer</h2>
        <p className="text-sm mt-2">
          You give{' '}
          {give
            .map((position) => propertyLabel(position, gameState.properties[position].mortgaged))
            .join(', ') || 'no properties'}
          {giveMoney ? ` and $${giveMoney}` : ''}.
        </p>
        <p className="text-sm">
          You receive{' '}
          {want
            .map((position) => propertyLabel(position, gameState.properties[position].mortgaged))
            .join(', ') || 'no properties'}
          {wantMoney ? ` and $${wantMoney}` : ''}.
        </p>
        <p className="text-sm mt-2">
          Cash after transfer: you ${playerMoney - giveMoney + wantMoney};{' '}
          {gameState.players[recipient].name} ${recipientMoney + giveMoney - wantMoney}.
        </p>
        {[...give, ...want].map((position) => {
          const property = propertyDetails(position);
          return property?.mortgaged ? (
            <p key={position} className="text-xs text-gray-600">
              {property.name} remains mortgaged; redemption costs ${property.redemption_cost} later.
            </p>
          ) : null;
        })}
        <p className="text-xs text-gray-600 mt-2">
          This ruleset charges no immediate mortgage transfer fee.
        </p>
        <div className="flex gap-2 mt-3">
          <button className="flex-1 p-2 rounded border" onClick={() => setReviewing(false)}>
            Edit
          </button>
          <button
            className="flex-1 p-2 rounded bg-blue-600 text-white disabled:opacity-50"
            disabled={isActionPending}
            onClick={submit}
          >
            Send offer
          </button>
        </div>
      </section>
    );
  }

  return (
    <section
      className={variant === 'embedded' ? '' : 'bg-white rounded-lg shadow-md p-4'}
      aria-label="Compose trade"
    >
      <div className="flex justify-between gap-2">
        <h2 className="font-semibold">Trade</h2>
        <span className="text-sm text-gray-500">
          {contract.trade.proposals_remaining} offers left
        </span>
      </div>
      <label className="block text-sm mt-3">
        Recipient
        <select
          className="block w-full border rounded p-2 mt-1"
          value={recipient ?? ''}
          onChange={(event) => {
            setRecipient(Number(event.target.value));
            setWant([]);
          }}
        >
          <option value="" disabled>
            Select a player
          </option>
          {contract.trade.eligible_recipients.map((id) => (
            <option key={id} value={id}>
              {gameState.players[id]?.name ?? `Player ${id + 1}`}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="mt-3">
        <legend className="text-sm font-medium">You give (up to two)</legend>
        {mine.map((property) => (
          <label key={property.position} className="block text-sm">
            <input
              type="checkbox"
              checked={give.includes(property.position)}
              onChange={() => toggle(property.position, give, setGive)}
            />{' '}
            {propertyLabel(property.position, property.mortgaged)}
          </label>
        ))}
      </fieldset>
      {recipient !== null && (
        <fieldset className="mt-3">
          <legend className="text-sm font-medium">You receive (up to two)</legend>
          {theirs.map((property) => (
            <label key={property.position} className="block text-sm">
              <input
                type="checkbox"
                checked={want.includes(property.position)}
                onChange={() => toggle(property.position, want, setWant)}
              />{' '}
              {propertyLabel(property.position, property.mortgaged)}
            </label>
          ))}
        </fieldset>
      )}
      <div className="flex gap-2 mt-3">
        <select
          aria-label="Cash direction"
          className="border rounded p-2"
          value={cashDirection}
          onChange={(event) => setCashDirection(event.target.value as 'give' | 'want')}
        >
          <option value="give">You pay</option>
          <option value="want">You receive</option>
        </select>
        <input
          aria-label="Trade cash"
          className="min-w-0 flex-1 border rounded p-2"
          type="number"
          min={0}
          value={cash}
          onChange={(event) => setCash(Math.max(0, Number(event.target.value)))}
        />
      </div>
      <button
        className="w-full p-2 mt-3 rounded bg-blue-600 text-white disabled:opacity-50"
        disabled={isActionPending || recipient === null || (!give.length && !want.length)}
        onClick={() => setReviewing(true)}
      >
        Review and send offer
      </button>
    </section>
  );
}
