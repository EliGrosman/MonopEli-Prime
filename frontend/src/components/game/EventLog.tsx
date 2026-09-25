import { useMemo, useRef, useState } from 'react';
import type { GameEvent, GameEventType } from '@/types';
import { useGameStore } from '@/store';
import { getPlayerColor } from '@/utils/colors';

interface EventLogProps {
  maxEvents?: number;
  className?: string;
  onOpenHistory?: () => void;
}

export function EventLog({ maxEvents = 100, className = '', onOpenHistory }: EventLogProps) {
  const allEvents = useGameStore((state) => state.events);
  const events = useMemo(() => allEvents.slice(-maxEvents).reverse(), [allEvents, maxEvents]);
  // Freeze the reader's snapshot while they explore older actions. A new action
  // must never move the line they are reading, including at the history cap.
  const [reading, setReading] = useState<GameEvent[] | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const newActivity = reading !== null && events[0]?.id !== reading[0]?.id;
  const visibleEvents = reading ?? events;
  const followLatest = () => {
    setReading(null);
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  };

  return (
    <section className={`activity-feed ${className}`} aria-label="Activity feed">
      <div className="activity-heading">
        <h3 className="font-semibold">Activity</h3>
        <div className="flex items-center gap-2">
          {newActivity && (
            <button
              className="text-xs font-semibold text-blue-800 underline"
              onClick={followLatest}
            >
              New activity
            </button>
          )}
          {onOpenHistory && (
            <button
              className="text-xs font-medium text-slate-600 underline"
              onClick={onOpenHistory}
            >
              History
            </button>
          )}
        </div>
      </div>
      <div
        ref={scrollRef}
        className="activity-scroll"
        role="log"
        aria-live={reading ? 'off' : 'polite'}
        aria-relevant="additions"
        aria-label="Game events"
        tabIndex={0}
        onScroll={(event) => {
          if (event.currentTarget.scrollTop <= 12) setReading(null);
          else if (reading === null) setReading(events);
        }}
      >
        {visibleEvents.length === 0 ? (
          <p className="p-3 text-sm text-slate-600">No events yet</p>
        ) : (
          visibleEvents.map((event) => <EventItem key={event.id} event={event} />)
        )}
      </div>
    </section>
  );
}

function EventItem({ event }: { event: GameEvent }) {
  const player = useGameStore((state) =>
    state.gameState?.players.find((p) => p.id === event.playerId)
  );
  return (
    <article className="activity-entry" data-event-id={event.id}>
      <span
        className="mt-1 h-2 w-2 shrink-0 rounded-full"
        style={{ background: player?.color ?? getPlayerColor(event.playerId) }}
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-slate-900">{event.message}</p>
        {event.details && event.details.length > 0 && (
          <ul className="mt-1 space-y-0.5 text-xs text-slate-600">
            {event.details.map((detail, index) => (
              <li key={index}>{detail}</li>
            ))}
          </ul>
        )}
        <p className="mt-1 text-[11px] text-slate-500">
          {event.turnNumber !== undefined ? `Turn ${event.turnNumber}` : 'Recorded action'}
          {event.timestamp > 0 && Number.isFinite(event.timestamp)
            ? ` · ${new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
            : ''}
        </p>
      </div>
    </article>
  );
}

export function CompactEventLog({ maxEvents = 5 }: { maxEvents?: number }) {
  const allEvents = useGameStore((state) => state.events);
  const events = allEvents.slice(-maxEvents).reverse();
  if (!events.length) return null;
  return (
    <div className="space-y-1">
      {events.map((event) => (
        <p key={event.id} className="text-xs text-slate-700" title={event.message}>
          {getEventIcon(event.type)} {event.message}
        </p>
      ))}
    </div>
  );
}

export function LatestActivity({ onOpenHistory }: { onOpenHistory: () => void }) {
  const latest = useGameStore((state) => state.events.at(-1));
  return (
    <button className="latest-activity" onClick={onOpenHistory} aria-label="Open activity history">
      <span className="shrink-0 text-[10px] font-bold uppercase tracking-wide text-blue-800">
        Activity ›
      </span>
      <span className="line-clamp-2 text-left text-xs text-slate-800">
        {latest?.message ?? 'No events yet'}
      </span>
    </button>
  );
}

function getEventIcon(type: GameEventType): string {
  const icons: Record<GameEventType, string> = {
    roll: '🎲',
    move: '👟',
    buy: '🏠',
    rent: '💰',
    build: '🔨',
    mortgage: '📋',
    jail: '🔒',
    card: '🃏',
    trade: '🤝',
    agent: '🧠',
    bankrupt: '💸',
    win: '🏆',
  };
  return icons[type];
}
