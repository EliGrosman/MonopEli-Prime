import { useRef, useEffect, useMemo } from 'react';
import type { GameEvent, GameEventType } from '@/types';
import { useGameStore } from '@/store';
import { getPlayerColor } from '@/utils/colors';
import { formatRelativeTime } from '@/utils/format';

interface EventLogProps {
  maxEvents?: number;
  className?: string;
}

/**
 * Game event history log showing rolls, purchases, rent payments, etc.
 */
export function EventLog({ maxEvents = 20, className = '' }: EventLogProps) {
  const allEvents = useGameStore((state) => state.events);
  const events = useMemo(() => allEvents.slice(-maxEvents).reverse(), [allEvents, maxEvents]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Game Log</h3>
        <p className="text-sm text-gray-500 text-center py-4">No events yet</p>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      <div className="p-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">Game Log</h3>
      </div>
      <div
        ref={scrollRef}
        className="max-h-64 overflow-y-auto p-2 space-y-1"
        role="log"
        aria-live="polite"
        aria-label="Game events"
      >
        {events.map((event) => (
          <EventItem key={event.id} event={event} />
        ))}
      </div>
    </div>
  );
}

interface EventItemProps {
  event: GameEvent;
}

function EventItem({ event }: EventItemProps) {
  const icon = getEventIcon(event.type);
  const playerColor = getPlayerColor(event.playerId);

  return (
    <div className="flex items-start gap-2 px-2 py-1.5 hover:bg-gray-50 rounded text-sm">
      {/* Event icon */}
      <div
        className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-white text-xs"
        style={{ backgroundColor: playerColor }}
      >
        {icon}
      </div>

      {/* Event message */}
      <div className="flex-1 min-w-0">
        <p className="text-gray-800">{event.message}</p>
        <p className="text-xs text-gray-400">{formatRelativeTime(event.timestamp)}</p>
      </div>
    </div>
  );
}

function getEventIcon(type: GameEventType): string {
  switch (type) {
    case 'roll':
      return '🎲';
    case 'move':
      return '👟';
    case 'buy':
      return '🏠';
    case 'rent':
      return '💰';
    case 'build':
      return '🔨';
    case 'mortgage':
      return '📋';
    case 'jail':
      return '🔒';
    case 'card':
      return '🃏';
    case 'trade':
      return '🤝';
    case 'bankrupt':
      return '💸';
    case 'win':
      return '🏆';
    default:
      return '📌';
  }
}

/**
 * Compact event log for sidebar use.
 */
export function CompactEventLog({ maxEvents = 5 }: { maxEvents?: number }) {
  const allEvents = useGameStore((state) => state.events);
  const events = useMemo(() => allEvents.slice(-maxEvents).reverse(), [allEvents, maxEvents]);

  if (events.length === 0) {
    return null;
  }

  return (
    <div className="space-y-1">
      {events.map((event) => (
        <div key={event.id} className="text-xs text-gray-600 truncate" title={event.message}>
          {getEventIcon(event.type)} {event.message}
        </div>
      ))}
    </div>
  );
}
