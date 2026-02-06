import type { LobbyListItem } from '@/types';

interface LobbyCardProps {
  lobby: LobbyListItem;
  onJoin: (lobbyId: string) => void;
  disabled?: boolean;
}

/**
 * Preview card for a single lobby in the lobby list.
 */
export function LobbyCard({ lobby, onJoin, disabled = false }: LobbyCardProps) {
  const isFull = lobby.current_players >= lobby.max_players;

  return (
    <div
      className={`
        bg-white rounded-lg shadow-md border border-gray-200 p-4
        transition-all duration-200
        ${!isFull && !disabled ? 'hover:shadow-lg hover:border-board-border cursor-pointer' : ''}
        ${isFull ? 'opacity-75' : ''}
      `}
      onClick={() => !isFull && !disabled && onJoin(lobby.id)}
      role="button"
      tabIndex={isFull || disabled ? -1 : 0}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !isFull && !disabled) {
          onJoin(lobby.id);
        }
      }}
      aria-label={`Join lobby ${lobby.name}`}
      aria-disabled={isFull || disabled}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-lg text-gray-900">{lobby.name}</h3>
          <p className="text-sm text-gray-700">Hosted by {lobby.host_name}</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <svg
              className="w-4 h-4 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
              />
            </svg>
            <span className="text-sm text-gray-700">
              {lobby.current_players}/{lobby.max_players}
            </span>
          </div>

          {isFull && (
            <span className="px-2 py-0.5 text-xs font-medium bg-red-100 text-red-800 rounded">
              Full
            </span>
          )}
        </div>

        <div className="text-xs text-gray-600 font-mono">{lobby.id.slice(0, 8)}</div>
      </div>
    </div>
  );
}
