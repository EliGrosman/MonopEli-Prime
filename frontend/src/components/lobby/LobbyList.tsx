import { useEffect } from 'react';
import type { LobbyListItem } from '@/types';
import { LobbyCard } from './LobbyCard';

interface LobbyListProps {
  lobbies: LobbyListItem[];
  isLoading: boolean;
  onJoin: (lobbyId: string) => void;
  onRefresh?: () => void;
}

/**
 * Grid display of available lobbies.
 */
export function LobbyList({ lobbies, isLoading, onJoin, onRefresh }: LobbyListProps) {
  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (onRefresh) {
      const interval = setInterval(onRefresh, 30000);
      return () => clearInterval(interval);
    }
  }, [onRefresh]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="bg-white rounded-lg shadow-md border border-gray-200 p-4 animate-pulse"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="space-y-2">
                <div className="h-5 w-32 bg-gray-200 rounded" />
                <div className="h-4 w-24 bg-gray-100 rounded" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="h-4 w-16 bg-gray-100 rounded" />
              <div className="h-4 w-12 bg-gray-100 rounded" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (lobbies.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
        <svg
          className="w-12 h-12 mx-auto text-gray-400 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
          />
        </svg>
        <h3 className="text-lg font-medium text-gray-900 mb-1">No public games available</h3>
        <p className="text-gray-600 mb-4">Create a new game or join with a code</p>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-sm text-board-border hover:text-green-800 underline"
          >
            Refresh list
          </button>
        )}
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-gray-600">
          {lobbies.length} {lobbies.length === 1 ? 'game' : 'games'} available
        </p>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-sm text-gray-600 hover:text-gray-700 flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            Refresh
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {lobbies.map((lobby) => (
          <LobbyCard key={lobby.id} lobby={lobby} onJoin={onJoin} />
        ))}
      </div>
    </div>
  );
}
