import type { LobbyPlayer } from '@/types';
import { getPlayerColor } from '@/utils/colors';

interface PlayerSlotProps {
  player: LobbyPlayer;
  isHost: boolean;
  isSelf: boolean;
  canManage: boolean;
  onKick?: () => void;
  onRemoveAi?: () => void;
}

/**
 * Individual player slot in lobby room.
 */
export function PlayerSlot({
  player,
  isHost,
  isSelf,
  canManage,
  onKick,
  onRemoveAi,
}: PlayerSlotProps) {
  const playerColor = getPlayerColor(player.slot_id);

  return (
    <div
      className={`
        flex items-center gap-3 p-3 bg-white rounded-lg border-2 transition-colors
        ${isSelf ? 'border-board-border bg-green-50' : 'border-gray-200'}
      `}
    >
      {/* Player color indicator */}
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
        style={{ backgroundColor: playerColor }}
      >
        {player.is_ai ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        ) : (
          player.name.charAt(0).toUpperCase()
        )}
      </div>

      {/* Player info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900 truncate">{player.name}</span>
          {isHost && (
            <span className="px-1.5 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 rounded">
              Host
            </span>
          )}
          {player.is_ai && (
            <span className="px-1.5 py-0.5 text-xs font-medium bg-purple-100 text-purple-800 rounded">
              AI
            </span>
          )}
          {isSelf && (
            <span className="px-1.5 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded">
              You
            </span>
          )}
        </div>
        {player.is_ai && player.ai_type && (
          <p className="text-xs text-gray-600 capitalize">{player.ai_type.replace('_', ' ')}</p>
        )}
      </div>

      {/* Ready status */}
      <div className="flex items-center gap-2">
        {!player.is_ai && (
          <div
            className={`
              flex items-center gap-1 px-2 py-1 rounded text-sm font-medium
              ${player.is_ready ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}
            `}
          >
            {player.is_ready ? (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
                Ready
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Waiting
              </>
            )}
          </div>
        )}

        {/* Management buttons */}
        {canManage && !isSelf && (
          <button
            onClick={player.is_ai ? onRemoveAi : onKick}
            className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
            title={player.is_ai ? 'Remove AI' : 'Kick player'}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

interface EmptySlotProps {
  slotNumber: number;
  canAddAi: boolean;
  onAddAi?: () => void;
}

/**
 * Empty player slot with option to add AI.
 */
export function EmptySlot({ slotNumber, canAddAi, onAddAi }: EmptySlotProps) {
  const playerColor = getPlayerColor(slotNumber);

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
      {/* Empty indicator */}
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center border-2 opacity-50"
        style={{ borderColor: playerColor }}
      >
        <span className="text-gray-400 font-medium">?</span>
      </div>

      {/* Slot info */}
      <div className="flex-1">
        <span className="text-gray-600">Empty Slot</span>
      </div>

      {/* Add AI button */}
      {canAddAi && onAddAi && (
        <button
          onClick={onAddAi}
          className="px-3 py-1.5 text-sm font-medium text-board-border hover:text-green-800 border border-board-border hover:bg-green-50 rounded transition-colors"
        >
          + Add AI
        </button>
      )}
    </div>
  );
}
