import { useState, useCallback } from 'react';
import type { LobbyState, LobbySettings as LobbySettingsType } from '@/types';
import { PlayerSlot, EmptySlot } from './PlayerSlot';
import { LobbySettings } from './LobbySettings';
import { ConnectionStatus } from '@/components/common';
import type { ConnectionState } from '@/types/websocket';

interface LobbyRoomProps {
  lobby: LobbyState;
  sessionId: string;
  isHost: boolean;
  isLoading: boolean;
  connectionState: ConnectionState;
  onLeave: () => void;
  onToggleReady: () => void;
  onAddAi: (aiType?: string) => void;
  onRemoveAi: (slot: number) => void;
  onChangeSettings: (settings: Partial<LobbySettingsType>) => void;
  onStart: () => void;
}

/**
 * Main lobby room view showing players, settings, and controls.
 */
export function LobbyRoom({
  lobby,
  sessionId,
  isHost,
  isLoading,
  connectionState,
  onLeave,
  onToggleReady,
  onAddAi,
  onRemoveAi,
  onChangeSettings,
  onStart,
}: LobbyRoomProps) {
  const [showAiMenu, setShowAiMenu] = useState(false);

  const mySlot = lobby.players.find((p) => p.session_id === sessionId);
  const allReady = lobby.players.every((p) => p.is_ready || p.is_ai);
  const canStart = isHost && allReady && lobby.players.length >= 2;

  const handleCopyCode = useCallback(() => {
    const code = lobby.invite_code || lobby.id;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(code);
    } else {
      // Fallback for non-HTTPS contexts
      const textArea = document.createElement('textarea');
      textArea.value = code;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }
  }, [lobby.invite_code, lobby.id]);

  const handleAddAi = useCallback(
    (aiType: string) => {
      onAddAi(aiType);
      setShowAiMenu(false);
    },
    [onAddAi]
  );

  const emptySlots = lobby.settings.max_players - lobby.players.length;

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{lobby.name}</h1>
          {lobby.settings.private && (
            <span className="inline-block mt-1 px-2 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-800 rounded">
              Private Lobby
            </span>
          )}
        </div>
        <ConnectionStatus state={connectionState} />
      </div>

      {/* Lobby code */}
      <div className="bg-gray-100 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm text-gray-600">Lobby Code</span>
            <p className="text-2xl font-mono font-bold tracking-wider">
              {lobby.invite_code || lobby.id.slice(0, 8)}
            </p>
          </div>
          <button
            onClick={handleCopyCode}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            Copy
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-2">Share this code with friends to invite them</p>
      </div>

      {/* Players */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Players ({lobby.players.length}/{lobby.settings.max_players})
        </h2>

        <div className="space-y-2">
          {lobby.players.map((player) => (
            <PlayerSlot
              key={player.session_id}
              player={player}
              isHost={player.session_id === lobby.host_session_id}
              isSelf={player.session_id === sessionId}
              canManage={isHost}
              onKick={() => onRemoveAi(player.slot_id)}
              onRemoveAi={() => onRemoveAi(player.slot_id)}
            />
          ))}

          {/* Empty slots */}
          {Array.from({ length: emptySlots }).map((_, i) => (
            <EmptySlot
              key={`empty-${i}`}
              slotNumber={lobby.players.length + i}
              canAddAi={isHost}
              onAddAi={() => setShowAiMenu(true)}
            />
          ))}
        </div>

        {/* AI type selection menu */}
        {showAiMenu && (
          <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg shadow-lg">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Select AI Type</h4>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleAddAi('random')}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-board-border hover:bg-green-50 transition-colors"
              >
                <span className="font-medium">Random</span>
                <p className="text-xs text-gray-600">Makes random valid moves</p>
              </button>
              <button
                onClick={() => handleAddAi('rule_based')}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-board-border hover:bg-green-50 transition-colors"
              >
                <span className="font-medium">Rule-Based</span>
                <p className="text-xs text-gray-600">Uses basic strategies</p>
              </button>
              <button
                onClick={() => handleAddAi('aggressive')}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-board-border hover:bg-green-50 transition-colors"
              >
                <span className="font-medium">Aggressive</span>
                <p className="text-xs text-gray-600">Buys everything possible</p>
              </button>
              <button
                onClick={() => handleAddAi('conservative')}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-board-border hover:bg-green-50 transition-colors"
              >
                <span className="font-medium">Conservative</span>
                <p className="text-xs text-gray-600">Saves money carefully</p>
              </button>
            </div>
            <button
              onClick={() => setShowAiMenu(false)}
              className="mt-3 w-full text-sm text-gray-600 hover:text-gray-700"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="mb-6">
        <LobbySettings settings={lobby.settings} isHost={isHost} onUpdate={onChangeSettings} />
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-3">
        {/* Ready/Start buttons */}
        {isHost ? (
          <button
            onClick={onStart}
            disabled={!canStart || isLoading}
            className={`
              w-full py-3 rounded-lg font-bold text-lg transition-colors
              ${
                canStart
                  ? 'bg-board-border text-white hover:bg-green-800'
                  : 'bg-gray-300 text-gray-700 cursor-not-allowed'
              }
            `}
          >
            {isLoading
              ? 'Starting...'
              : canStart
                ? 'Start Game'
                : 'Waiting for players to ready up...'}
          </button>
        ) : mySlot && !mySlot.is_ai ? (
          <button
            onClick={onToggleReady}
            disabled={isLoading}
            className={`
              w-full py-3 rounded-lg font-bold text-lg transition-colors
              ${
                mySlot.is_ready
                  ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  : 'bg-board-border text-white hover:bg-green-800'
              }
            `}
          >
            {isLoading ? 'Updating...' : mySlot.is_ready ? 'Cancel Ready' : "I'm Ready!"}
          </button>
        ) : null}

        {/* Leave button */}
        <button
          onClick={onLeave}
          disabled={isLoading}
          className="w-full py-2 border border-red-300 text-red-600 rounded-lg font-medium hover:bg-red-50 transition-colors"
        >
          Leave Lobby
        </button>
      </div>

      {/* Status message */}
      {!canStart && isHost && (
        <p className="text-center text-sm text-gray-600 mt-4">
          {lobby.players.length < 2
            ? 'Need at least 2 players to start'
            : 'Waiting for all players to ready up'}
        </p>
      )}
    </div>
  );
}
