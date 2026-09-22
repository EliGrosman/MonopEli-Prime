import { useState, useCallback } from 'react';
import type { LobbySettings as LobbySettingsType } from '@/types';

interface LobbySettingsProps {
  settings: LobbySettingsType;
  isHost: boolean;
  onUpdate?: (settings: Partial<LobbySettingsType>) => void;
}

/**
 * Lobby settings form for host to configure game options.
 */
export function LobbySettings({ settings, isHost, onUpdate }: LobbySettingsProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [localSettings, setLocalSettings] = useState(settings);

  const handleChange = useCallback(
    (key: keyof LobbySettingsType, value: number | boolean | string) => {
      setLocalSettings((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleSave = useCallback(() => {
    onUpdate?.(localSettings);
    setIsEditing(false);
  }, [localSettings, onUpdate]);

  const handleCancel = useCallback(() => {
    setLocalSettings(settings);
    setIsEditing(false);
  }, [settings]);

  if (!isHost) {
    // Read-only view for non-hosts
    return (
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Game Settings</h3>
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <dt className="text-gray-600">Starting Money</dt>
          <dd className="text-gray-900 font-medium">${settings.starting_money}</dd>

          <dt className="text-gray-600">Max Players</dt>
          <dd className="text-gray-900 font-medium">{settings.max_players}</dd>

          <dt className="text-gray-600">GO Salary</dt>
          <dd className="text-gray-900 font-medium">${settings.go_salary}</dd>

          <dt className="text-gray-600">Spectators</dt>
          <dd className="text-gray-900 font-medium">
            {settings.allow_spectators ? 'Allowed' : 'Not Allowed'}
          </dd>
          <dt className="text-gray-600">Trading</dt>
          <dd className="text-gray-900 font-medium">
            {settings.rules_id === 'foundation-trade-v1' ? 'Enabled' : 'Disabled'}
          </dd>
        </dl>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Game Settings</h3>
        {!isEditing && (
          <button
            onClick={() => {
              setLocalSettings(settings); // Sync to latest before editing
              setIsEditing(true);
            }}
            className="text-sm text-board-border hover:text-green-800"
          >
            Edit
          </button>
        )}
      </div>

      {isEditing ? (
        <div className="space-y-4">
          {/* Starting Money */}
          <div>
            <label htmlFor="rulesId" className="block text-sm font-medium text-gray-700 mb-1">
              Rules
            </label>
            <select
              id="rulesId"
              value={localSettings.rules_id ?? 'foundation-v1'}
              onChange={(e) => handleChange('rules_id', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-gray-900 bg-white"
            >
              <option value="foundation-v1">Foundation (no trading)</option>
              <option value="foundation-trade-v1">Foundation with trading</option>
            </select>
          </div>

          {/* Starting Money */}
          <div>
            <label htmlFor="startingMoney" className="block text-sm font-medium text-gray-700 mb-1">
              Starting Money
            </label>
            <select
              id="startingMoney"
              value={localSettings.starting_money}
              onChange={(e) => handleChange('starting_money', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border focus:border-transparent text-gray-900 bg-white"
            >
              <option value={1000}>$1,000</option>
              <option value={1500}>$1,500 (Standard)</option>
              <option value={2000}>$2,000</option>
              <option value={2500}>$2,500</option>
              <option value={3000}>$3,000</option>
            </select>
          </div>

          {/* Max Players */}
          <div>
            <label htmlFor="maxPlayers" className="block text-sm font-medium text-gray-700 mb-1">
              Max Players
            </label>
            <select
              id="maxPlayers"
              value={localSettings.max_players}
              onChange={(e) => handleChange('max_players', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border focus:border-transparent text-gray-900 bg-white"
            >
              <option value={2}>2 Players</option>
              <option value={3}>3 Players</option>
              <option value={4}>4 Players</option>
              <option value={5}>5 Players</option>
              <option value={6}>6 Players</option>
              <option value={7}>7 Players</option>
              <option value={8}>8 Players</option>
            </select>
          </div>

          {/* GO Salary */}
          <div>
            <label htmlFor="goSalary" className="block text-sm font-medium text-gray-700 mb-1">
              GO Salary
            </label>
            <select
              id="goSalary"
              value={localSettings.go_salary}
              onChange={(e) => handleChange('go_salary', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border focus:border-transparent text-gray-900 bg-white"
            >
              <option value={100}>$100</option>
              <option value={200}>$200 (Standard)</option>
              <option value={300}>$300</option>
              <option value={400}>$400</option>
              <option value={500}>$500</option>
            </select>
          </div>

          {/* Allow Spectators */}
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="allowSpectators"
              checked={localSettings.allow_spectators}
              onChange={(e) => handleChange('allow_spectators', e.target.checked)}
              className="w-4 h-4 text-board-border border-gray-300 rounded focus:ring-board-border"
            />
            <label htmlFor="allowSpectators" className="text-sm text-gray-700">
              Allow Spectators
            </label>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2 pt-2">
            <button
              onClick={handleSave}
              className="flex-1 px-4 py-2 bg-board-border text-white rounded-lg font-medium hover:bg-green-800 transition-colors"
            >
              Save
            </button>
            <button
              onClick={handleCancel}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <dt className="text-gray-600">Starting Money</dt>
          <dd className="text-gray-900 font-medium">${settings.starting_money}</dd>

          <dt className="text-gray-600">Max Players</dt>
          <dd className="text-gray-900 font-medium">{settings.max_players}</dd>

          <dt className="text-gray-600">GO Salary</dt>
          <dd className="text-gray-900 font-medium">${settings.go_salary}</dd>

          <dt className="text-gray-600">Spectators</dt>
          <dd className="text-gray-900 font-medium">
            {settings.allow_spectators ? 'Allowed' : 'Not Allowed'}
          </dd>
          <dt className="text-gray-600">Trading</dt>
          <dd className="text-gray-900 font-medium">
            {settings.rules_id === 'foundation-trade-v1' ? 'Enabled' : 'Disabled'}
          </dd>
        </dl>
      )}
    </div>
  );
}
