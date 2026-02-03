import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useUIStore } from '@/store/uiStore';
import { PlayerCard } from './PlayerCard';

/**
 * Sidebar showing all players' status.
 *
 * Displays for each player:
 * - Name and color
 * - Money and net worth
 * - Properties owned (grouped by color)
 * - Jail status
 * - Current turn indicator
 * - Bankrupt status
 */
export function PlayerPanel() {
  const { gameState } = useGameStore();
  const { playerId } = useSessionStore();
  const { selectPlayer, selectedPlayerId } = useUIStore();

  if (!gameState) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-lg font-semibold mb-4">Players</h2>
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }

  // Sort players: current user first, then by turn order
  const sortedPlayers = [...gameState.players].sort((a, b) => {
    if (a.id === playerId) return -1;
    if (b.id === playerId) return 1;
    return a.id - b.id;
  });

  // Count active players
  const activePlayers = gameState.players.filter((p) => !p.bankrupt).length;

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Players</h2>
        <span className="text-sm text-gray-500">
          {activePlayers} active / {gameState.players.length} total
        </span>
      </div>

      {/* Player list */}
      <div className="space-y-3">
        {sortedPlayers.map((player) => (
          <PlayerCard
            key={player.id}
            player={player}
            isCurrentTurn={gameState.currentPlayer === player.id}
            isCurrentUser={player.id === playerId}
            properties={gameState.properties}
            onSelect={(id) => selectPlayer(selectedPlayerId === id ? null : id)}
          />
        ))}
      </div>

      {/* Game stats */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Game Stats</h3>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-500">Turn</span>
            <span className="font-medium">{gameState.turnNumber}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Phase</span>
            <span className="font-medium capitalize">
              {gameState.gamePhase.replace('_', ' ')}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Houses</span>
            <span className="font-medium text-green-600">
              {gameState.housesRemaining}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Hotels</span>
            <span className="font-medium text-red-600">
              {gameState.hotelsRemaining}
            </span>
          </div>
        </div>
      </div>

      {/* Last roll display */}
      {gameState.lastRoll && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Last Roll</span>
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold">
                {gameState.lastRoll.die1} + {gameState.lastRoll.die2} ={' '}
                {gameState.lastRoll.die1 + gameState.lastRoll.die2}
              </span>
              {gameState.lastRoll.isDoubles && (
                <span className="px-2 py-0.5 text-xs bg-yellow-400 text-yellow-900 rounded-full">
                  Doubles!
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
