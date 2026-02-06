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
      <aside className="bg-white rounded-lg shadow-md p-4" aria-label="Players" aria-busy="true">
        <h2 className="text-lg font-semibold mb-4 text-gray-900">Players</h2>
        <p className="text-gray-500" role="status">
          Loading...
        </p>
      </aside>
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
    <aside className="bg-white rounded-lg shadow-md p-4" aria-label="Players panel">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900" id="players-heading">
          Players
        </h2>
        <span
          className="text-sm text-gray-500"
          aria-label={`${activePlayers} active players out of ${gameState.players.length} total`}
        >
          {activePlayers} active / {gameState.players.length} total
        </span>
      </div>

      {/* Player list */}
      <ul className="space-y-3" role="list" aria-labelledby="players-heading">
        {sortedPlayers.map((player) => (
          <li key={player.id}>
            <PlayerCard
              player={player}
              isCurrentTurn={gameState.currentPlayer === player.id}
              isCurrentUser={player.id === playerId}
              properties={gameState.properties}
              onSelect={(id) => selectPlayer(selectedPlayerId === id ? null : id)}
            />
          </li>
        ))}
      </ul>

      {/* Game stats */}
      <section className="mt-4 pt-4 border-t border-gray-200" aria-label="Game statistics">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Game Stats</h3>
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex justify-between">
            <dt className="text-gray-500">Turn</dt>
            <dd className="font-medium text-gray-900">{gameState.turnNumber}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Phase</dt>
            <dd className="font-medium capitalize text-gray-900">
              {gameState.gamePhase?.replace('_', ' ') ?? 'Playing'}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Houses</dt>
            <dd
              className="font-medium text-green-600"
              aria-label={`${gameState.housesRemaining} houses remaining`}
            >
              {gameState.housesRemaining}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Hotels</dt>
            <dd
              className="font-medium text-red-600"
              aria-label={`${gameState.hotelsRemaining} hotels remaining`}
            >
              {gameState.hotelsRemaining}
            </dd>
          </div>
        </dl>
      </section>

      {/* Last roll display */}
      {gameState.lastRoll && (
        <section className="mt-4 pt-4 border-t border-gray-200" aria-label="Last dice roll">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Last Roll</span>
            <div className="flex items-center gap-2">
              <span
                className="font-mono font-bold text-gray-900"
                aria-label={`Rolled ${gameState.lastRoll.die1} and ${gameState.lastRoll.die2} for a total of ${gameState.lastRoll.die1 + gameState.lastRoll.die2}`}
              >
                {gameState.lastRoll.die1} + {gameState.lastRoll.die2} ={' '}
                {gameState.lastRoll.die1 + gameState.lastRoll.die2}
              </span>
              {gameState.lastRoll.isDoubles && (
                <span
                  className="px-2 py-0.5 text-xs bg-yellow-400 text-yellow-900 rounded-full"
                  role="status"
                  aria-live="polite"
                >
                  Doubles!
                </span>
              )}
            </div>
          </div>
        </section>
      )}
    </aside>
  );
}
