import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { BOARD_SPACES } from '@/utils/board';
import { decisionDescription } from '@/utils/activity';

export function PlayerPanel({ onSelect }: { onSelect?: (id: number) => void }) {
  const { gameState, events } = useGameStore();
  const playerId = useSessionStore((state) => state.playerId);
  if (!gameState)
    return (
      <aside aria-label="Players panel" aria-busy="true">
        Loading players…
      </aside>
    );
  const latest = new Map<number, string>();
  events.forEach((event) => latest.set(event.playerId, event.message));
  return (
    <aside className="players-panel" aria-label="Players panel">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <h2 className="font-semibold">Players</h2>
        <span className="text-xs text-slate-600">
          {gameState.players.filter((p) => !p.bankrupt).length} active / {gameState.players.length}{' '}
          total
        </span>
      </div>
      <ul className="player-list" aria-label="Players in turn order">
        {[...gameState.players]
          .sort((a, b) => a.id - b.id)
          .map((player) => {
            const deciding = player.id === gameState.decision_player && !gameState.gameOver;
            return (
              <li key={player.id}>
                <button
                  className={`player-summary ${deciding ? 'player-deciding' : ''}`}
                  onClick={() => onSelect?.(player.id)}
                  aria-label={`View ${player.name}${player.id === playerId ? ' (You)' : ''} details`}
                  aria-current={deciding ? 'true' : undefined}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className="h-3 w-3 shrink-0 rounded-full"
                      style={{ background: player.color }}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate font-semibold" title={player.name}>
                      {player.name}
                      {player.id === playerId ? ' (You)' : ''}
                    </span>
                    <span className="shrink-0 font-mono text-emerald-800">
                      ${player.money.toLocaleString()}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-xs text-slate-600">
                    {player.bankrupt
                      ? 'Bankrupt'
                      : player.inJail
                        ? 'In jail'
                        : BOARD_SPACES[player.position]?.name}
                    {player.isAi ? ` · ${player.aiType === 'jev' ? 'Jev' : 'AI'}` : ''}
                  </p>
                  {deciding && (
                    <p className="mt-1 text-xs font-medium text-blue-800">
                      {decisionDescription(gameState)}
                    </p>
                  )}
                  <p
                    className="mt-1 line-clamp-2 text-xs text-slate-600"
                    title={latest.get(player.id)}
                  >
                    {latest.get(player.id) ?? 'No actions yet'}
                  </p>
                </button>
              </li>
            );
          })}
      </ul>
      <details className="game-details border-t border-slate-200 px-3 py-2 text-xs text-slate-700">
        <summary className="cursor-pointer font-medium">
          Game details · Turn {gameState.turnNumber}
        </summary>
        <p className="mt-2">Rules: {gameState.rules_id ?? 'foundation-v1'} · No auctions</p>
        <p className="mt-1">
          Bank: {gameState.housesRemaining} houses · {gameState.hotelsRemaining} hotels
        </p>
      </details>
    </aside>
  );
}
