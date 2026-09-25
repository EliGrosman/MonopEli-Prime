import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { decisionDescription } from '@/utils/activity';

export function TurnStatus() {
  const { gameState, events, isConnected } = useGameStore();
  const playerId = useSessionStore((state) => state.playerId);
  if (!gameState) return null;
  const actor = gameState.players.find((p) => p.id === gameState.decision_player);
  const owner = gameState.players.find((p) => p.id === gameState.currentPlayer);
  const inspection = actor ? gameState.agent_inspections?.[actor.id] : undefined;
  const thinking =
    actor?.isAi &&
    inspection?.status === 'thinking' &&
    inspection.basis_revision === gameState.revision &&
    isConnected;
  const latest = [...events].reverse().find((event) => event.playerId === actor?.id);
  return (
    <section className="turn-status" aria-label="Current decision">
      <div className="flex min-w-0 items-center gap-2">
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ background: actor?.color ?? '#64748b' }}
          aria-hidden="true"
        />
        <h2 className="min-w-0 truncate font-semibold" title={actor?.name}>
          {gameState.gameOver
            ? 'Game finished'
            : actor?.id === playerId
              ? 'Your Turn'
              : (actor?.name ?? 'Waiting for players')}
        </h2>
        {thinking && <span className="shrink-0 text-xs font-medium text-blue-700">Thinking…</span>}
      </div>
      <p className="decision-description text-gray-600">
        {decisionDescription(gameState)}
        {actor?.id !== owner?.id && owner ? ` · ${owner.name}’s turn` : ''}
      </p>
      {actor?.isAi && inspection?.short_term_objective && !gameState.gameOver && (
        <p className="strategy-objective text-violet-900" title={inspection.short_term_objective}>
          <span className="font-semibold">Goal:</span> {inspection.short_term_objective}
        </p>
      )}
      {latest && (
        <p className="actor-last-action text-gray-600" title={latest.message}>
          Last: {latest.message}
        </p>
      )}
    </section>
  );
}
