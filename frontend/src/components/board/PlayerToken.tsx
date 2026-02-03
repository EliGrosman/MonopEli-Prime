import { getPlayerColor } from '@/utils/colors';

interface PlayerTokenProps {
  playerId: number;
  color?: string;
  offset?: number;
}

/**
 * Player token component.
 * Displays a colored circle representing a player on the board.
 */
export function PlayerToken({ playerId, color, offset = 0 }: PlayerTokenProps) {
  const tokenColor = color || getPlayerColor(playerId);

  // Calculate offset position for stacking multiple tokens
  const offsetStyle = {
    marginLeft: offset > 0 ? `-${offset * 4}px` : '0',
    zIndex: offset,
  };

  return (
    <div
      className="player-token w-4 h-4 rounded-full border-2 border-white shadow-sm animate-token-move"
      style={{
        backgroundColor: tokenColor,
        ...offsetStyle,
      }}
      data-testid={`token-player-${playerId}`}
      title={`Player ${playerId + 1}`}
      aria-label={`Player ${playerId + 1} token`}
    />
  );
}
