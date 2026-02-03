import { BOARD_LAYOUT, BOARD_SPACES } from '@/utils/board';
import { BoardSpace } from './BoardSpace';
import { PlayerToken } from './PlayerToken';
import type { PlayerState, PropertyState } from '@/types';

interface BoardProps {
  players?: PlayerState[];
  properties?: Record<number, PropertyState>;
  onSpaceClick?: (position: number) => void;
  highlightedSpaces?: number[];
}

/**
 * Main Monopoly board component.
 *
 * Renders the 40-space board in a square layout with:
 * - Property spaces with color bands
 * - Corner spaces (GO, Jail, Free Parking, Go To Jail)
 * - Chance and Community Chest spaces
 * - Railroad and Utility spaces
 * - Tax spaces
 * - Player tokens at their positions
 * - House/hotel indicators on developed properties
 *
 * Layout uses CSS Grid for the square board shape.
 */
export function Board({
  players = [],
  properties = {},
  onSpaceClick,
  highlightedSpaces = [],
}: BoardProps) {
  const getPlayersAtPosition = (position: number): PlayerState[] => {
    return players.filter((p) => p.position === position && !p.bankrupt);
  };

  const getSpaceInfo = (position: number) => {
    return BOARD_SPACES.find((s) => s.position === position);
  };

  return (
    <div className="board-container p-4" data-testid="board">
      <div
        className="board-grid relative"
        style={{
          display: 'grid',
          gridTemplateColumns: 'var(--corner-size) repeat(9, var(--space-width)) var(--corner-size)',
          gridTemplateRows: 'var(--corner-size) repeat(9, var(--space-width)) var(--corner-size)',
          width: 'var(--board-size)',
          height: 'var(--board-size)',
          backgroundColor: 'var(--color-board-bg)',
          border: '3px solid var(--color-board-border)',
          borderRadius: '8px',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        {/* Bottom row (GO to Jail) - right to left */}
        {BOARD_LAYOUT.bottom.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 11 - index,
                gridRow: 11,
              }}
            >
              <BoardSpace
                position={position}
                name={spaceInfo?.name || ''}
                type={spaceInfo?.type || 'property'}
                color={spaceInfo?.color}
                property={properties[position]}
                isHighlighted={highlightedSpaces.includes(position)}
                onClick={() => onSpaceClick?.(position)}
                orientation="bottom"
              >
                {playersHere.map((player, idx) => (
                  <PlayerToken
                    key={player.id}
                    playerId={player.id}
                    color={player.color}
                    offset={idx}
                  />
                ))}
              </BoardSpace>
            </div>
          );
        })}

        {/* Left column (St. Charles to New York) - bottom to top */}
        {BOARD_LAYOUT.left.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 1,
                gridRow: 10 - index,
              }}
            >
              <BoardSpace
                position={position}
                name={spaceInfo?.name || ''}
                type={spaceInfo?.type || 'property'}
                color={spaceInfo?.color}
                property={properties[position]}
                isHighlighted={highlightedSpaces.includes(position)}
                onClick={() => onSpaceClick?.(position)}
                orientation="left"
              >
                {playersHere.map((player, idx) => (
                  <PlayerToken
                    key={player.id}
                    playerId={player.id}
                    color={player.color}
                    offset={idx}
                  />
                ))}
              </BoardSpace>
            </div>
          );
        })}

        {/* Top row (Free Parking to Go To Jail) - left to right */}
        {BOARD_LAYOUT.top.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: index + 1,
                gridRow: 1,
              }}
            >
              <BoardSpace
                position={position}
                name={spaceInfo?.name || ''}
                type={spaceInfo?.type || 'property'}
                color={spaceInfo?.color}
                property={properties[position]}
                isHighlighted={highlightedSpaces.includes(position)}
                onClick={() => onSpaceClick?.(position)}
                orientation="top"
              >
                {playersHere.map((player, idx) => (
                  <PlayerToken
                    key={player.id}
                    playerId={player.id}
                    color={player.color}
                    offset={idx}
                  />
                ))}
              </BoardSpace>
            </div>
          );
        })}

        {/* Right column (Pacific to Boardwalk) - top to bottom */}
        {BOARD_LAYOUT.right.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 11,
                gridRow: index + 2,
              }}
            >
              <BoardSpace
                position={position}
                name={spaceInfo?.name || ''}
                type={spaceInfo?.type || 'property'}
                color={spaceInfo?.color}
                property={properties[position]}
                isHighlighted={highlightedSpaces.includes(position)}
                onClick={() => onSpaceClick?.(position)}
                orientation="right"
              >
                {playersHere.map((player, idx) => (
                  <PlayerToken
                    key={player.id}
                    playerId={player.id}
                    color={player.color}
                    offset={idx}
                  />
                ))}
              </BoardSpace>
            </div>
          );
        })}

        {/* Center area */}
        <div
          className="board-center flex flex-col items-center justify-center"
          style={{
            gridColumn: '2 / 11',
            gridRow: '2 / 11',
          }}
        >
          <h1 className="text-4xl font-bold text-board-border mb-4">MONOPOLY</h1>
          <p className="text-gray-600 text-sm">The Classic Property Trading Game</p>
        </div>
      </div>
    </div>
  );
}
