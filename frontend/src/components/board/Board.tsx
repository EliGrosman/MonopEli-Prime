import { BOARD_LAYOUT, BOARD_SPACES } from '@/utils/board';
import { BoardSpace } from './BoardSpace';
import { PlayerToken } from './PlayerToken';
import { useResponsive } from '@/hooks';
import type { PlayerState, PropertyState } from '@/types';

interface BoardProps {
  players?: PlayerState[];
  properties?: Record<number, PropertyState>;
  onSpaceClick?: (position: number) => void;
  highlightedSpaces?: number[];
  /** Allow scrolling/panning on mobile */
  enableMobileScroll?: boolean;
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
  enableMobileScroll = true,
}: BoardProps) {
  const { isMobile, isTablet } = useResponsive();

  const getPlayersAtPosition = (position: number): PlayerState[] => {
    return players.filter((p) => p.position === position && !p.bankrupt);
  };

  const getSpaceInfo = (position: number) => {
    return BOARD_SPACES.find((s) => s.position === position);
  };

  // Container classes for responsive layout
  const containerClasses = [
    'board-container',
    isMobile && enableMobileScroll ? 'overflow-auto touch-pan-x touch-pan-y' : '',
    isTablet ? 'flex justify-center' : '',
    'p-2 sm:p-4',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={containerClasses} data-testid="board">
      <div
        className="board-grid relative mx-auto"
        style={{
          display: 'grid',
          gridTemplateColumns:
            'var(--corner-size) repeat(9, var(--space-width)) var(--corner-size)',
          gridTemplateRows: 'var(--corner-size) repeat(9, var(--space-width)) var(--corner-size)',
          width: 'var(--board-size)',
          height: 'var(--board-size)',
          minWidth: isMobile ? '320px' : undefined,
          minHeight: isMobile ? '320px' : undefined,
          backgroundColor: 'var(--color-board-bg)',
          border: isMobile
            ? '2px solid var(--color-board-border)'
            : '3px solid var(--color-board-border)',
          borderRadius: isMobile ? '4px' : '8px',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        {/* Bottom row: Jail (10) on left to GO (0) on right */}
        {BOARD_LAYOUT.bottom.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: index + 1,
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

        {/* Left column: New York (19) at top to St. Charles (11) at bottom */}
        {BOARD_LAYOUT.left.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 1,
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

        {/* Top row: Go To Jail (30) on right to Free Parking (20) on left */}
        {BOARD_LAYOUT.top.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 11 - index,
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

        {/* Right column: Boardwalk (39) at bottom to Pacific (31) at top */}
        {BOARD_LAYOUT.right.map((position, index) => {
          const spaceInfo = getSpaceInfo(position);
          const playersHere = getPlayersAtPosition(position);
          return (
            <div
              key={position}
              style={{
                gridColumn: 11,
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
          className="board-center flex flex-col items-center justify-center px-2"
          style={{
            gridColumn: '2 / 11',
            gridRow: '2 / 11',
          }}
        >
          <h1
            className="font-bold text-board-border mb-1 sm:mb-2 md:mb-4 text-center"
            style={{ fontSize: 'var(--board-title-size)' }}
          >
            MONOPOLY
          </h1>
          <p
            className="text-gray-600 text-center hidden sm:block"
            style={{ fontSize: 'var(--board-subtitle-size)' }}
          >
            The Classic Property Trading Game
          </p>
        </div>
      </div>
    </div>
  );
}
