import type { ReactNode } from 'react';
import type { PropertyColor, PropertyState, SpaceType } from '@/types';
import { PropertySpace } from './PropertySpace';
import { CornerSpace } from './CornerSpace';
import { HouseIndicator } from './HouseIndicator';
import { getColorHex, isCornerSpace } from '@/utils/board';

export type SpaceOrientation = 'top' | 'bottom' | 'left' | 'right';

interface BoardSpaceProps {
  position: number;
  name: string;
  type: SpaceType;
  color?: PropertyColor;
  property?: PropertyState;
  isHighlighted?: boolean;
  onClick?: () => void;
  children?: ReactNode;
  orientation: SpaceOrientation;
}

/**
 * Individual board space component.
 * Renders different content based on space type.
 */
export function BoardSpace({
  position,
  name,
  type,
  color,
  property,
  isHighlighted = false,
  onClick,
  children,
  orientation,
}: BoardSpaceProps) {
  const isCorner = isCornerSpace(position);

  // Determine rotation based on orientation
  const getRotation = (): string => {
    switch (orientation) {
      case 'left':
        return 'rotate-90';
      case 'right':
        return '-rotate-90';
      case 'top':
        return 'rotate-180';
      default:
        return '';
    }
  };

  const baseClasses = `
    board-space relative flex flex-col items-center justify-end
    bg-space-bg border border-board-border
    text-xs text-center overflow-hidden
    cursor-pointer transition-transform duration-150
    ${isHighlighted ? 'animate-pulse-highlight ring-2 ring-blue-400' : ''}
    ${isCorner ? 'w-full h-full' : 'w-full h-full'}
    hover:scale-105 hover:z-10
  `.trim();

  if (isCorner) {
    return (
      <div
        className={baseClasses}
        onClick={onClick}
        data-testid={`space-${position}`}
        data-position={position}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClick?.();
          }
        }}
        aria-label={name}
      >
        <CornerSpace position={position} name={name} />
        {/* Player tokens */}
        <div className="absolute bottom-1 flex flex-wrap gap-0.5 justify-center">{children}</div>
      </div>
    );
  }

  return (
    <div
      className={baseClasses}
      onClick={onClick}
      data-testid={`space-${position}`}
      data-position={position}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick?.();
        }
      }}
      aria-label={`${name}${property?.owner !== undefined && property?.owner !== null ? ', owned' : ''}`}
    >
      {/* Color band for properties */}
      {color && (
        <div
          className="absolute top-0 left-0 right-0 h-5"
          style={{ backgroundColor: getColorHex(color) }}
        >
          {/* House indicators */}
          {property && property.houses > 0 && (
            <HouseIndicator houses={property.houses} />
          )}
        </div>
      )}

      {/* Space content */}
      <div className={`flex-1 flex flex-col items-center justify-center p-1 ${getRotation()}`}>
        <PropertySpace
          position={position}
          name={name}
          type={type}
          color={color}
          property={property}
        />
      </div>

      {/* Ownership indicator */}
      {property?.owner !== undefined && property?.owner !== null && (
        <div
          className="absolute bottom-0 left-0 right-0 h-1"
          style={{ backgroundColor: `var(--player-${property.owner}-color, #666)` }}
        />
      )}

      {/* Mortgage indicator */}
      {property?.mortgaged && (
        <div className="absolute inset-0 bg-gray-500 bg-opacity-50 flex items-center justify-center">
          <span className="text-white text-xs font-bold transform -rotate-45">MORTGAGED</span>
        </div>
      )}

      {/* Player tokens */}
      <div className="absolute bottom-1 flex flex-wrap gap-0.5 justify-center max-w-full">
        {children}
      </div>
    </div>
  );
}
