import { useMemo } from 'react';
import { PropertyCard } from './PropertyCard';
import { useGameStore } from '@/store/gameStore';
import { BOARD_SPACES } from '@/utils/board';
import type { PropertyState, PropertyColor } from '@/types';

interface PropertyListProps {
  playerId: number;
  onPropertyClick?: (position: number) => void;
  showMonopolyIndicator?: boolean;
  compact?: boolean;
}

/**
 * List of properties owned by a player, grouped by color.
 */
export function PropertyList({
  playerId,
  onPropertyClick,
  showMonopolyIndicator = true,
  compact = true,
}: PropertyListProps) {
  const { gameState, hasMonopoly, getPlayerProperties } = useGameStore();

  const groupedProperties = useMemo(() => {
    if (!gameState) return [];

    const properties = getPlayerProperties(playerId);

    // Group by color
    const groups: Map<
      string,
      { color: string; properties: PropertyState[]; hasMonopoly: boolean }
    > = new Map();

    // Define color order
    const colorOrder: string[] = [
      'brown',
      'lightblue',
      'magenta',
      'orange',
      'red',
      'yellow',
      'green',
      'blue',
      'railroad',
      'utility',
    ];

    for (const prop of properties) {
      const space = BOARD_SPACES[prop.position];
      const color = space?.color || space?.type || 'other';

      if (!groups.has(color)) {
        groups.set(color, {
          color,
          properties: [],
          hasMonopoly: hasMonopoly(playerId, color as PropertyColor),
        });
      }

      groups.get(color)!.properties.push(prop);
    }

    // Sort by color order
    return colorOrder.filter((color) => groups.has(color)).map((color) => groups.get(color)!);
  }, [gameState, playerId, getPlayerProperties, hasMonopoly]);

  if (!gameState || groupedProperties.length === 0) {
    return <div className="text-gray-500 text-sm text-center py-4">No properties owned</div>;
  }

  return (
    <div className="space-y-4">
      {groupedProperties.map(({ color, properties, hasMonopoly: monopoly }) => (
        <div key={color}>
          {/* Color group header */}
          <div className="flex items-center gap-2 mb-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: getColorValue(color) }} />
            <span className="text-sm font-medium capitalize">{color}</span>
            {showMonopolyIndicator && monopoly && (
              <span className="px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded-full">
                Monopoly
              </span>
            )}
            <span className="text-xs text-gray-400">
              ({properties.length}/{getGroupSize(color)})
            </span>
          </div>

          {/* Properties in group */}
          <div className={compact ? 'space-y-1' : 'grid gap-3 grid-cols-1 sm:grid-cols-2'}>
            {properties.map((prop) => (
              <PropertyCard
                key={prop.position}
                position={prop.position}
                property={prop}
                compact={compact}
                onClick={() => onPropertyClick?.(prop.position)}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Summary */}
      <div className="pt-3 border-t border-gray-200 text-sm text-gray-600">
        <div className="flex justify-between">
          <span>Total Properties</span>
          <span className="font-medium">
            {groupedProperties.reduce((sum, g) => sum + g.properties.length, 0)}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Monopolies</span>
          <span className="font-medium text-green-600">
            {groupedProperties.filter((g) => g.hasMonopoly).length}
          </span>
        </div>
      </div>
    </div>
  );
}

function getColorValue(color: string): string {
  const colors: Record<string, string> = {
    brown: '#8B4513',
    lightblue: '#87CEEB',
    magenta: '#FF00FF',
    orange: '#FFA500',
    red: '#FF0000',
    yellow: '#FFFF00',
    green: '#008000',
    blue: '#0000FF',
    railroad: '#333333',
    utility: '#666666',
  };
  return colors[color] || '#999999';
}

function getGroupSize(color: string): number {
  if (color === 'railroad') return 4;
  if (color === 'utility') return 2;
  if (color === 'brown' || color === 'blue') return 2;
  return 3;
}

/**
 * Compact property ownership summary for player cards.
 */
interface PropertySummaryProps {
  playerId: number;
}

export function PropertySummary({ playerId }: PropertySummaryProps) {
  const { gameState, getPlayerProperties, hasMonopoly } = useGameStore();

  const summary = useMemo(() => {
    if (!gameState) return { total: 0, monopolies: 0, houses: 0, hotels: 0 };

    const properties = getPlayerProperties(playerId);
    const colors: PropertyColor[] = [
      'brown',
      'lightblue',
      'magenta',
      'orange',
      'red',
      'yellow',
      'green',
      'blue',
    ];

    return {
      total: properties.length,
      monopolies: colors.filter((c) => hasMonopoly(playerId, c)).length,
      houses: properties.reduce((sum, p) => sum + (p.houses < 5 ? p.houses : 0), 0),
      hotels: properties.filter((p) => p.houses === 5).length,
    };
  }, [gameState, playerId, getPlayerProperties, hasMonopoly]);

  return (
    <div className="flex items-center gap-3 text-xs text-gray-600">
      <span title="Properties">{summary.total} props</span>
      {summary.monopolies > 0 && (
        <span className="text-green-600" title="Monopolies">
          {summary.monopolies} mono
        </span>
      )}
      {summary.houses > 0 && (
        <span className="text-green-700" title="Houses">
          {summary.houses} houses
        </span>
      )}
      {summary.hotels > 0 && (
        <span className="text-red-600" title="Hotels">
          {summary.hotels} hotels
        </span>
      )}
    </div>
  );
}
