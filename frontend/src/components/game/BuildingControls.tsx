import { useMemo } from 'react';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useActions } from '@/hooks/useActions';
import { PROPERTY_INFO, getColorGroupPositions } from '@/utils/board';
import type { ClientActionMessage, PropertyColor } from '@/types';

interface BuildingControlsProps {
  send: (message: ClientActionMessage) => void;
}

/**
 * Building controls for buying houses/hotels on properties.
 */
export function BuildingControls({ send }: BuildingControlsProps) {
  const { gameState, hasMonopoly, getPlayerProperties } = useGameStore();
  const { playerId } = useSessionStore();
  const {
    buildHouse,
    buildHotel,
    sellHouse,
    sellHotel,
    mortgageProperty,
    unmortgageProperty,
    isActionPending,
  } = useActions({ send });

  // Get buildable properties (monopolies only)
  const buildableGroups = useMemo(() => {
    if (!gameState || playerId === null) return [];

    const groups: Array<{
      color: PropertyColor;
      properties: Array<{
        position: number;
        name: string;
        houses: number;
        mortgaged: boolean;
        buildCost: number;
        canBuild: boolean;
        canSell: boolean;
      }>;
    }> = [];

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

    for (const color of colors) {
      if (hasMonopoly(playerId, color)) {
        const positions = getColorGroupPositions(color);
        const props = positions.map((pos) => {
          const state = gameState.properties[pos];
          const propInfo = PROPERTY_INFO[pos];
          const houses = state?.houses ?? 0;

          // Check even building rule
          const groupHouses = positions.map((p) => gameState.properties[p]?.houses ?? 0);
          const minHouses = Math.min(...groupHouses);
          const maxHouses = Math.max(...groupHouses);

          return {
            position: pos,
            name: propInfo?.name || `Property ${pos}`,
            houses,
            mortgaged: state?.mortgaged ?? false,
            buildCost: propInfo?.buildCost || 0,
            canBuild:
              houses < 5 &&
              !state?.mortgaged &&
              houses <= minHouses &&
              (gameState.housesRemaining > 0 || houses === 4),
            canSell: houses > 0 && houses >= maxHouses,
          };
        });

        // Only include if at least one property can be built on or sold
        if (props.some((p) => p.canBuild || p.canSell || p.mortgaged)) {
          groups.push({ color, properties: props });
        }
      }
    }

    return groups;
  }, [gameState, playerId, hasMonopoly]);

  // Get mortgageable properties (non-monopoly or no houses)
  const mortgageableProperties = useMemo(() => {
    if (!gameState || playerId === null) return [];

    const ownedProps = getPlayerProperties(playerId);
    return ownedProps
      .filter((p) => {
        const propInfo = PROPERTY_INFO[p.position];
        // Can mortgage if no houses and not already mortgaged
        return p.houses === 0 && !p.mortgaged && propInfo;
      })
      .map((p) => {
        const propInfo = PROPERTY_INFO[p.position];
        return {
          position: p.position,
          name: propInfo?.name || `Property ${p.position}`,
          mortgageValue: propInfo?.mortgageValue || 0,
        };
      });
  }, [gameState, playerId, getPlayerProperties]);

  // Get unmortgageable properties
  const unmortgageableProperties = useMemo(() => {
    if (!gameState || playerId === null) return [];

    const ownedProps = getPlayerProperties(playerId);
    const player = gameState.players.find((p) => p.id === playerId);

    return ownedProps
      .filter((p) => {
        const propInfo = PROPERTY_INFO[p.position];
        const unmortgageCost = Math.floor((propInfo?.mortgageValue || 0) * 1.1);
        // Can unmortgage if mortgaged and have enough money
        return p.mortgaged && (player?.money ?? 0) >= unmortgageCost;
      })
      .map((p) => {
        const propInfo = PROPERTY_INFO[p.position];
        return {
          position: p.position,
          name: propInfo?.name || `Property ${p.position}`,
          unmortgageCost: Math.floor((propInfo?.mortgageValue || 0) * 1.1),
        };
      });
  }, [gameState, playerId, getPlayerProperties]);

  if (
    buildableGroups.length === 0 &&
    mortgageableProperties.length === 0 &&
    unmortgageableProperties.length === 0
  ) {
    return null;
  }

  return (
    <div className="space-y-3">
      {/* Building section */}
      {buildableGroups.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">Build</h3>
          <div className="space-y-2">
            {buildableGroups.map(({ color, properties }) => (
              <div key={color} className="p-2 bg-gray-50 rounded">
                <div
                  className="text-xs font-medium mb-1 capitalize"
                  style={{ color: getColorValue(color) }}
                >
                  {color} Properties
                </div>
                <div className="space-y-1">
                  {properties.map((prop) => (
                    <div key={prop.position} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span
                          className={
                            prop.mortgaged ? 'text-gray-400 line-through' : 'text-gray-900'
                          }
                        >
                          {prop.name}
                        </span>
                        {prop.houses > 0 && (
                          <span className="flex gap-0.5">
                            {prop.houses === 5 ? (
                              <span className="w-2 h-2 bg-red-500 rounded-sm" title="Hotel" />
                            ) : (
                              Array.from({ length: prop.houses }).map((_, i) => (
                                <span
                                  key={i}
                                  className="w-2 h-2 bg-green-500 rounded-sm"
                                  title="House"
                                />
                              ))
                            )}
                          </span>
                        )}
                      </div>
                      <div className="flex gap-1">
                        {prop.canSell && (
                          <button
                            onClick={() =>
                              prop.houses === 5
                                ? sellHotel(prop.position)
                                : sellHouse(prop.position)
                            }
                            disabled={isActionPending}
                            className="px-1.5 py-0.5 bg-red-100 text-red-700 rounded text-xs hover:bg-red-200 disabled:opacity-50"
                            title="Sell"
                          >
                            -
                          </button>
                        )}
                        {prop.canBuild && (
                          <button
                            onClick={() =>
                              prop.houses === 4
                                ? buildHotel(prop.position)
                                : buildHouse(prop.position)
                            }
                            disabled={isActionPending}
                            className="px-1.5 py-0.5 bg-green-100 text-green-700 rounded text-xs hover:bg-green-200 disabled:opacity-50"
                            title={`Build ($${prop.buildCost})`}
                          >
                            +
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mortgage section */}
      {(mortgageableProperties.length > 0 || unmortgageableProperties.length > 0) && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">Mortgage</h3>
          <div className="space-y-1">
            {mortgageableProperties.map((prop) => (
              <div
                key={prop.position}
                className="flex items-center justify-between text-xs p-1 bg-gray-50 rounded"
              >
                <span className="text-gray-900">{prop.name}</span>
                <button
                  onClick={() => mortgageProperty(prop.position)}
                  disabled={isActionPending}
                  className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 disabled:opacity-50"
                >
                  Mortgage (+${prop.mortgageValue})
                </button>
              </div>
            ))}
            {unmortgageableProperties.map((prop) => (
              <div
                key={prop.position}
                className="flex items-center justify-between text-xs p-1 bg-gray-50 rounded"
              >
                <span className="text-gray-400">{prop.name} (Mortgaged)</span>
                <button
                  onClick={() => unmortgageProperty(prop.position)}
                  disabled={isActionPending}
                  className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50"
                >
                  Unmortgage (-${prop.unmortgageCost})
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function getColorValue(color: string): string {
  const colors: Record<string, string> = {
    brown: '#8B4513',
    lightblue: '#0EA5E9',
    magenta: '#DB2777',
    orange: '#EA580C',
    red: '#DC2626',
    yellow: '#CA8A04',
    green: '#16A34A',
    blue: '#2563EB',
  };
  return colors[color] || '#666666';
}
