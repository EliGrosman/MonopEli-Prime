import type { PlayerState, PropertyState } from '@/types';
import { BOARD_SPACES, PROPERTY_INFO } from '@/utils/board';

interface PlayerCardProps {
  player: PlayerState;
  isCurrentTurn: boolean;
  isCurrentUser: boolean;
  properties: Record<number, PropertyState>;
  onSelect?: (playerId: number) => void;
}

/**
 * Individual player card showing player info.
 */
export function PlayerCard({
  player,
  isCurrentTurn,
  isCurrentUser,
  properties,
  onSelect,
}: PlayerCardProps) {
  // Get properties owned by this player
  const ownedProperties = Object.values(properties).filter((p) => p.owner === player.id);

  // Group properties by color
  const propertyGroups = ownedProperties.reduce(
    (groups, prop) => {
      const propInfo = PROPERTY_INFO[prop.position];
      const color = propInfo?.color || 'other';
      if (!groups[color]) {
        groups[color] = [];
      }
      groups[color].push(prop);
      return groups;
    },
    {} as Record<string, PropertyState[]>
  );

  // Calculate total property value
  const totalPropertyValue = ownedProperties.reduce((sum, prop) => {
    const propInfo = PROPERTY_INFO[prop.position];
    return sum + (propInfo?.price || 0);
  }, 0);

  // Calculate net worth (money + property value + houses)
  const houseValue = ownedProperties.reduce((sum, prop) => {
    const propInfo = PROPERTY_INFO[prop.position];
    const houses = prop.houses < 5 ? prop.houses : 4;
    const hotels = prop.houses === 5 ? 1 : 0;
    return sum + houses * (propInfo?.buildCost || 0) + hotels * (propInfo?.buildCost || 0);
  }, 0);

  const netWorth = player.money + totalPropertyValue + houseValue;

  return (
    <div
      className={`
        rounded-lg border-2 transition-all duration-200 cursor-pointer
        ${isCurrentTurn ? 'border-blue-500 shadow-lg bg-blue-50' : 'border-gray-200 bg-white'}
        ${isCurrentUser ? 'ring-2 ring-green-400' : ''}
        ${player.bankrupt ? 'opacity-50 grayscale' : ''}
        hover:shadow-md
      `}
      onClick={() => onSelect?.(player.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect?.(player.id);
        }
      }}
      aria-label={`${player.name}${isCurrentUser ? ' (You)' : ''}${isCurrentTurn ? ', current turn' : ''}`}
    >
      {/* Header with color and name */}
      <div className="h-2 rounded-t-md" style={{ backgroundColor: player.color }} />

      <div className="p-3">
        {/* Name and status */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full border-2 border-white shadow-sm"
              style={{ backgroundColor: player.color }}
            />
            <span className="font-semibold text-gray-900">
              {player.name}
              {isCurrentUser && <span className="text-green-600 text-sm ml-1">(You)</span>}
            </span>
          </div>

          {/* Status badges */}
          <div className="flex gap-1">
            {isCurrentTurn && !player.bankrupt && (
              <span className="px-2 py-0.5 text-xs bg-blue-500 text-white rounded-full">Turn</span>
            )}
            {player.isAi && (
              <span className="px-2 py-0.5 text-xs bg-purple-500 text-white rounded-full">AI</span>
            )}
            {player.bankrupt && (
              <span className="px-2 py-0.5 text-xs bg-red-500 text-white rounded-full">
                Bankrupt
              </span>
            )}
            {player.inJail && !player.bankrupt && (
              <span className="px-2 py-0.5 text-xs bg-orange-500 text-white rounded-full">
                In Jail
              </span>
            )}
          </div>
        </div>

        {/* Money */}
        <div className="flex justify-between items-center mb-2 text-sm">
          <span className="text-gray-600">Cash</span>
          <span className="font-mono font-semibold text-green-600">
            ${player.money.toLocaleString()}
          </span>
        </div>

        {/* Net worth */}
        <div className="flex justify-between items-center mb-2 text-sm">
          <span className="text-gray-600">Net Worth</span>
          <span className="font-mono text-gray-800">${netWorth.toLocaleString()}</span>
        </div>

        {/* Property indicators */}
        {Object.keys(propertyGroups).length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-100">
            <div className="text-xs text-gray-500 mb-1">Properties</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(propertyGroups).map(([color, props]) => (
                <div
                  key={color}
                  className="flex items-center gap-0.5 px-1.5 py-0.5 rounded-sm text-xs"
                  style={{
                    backgroundColor: getColorValue(color),
                    color: getTextColor(color),
                  }}
                  title={`${props.length} ${color} properties`}
                >
                  <span>{props.length}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Jail cards */}
        {player.jailCards > 0 && (
          <div className="mt-2 text-xs text-gray-600">Get Out of Jail Free: {player.jailCards}</div>
        )}

        {/* Position */}
        {!player.bankrupt && (
          <div className="mt-2 text-xs text-gray-500">
            Position:{' '}
            {BOARD_SPACES.find((s) => s.position === player.position)?.name ||
              `Space ${player.position}`}
          </div>
        )}
      </div>
    </div>
  );
}

// Color value mappings
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
    other: '#999999',
  };
  return colors[color] || colors.other;
}

function getTextColor(color: string): string {
  const lightColors = ['lightblue', 'yellow'];
  return lightColors.includes(color) ? '#000000' : '#FFFFFF';
}
