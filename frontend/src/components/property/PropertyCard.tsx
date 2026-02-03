import type { PropertyState } from '@/types';
import { BOARD_SPACES, PROPERTY_INFO } from '@/utils/board';

interface PropertyCardProps {
  position: number;
  property?: PropertyState;
  compact?: boolean;
  showActions?: boolean;
  onBuild?: () => void;
  onSell?: () => void;
  onMortgage?: () => void;
  onUnmortgage?: () => void;
  onClick?: () => void;
}

/**
 * Property card displaying property information.
 *
 * Shows:
 * - Property name and color band
 * - Current rent (based on houses)
 * - Rent schedule
 * - House/hotel cost
 * - Mortgage value
 * - Owner (if owned)
 * - Houses built
 */
export function PropertyCard({
  position,
  property,
  compact = false,
  showActions = false,
  onBuild,
  onSell,
  onMortgage,
  onUnmortgage,
  onClick,
}: PropertyCardProps) {
  const space = BOARD_SPACES[position];
  const info = PROPERTY_INFO[position];

  if (!space || !info) {
    return null;
  }

  const houses = property?.houses ?? 0;
  const isMortgaged = property?.mortgaged ?? false;
  const isOwned = property?.owner !== null && property?.owner !== undefined;

  // Color mapping
  const colorClasses = getColorClasses(space.color);

  if (compact) {
    return (
      <div
        className={`
          flex items-center gap-2 p-2 rounded border cursor-pointer
          ${isMortgaged ? 'opacity-60 bg-gray-100' : 'bg-white'}
          hover:shadow-md transition-shadow
        `}
        onClick={onClick}
        role={onClick ? 'button' : undefined}
        tabIndex={onClick ? 0 : undefined}
      >
        <div className={`w-4 h-4 rounded ${colorClasses.bg}`} />
        <span className="flex-1 text-sm font-medium truncate">{space.name}</span>
        {houses > 0 && (
          <div className="flex gap-0.5">
            {houses === 5 ? (
              <span className="w-3 h-3 bg-red-500 rounded-sm" title="Hotel" />
            ) : (
              Array.from({ length: houses }).map((_, i) => (
                <span key={i} className="w-2 h-2 bg-green-500 rounded-sm" title="House" />
              ))
            )}
          </div>
        )}
        {isMortgaged && (
          <span className="text-xs text-red-500 font-medium">M</span>
        )}
      </div>
    );
  }

  return (
    <div
      className={`
        bg-white rounded-lg shadow-md overflow-hidden border-2
        ${isMortgaged ? 'border-red-300 opacity-75' : 'border-gray-200'}
        ${onClick ? 'cursor-pointer hover:shadow-lg transition-shadow' : ''}
      `}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {/* Color band */}
      <div className={`h-6 ${colorClasses.bg}`} />

      {/* Property name */}
      <div className="px-4 py-2 text-center border-b">
        <h3 className="font-bold text-gray-900">{space.name}</h3>
        {isMortgaged && (
          <span className="text-xs text-red-500 font-semibold">MORTGAGED</span>
        )}
      </div>

      {/* Rent schedule */}
      <div className="px-4 py-3 text-sm">
        {space.type === 'property' && (
          <>
            <div className={`flex justify-between ${houses === 0 ? 'font-semibold' : ''}`}>
              <span>Rent</span>
              <span>${info.rent[0]}</span>
            </div>
            <div className={`flex justify-between ${houses === 1 ? 'font-semibold bg-yellow-100 -mx-2 px-2' : ''}`}>
              <span>With 1 House</span>
              <span>${info.rent[1]}</span>
            </div>
            <div className={`flex justify-between ${houses === 2 ? 'font-semibold bg-yellow-100 -mx-2 px-2' : ''}`}>
              <span>With 2 Houses</span>
              <span>${info.rent[2]}</span>
            </div>
            <div className={`flex justify-between ${houses === 3 ? 'font-semibold bg-yellow-100 -mx-2 px-2' : ''}`}>
              <span>With 3 Houses</span>
              <span>${info.rent[3]}</span>
            </div>
            <div className={`flex justify-between ${houses === 4 ? 'font-semibold bg-yellow-100 -mx-2 px-2' : ''}`}>
              <span>With 4 Houses</span>
              <span>${info.rent[4]}</span>
            </div>
            <div className={`flex justify-between ${houses === 5 ? 'font-semibold bg-yellow-100 -mx-2 px-2' : ''}`}>
              <span>With Hotel</span>
              <span>${info.rent[5]}</span>
            </div>
            <div className="border-t mt-2 pt-2 flex justify-between">
              <span>House Cost</span>
              <span>${info.buildCost}</span>
            </div>
          </>
        )}

        {space.type === 'railroad' && (
          <>
            <div className="flex justify-between">
              <span>Rent (1 RR)</span>
              <span>${info.rent[0]}</span>
            </div>
            <div className="flex justify-between">
              <span>Rent (2 RR)</span>
              <span>${info.rent[1]}</span>
            </div>
            <div className="flex justify-between">
              <span>Rent (3 RR)</span>
              <span>${info.rent[2]}</span>
            </div>
            <div className="flex justify-between">
              <span>Rent (4 RR)</span>
              <span>${info.rent[3]}</span>
            </div>
          </>
        )}

        {space.type === 'utility' && (
          <div className="text-center text-gray-600">
            <p>If one Utility is owned, rent is 4x dice roll.</p>
            <p className="mt-1">If both Utilities are owned, rent is 10x dice roll.</p>
          </div>
        )}

        {/* Mortgage value */}
        <div className="border-t mt-2 pt-2 flex justify-between">
          <span>Mortgage Value</span>
          <span>${info.mortgageValue}</span>
        </div>

        {/* Price */}
        <div className="flex justify-between font-semibold mt-1">
          <span>Price</span>
          <span>${info.price}</span>
        </div>
      </div>

      {/* Actions */}
      {showActions && isOwned && (
        <div className="px-4 pb-3 flex flex-wrap gap-2">
          {!isMortgaged && onBuild && houses < 5 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onBuild();
              }}
              className="px-2 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600"
            >
              Build
            </button>
          )}
          {!isMortgaged && onSell && houses > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSell();
              }}
              className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600"
            >
              Sell
            </button>
          )}
          {!isMortgaged && onMortgage && houses === 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onMortgage();
              }}
              className="px-2 py-1 text-xs bg-yellow-500 text-white rounded hover:bg-yellow-600"
            >
              Mortgage
            </button>
          )}
          {isMortgaged && onUnmortgage && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onUnmortgage();
              }}
              className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Unmortgage
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function getColorClasses(color?: string): { bg: string; text: string } {
  const colors: Record<string, { bg: string; text: string }> = {
    brown: { bg: 'bg-property-brown', text: 'text-white' },
    lightblue: { bg: 'bg-property-lightblue', text: 'text-gray-900' },
    magenta: { bg: 'bg-property-magenta', text: 'text-white' },
    orange: { bg: 'bg-property-orange', text: 'text-white' },
    red: { bg: 'bg-property-red', text: 'text-white' },
    yellow: { bg: 'bg-property-yellow', text: 'text-gray-900' },
    green: { bg: 'bg-property-green', text: 'text-white' },
    blue: { bg: 'bg-property-blue', text: 'text-white' },
    railroad: { bg: 'bg-gray-800', text: 'text-white' },
    utility: { bg: 'bg-gray-500', text: 'text-white' },
  };
  return colors[color || ''] || { bg: 'bg-gray-300', text: 'text-gray-900' };
}
