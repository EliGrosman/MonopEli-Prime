import { Modal } from '@/components/common';
import { PropertyCard } from './PropertyCard';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useActions } from '@/hooks/useActions';
import { BOARD_SPACES, PROPERTY_INFO, getColorGroupPositions } from '@/utils/board';
import type { ClientActionMessage, PropertyColor } from '@/types';

interface PropertyModalProps {
  position: number | null;
  isOpen: boolean;
  onClose: () => void;
  send: (message: ClientActionMessage) => void;
}

/**
 * Modal displaying full property information with actions.
 */
export function PropertyModal({ position, isOpen, onClose, send }: PropertyModalProps) {
  const { gameState, hasMonopoly } = useGameStore();
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

  if (position === null || !gameState) {
    return null;
  }

  const space = BOARD_SPACES[position];
  const info = PROPERTY_INFO[position];
  const property = gameState.properties[position];

  if (!space || !info) {
    return null;
  }

  const isOwned = property?.owner !== null && property?.owner !== undefined;
  const isOwnedByMe = property?.owner === playerId;
  const owner = isOwned ? gameState.players.find((p) => p.id === property.owner) : null;
  const hasMonopolyOnColor =
    playerId !== null && space.color ? hasMonopoly(playerId, space.color) : false;

  // Get other properties in the same color group
  const colorGroupPositions = space.color
    ? getColorGroupPositions(space.color as PropertyColor)
    : [];
  const colorGroupProperties = colorGroupPositions.map((pos) => ({
    position: pos,
    space: BOARD_SPACES[pos],
    property: gameState.properties[pos],
  }));

  // Check if can build/sell (even building rule)
  const houses = property?.houses ?? 0;
  const groupHouses = colorGroupPositions.map((p) => gameState.properties[p]?.houses ?? 0);
  const minHouses = Math.min(...groupHouses);
  const maxHouses = Math.max(...groupHouses);
  const canBuild =
    isOwnedByMe &&
    hasMonopolyOnColor &&
    !property?.mortgaged &&
    houses < 5 &&
    houses <= minHouses &&
    (gameState.housesRemaining > 0 || houses === 4);
  const canSell = isOwnedByMe && houses > 0 && houses >= maxHouses;
  const canMortgage = isOwnedByMe && !property?.mortgaged && houses === 0;
  const canUnmortgage =
    isOwnedByMe &&
    property?.mortgaged &&
    (owner?.money ?? 0) >= Math.floor(info.mortgageValue * 1.1);

  const handleBuild = () => {
    if (houses === 4) {
      buildHotel(position);
    } else {
      buildHouse(position);
    }
  };

  const handleSell = () => {
    if (houses === 5) {
      sellHotel(position);
    } else {
      sellHouse(position);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={space.name} size="lg">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Property card */}
        <div className="md:w-64 flex-shrink-0">
          <PropertyCard position={position} property={property} />
        </div>

        {/* Details */}
        <div className="flex-1 space-y-4">
          {/* Owner info */}
          <div className="p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-700 mb-2">Ownership</h4>
            {isOwned && owner ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: owner.color }} />
                <span className="font-medium">
                  {owner.name}
                  {owner.id === playerId && ' (You)'}
                </span>
              </div>
            ) : (
              <span className="text-gray-500">Not owned</span>
            )}
          </div>

          {/* Monopoly status */}
          {space.color && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-700 mb-2">
                {space.color.charAt(0).toUpperCase() + space.color.slice(1)} Group
              </h4>
              <div className="space-y-1">
                {colorGroupProperties.map((p) => {
                  const propOwner =
                    p.property?.owner !== null
                      ? gameState.players.find((pl) => pl.id === p.property?.owner)
                      : null;
                  return (
                    <div
                      key={p.position}
                      className={`flex items-center justify-between text-sm ${
                        p.position === position ? 'font-semibold' : ''
                      }`}
                    >
                      <span>{p.space?.name}</span>
                      {propOwner ? (
                        <div className="flex items-center gap-1">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: propOwner.color }}
                          />
                          <span className="text-gray-600">{propOwner.name}</span>
                        </div>
                      ) : (
                        <span className="text-gray-400">Unowned</span>
                      )}
                    </div>
                  );
                })}
              </div>
              {hasMonopolyOnColor && (
                <div className="mt-2 text-green-600 text-sm font-medium">
                  You have a monopoly! Double rent applies.
                </div>
              )}
            </div>
          )}

          {/* Building status */}
          {space.type === 'property' && isOwned && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-700 mb-2">Development</h4>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm text-gray-600">Current:</span>
                {houses === 0 && <span className="text-gray-500">No buildings</span>}
                {houses > 0 && houses < 5 && (
                  <div className="flex gap-1">
                    {Array.from({ length: houses }).map((_, i) => (
                      <span key={i} className="w-4 h-4 bg-green-500 rounded-sm" title="House" />
                    ))}
                  </div>
                )}
                {houses === 5 && <span className="w-5 h-5 bg-red-500 rounded-sm" title="Hotel" />}
              </div>
              {property?.mortgaged && (
                <div className="text-red-500 text-sm font-medium">This property is mortgaged</div>
              )}
            </div>
          )}

          {/* Actions */}
          {isOwnedByMe && (
            <div className="flex flex-wrap gap-2 pt-2">
              {canBuild && (
                <button
                  onClick={handleBuild}
                  disabled={isActionPending}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 transition-colors"
                >
                  {houses === 4
                    ? `Build Hotel ($${info.buildCost})`
                    : `Build House ($${info.buildCost})`}
                </button>
              )}
              {canSell && (
                <button
                  onClick={handleSell}
                  disabled={isActionPending}
                  className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 transition-colors"
                >
                  {houses === 5
                    ? `Sell Hotel (+$${info.buildCost / 2})`
                    : `Sell House (+$${info.buildCost / 2})`}
                </button>
              )}
              {canMortgage && (
                <button
                  onClick={() => mortgageProperty(position)}
                  disabled={isActionPending}
                  className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 disabled:opacity-50 transition-colors"
                >
                  Mortgage (+${info.mortgageValue})
                </button>
              )}
              {canUnmortgage && (
                <button
                  onClick={() => unmortgageProperty(position)}
                  disabled={isActionPending}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
                >
                  Unmortgage (-${Math.floor(info.mortgageValue * 1.1)})
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}
