import type { PropertyColor, PropertyState, SpaceType } from '@/types';
import { getPropertyInfo } from '@/utils/board';
import { formatMoney } from '@/utils/format';

interface PropertySpaceProps {
  position: number;
  name: string;
  type: SpaceType;
  color?: PropertyColor;
  property?: PropertyState;
}

/**
 * Property-specific space content.
 * Shows property name, price, and type-specific icons.
 */
export function PropertySpace({ position, name, type }: PropertySpaceProps) {
  const propertyInfo = getPropertyInfo(position);

  // Get display name (shortened for display)
  const displayName = getShortName(name);

  // Render based on space type
  switch (type) {
    case 'property':
    case 'railroad':
    case 'utility':
      return (
        <div className="text-center leading-tight">
          <div
            className="font-semibold leading-tight mb-0.5"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            {displayName}
          </div>
          {propertyInfo && (
            <div
              className="text-gray-600"
              style={{ fontSize: 'var(--space-price-size)' }}
            >
              {formatMoney(propertyInfo.price)}
            </div>
          )}
          {type === 'railroad' && <div className="text-sm sm:text-lg mt-0.5">🚂</div>}
          {type === 'utility' && (
            <div className="text-sm sm:text-lg mt-0.5">{position === 12 ? '💡' : '🚰'}</div>
          )}
        </div>
      );

    case 'tax':
      return (
        <div className="text-center leading-tight">
          <div
            className="font-semibold leading-tight"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            {displayName}
          </div>
          <div className="text-sm sm:text-lg mt-0.5">💰</div>
          <div
            className="text-gray-600"
            style={{ fontSize: 'var(--space-price-size)' }}
          >
            {position === 4 ? '$200' : '$100'}
          </div>
        </div>
      );

    case 'chance':
      return (
        <div className="text-center leading-tight">
          <div
            className="font-semibold leading-tight"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            CHANCE
          </div>
          <div className="text-lg sm:text-2xl mt-0.5">❓</div>
        </div>
      );

    case 'community_chest':
      return (
        <div className="text-center leading-tight">
          <div
            className="font-semibold leading-tight whitespace-nowrap"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            COMMUNITY
          </div>
          <div
            className="font-semibold leading-tight"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            CHEST
          </div>
          <div className="text-base sm:text-xl mt-0.5">📦</div>
        </div>
      );

    default:
      return (
        <div className="text-center leading-tight">
          <div
            className="font-semibold leading-tight"
            style={{ fontSize: 'var(--space-name-size)' }}
          >
            {displayName}
          </div>
        </div>
      );
  }
}

/**
 * Shorten property names for display in small spaces.
 */
function getShortName(name: string): string {
  // Replace common words with abbreviations
  return name
    .replace('Avenue', 'Ave')
    .replace('Place', 'Pl')
    .replace('Railroad', 'RR')
    .replace('Community Chest', 'Comm.\nChest')
    .replace('Electric Company', 'Electric\nCo.')
    .replace('Water Works', 'Water\nWorks')
    .replace('Income Tax', 'Income\nTax')
    .replace('Luxury Tax', 'Luxury\nTax');
}
