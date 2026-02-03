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
          <div className="font-semibold text-[8px] leading-tight mb-1">{displayName}</div>
          {propertyInfo && (
            <div className="text-[7px] text-gray-600">{formatMoney(propertyInfo.price)}</div>
          )}
          {type === 'railroad' && <div className="text-lg mt-0.5">🚂</div>}
          {type === 'utility' && (
            <div className="text-lg mt-0.5">{position === 12 ? '💡' : '🚰'}</div>
          )}
        </div>
      );

    case 'tax':
      return (
        <div className="text-center leading-tight">
          <div className="font-semibold text-[8px] leading-tight">{displayName}</div>
          <div className="text-lg mt-1">💰</div>
          <div className="text-[7px] text-gray-600">{position === 4 ? '$200' : '$100'}</div>
        </div>
      );

    case 'chance':
      return (
        <div className="text-center leading-tight">
          <div className="font-semibold text-[8px] leading-tight">CHANCE</div>
          <div className="text-2xl mt-1">❓</div>
        </div>
      );

    case 'community_chest':
      return (
        <div className="text-center leading-tight">
          <div className="font-semibold text-[8px] leading-tight whitespace-nowrap">
            COMMUNITY
          </div>
          <div className="font-semibold text-[8px] leading-tight">CHEST</div>
          <div className="text-xl mt-1">📦</div>
        </div>
      );

    default:
      return (
        <div className="text-center leading-tight">
          <div className="font-semibold text-[8px] leading-tight">{displayName}</div>
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
