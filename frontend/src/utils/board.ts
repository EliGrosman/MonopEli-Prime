import type { BoardSpace, PropertyInfo, PropertyColor } from '@/types';

/**
 * Board layout positions for CSS Grid rendering.
 * Each array contains the positions for that side of the board.
 */
export const BOARD_LAYOUT = {
  // Bottom row: GO (0) to Jail (10) - right to left
  bottom: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
  // Left column: Connecticut (9) up to St. Charles (11) - bottom to top
  left: [11, 12, 13, 14, 15, 16, 17, 18, 19],
  // Top row: Free Parking (20) to Go To Jail (30) - left to right
  top: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
  // Right column: Pacific (31) down to Park Place (39) - top to bottom
  right: [31, 32, 33, 34, 35, 36, 37, 38, 39],
};

/**
 * All 40 board spaces with their information.
 */
export const BOARD_SPACES: BoardSpace[] = [
  // Bottom row (0-10)
  { position: 0, name: 'GO', type: 'corner' },
  { position: 1, name: 'Mediterranean Avenue', type: 'property', color: 'brown' },
  { position: 2, name: 'Community Chest', type: 'community_chest' },
  { position: 3, name: 'Baltic Avenue', type: 'property', color: 'brown' },
  { position: 4, name: 'Income Tax', type: 'tax' },
  { position: 5, name: 'Reading Railroad', type: 'railroad' },
  { position: 6, name: 'Oriental Avenue', type: 'property', color: 'lightblue' },
  { position: 7, name: 'Chance', type: 'chance' },
  { position: 8, name: 'Vermont Avenue', type: 'property', color: 'lightblue' },
  { position: 9, name: 'Connecticut Avenue', type: 'property', color: 'lightblue' },
  { position: 10, name: 'Jail / Just Visiting', type: 'corner' },

  // Left column (11-19)
  { position: 11, name: 'St. Charles Place', type: 'property', color: 'magenta' },
  { position: 12, name: 'Electric Company', type: 'utility' },
  { position: 13, name: 'States Avenue', type: 'property', color: 'magenta' },
  { position: 14, name: 'Virginia Avenue', type: 'property', color: 'magenta' },
  { position: 15, name: 'Pennsylvania Railroad', type: 'railroad' },
  { position: 16, name: 'St. James Place', type: 'property', color: 'orange' },
  { position: 17, name: 'Community Chest', type: 'community_chest' },
  { position: 18, name: 'Tennessee Avenue', type: 'property', color: 'orange' },
  { position: 19, name: 'New York Avenue', type: 'property', color: 'orange' },

  // Top row (20-30)
  { position: 20, name: 'Free Parking', type: 'corner' },
  { position: 21, name: 'Kentucky Avenue', type: 'property', color: 'red' },
  { position: 22, name: 'Chance', type: 'chance' },
  { position: 23, name: 'Indiana Avenue', type: 'property', color: 'red' },
  { position: 24, name: 'Illinois Avenue', type: 'property', color: 'red' },
  { position: 25, name: 'B&O Railroad', type: 'railroad' },
  { position: 26, name: 'Atlantic Avenue', type: 'property', color: 'yellow' },
  { position: 27, name: 'Ventnor Avenue', type: 'property', color: 'yellow' },
  { position: 28, name: 'Water Works', type: 'utility' },
  { position: 29, name: 'Marvin Gardens', type: 'property', color: 'yellow' },
  { position: 30, name: 'Go To Jail', type: 'corner' },

  // Right column (31-39)
  { position: 31, name: 'Pacific Avenue', type: 'property', color: 'green' },
  { position: 32, name: 'North Carolina Avenue', type: 'property', color: 'green' },
  { position: 33, name: 'Community Chest', type: 'community_chest' },
  { position: 34, name: 'Pennsylvania Avenue', type: 'property', color: 'green' },
  { position: 35, name: 'Short Line', type: 'railroad' },
  { position: 36, name: 'Chance', type: 'chance' },
  { position: 37, name: 'Park Place', type: 'property', color: 'blue' },
  { position: 38, name: 'Luxury Tax', type: 'tax' },
  { position: 39, name: 'Boardwalk', type: 'property', color: 'blue' },
];

/**
 * Property information including prices and rents.
 */
export const PROPERTY_INFO: Record<number, PropertyInfo> = {
  1: {
    position: 1,
    name: 'Mediterranean Avenue',
    type: 'property',
    color: 'brown',
    price: 60,
    rent: [2, 10, 30, 90, 160, 250],
    buildCost: 50,
    mortgageValue: 30,
  },
  3: {
    position: 3,
    name: 'Baltic Avenue',
    type: 'property',
    color: 'brown',
    price: 60,
    rent: [4, 20, 60, 180, 320, 450],
    buildCost: 50,
    mortgageValue: 30,
  },
  5: {
    position: 5,
    name: 'Reading Railroad',
    type: 'railroad',
    price: 200,
    rent: [25, 50, 100, 200],
    buildCost: 0,
    mortgageValue: 100,
  },
  6: {
    position: 6,
    name: 'Oriental Avenue',
    type: 'property',
    color: 'lightblue',
    price: 100,
    rent: [6, 30, 90, 270, 400, 550],
    buildCost: 50,
    mortgageValue: 50,
  },
  8: {
    position: 8,
    name: 'Vermont Avenue',
    type: 'property',
    color: 'lightblue',
    price: 100,
    rent: [6, 30, 90, 270, 400, 550],
    buildCost: 50,
    mortgageValue: 50,
  },
  9: {
    position: 9,
    name: 'Connecticut Avenue',
    type: 'property',
    color: 'lightblue',
    price: 120,
    rent: [8, 40, 100, 300, 450, 600],
    buildCost: 50,
    mortgageValue: 60,
  },
  11: {
    position: 11,
    name: 'St. Charles Place',
    type: 'property',
    color: 'magenta',
    price: 140,
    rent: [10, 50, 150, 450, 625, 750],
    buildCost: 100,
    mortgageValue: 70,
  },
  12: {
    position: 12,
    name: 'Electric Company',
    type: 'utility',
    price: 150,
    rent: [4, 10], // Multiplied by dice roll
    buildCost: 0,
    mortgageValue: 75,
  },
  13: {
    position: 13,
    name: 'States Avenue',
    type: 'property',
    color: 'magenta',
    price: 140,
    rent: [10, 50, 150, 450, 625, 750],
    buildCost: 100,
    mortgageValue: 70,
  },
  14: {
    position: 14,
    name: 'Virginia Avenue',
    type: 'property',
    color: 'magenta',
    price: 160,
    rent: [12, 60, 180, 500, 700, 900],
    buildCost: 100,
    mortgageValue: 80,
  },
  15: {
    position: 15,
    name: 'Pennsylvania Railroad',
    type: 'railroad',
    price: 200,
    rent: [25, 50, 100, 200],
    buildCost: 0,
    mortgageValue: 100,
  },
  16: {
    position: 16,
    name: 'St. James Place',
    type: 'property',
    color: 'orange',
    price: 180,
    rent: [14, 70, 200, 550, 750, 950],
    buildCost: 100,
    mortgageValue: 90,
  },
  18: {
    position: 18,
    name: 'Tennessee Avenue',
    type: 'property',
    color: 'orange',
    price: 180,
    rent: [14, 70, 200, 550, 750, 950],
    buildCost: 100,
    mortgageValue: 90,
  },
  19: {
    position: 19,
    name: 'New York Avenue',
    type: 'property',
    color: 'orange',
    price: 200,
    rent: [16, 80, 220, 600, 800, 1000],
    buildCost: 100,
    mortgageValue: 100,
  },
  21: {
    position: 21,
    name: 'Kentucky Avenue',
    type: 'property',
    color: 'red',
    price: 220,
    rent: [18, 90, 250, 700, 875, 1050],
    buildCost: 150,
    mortgageValue: 110,
  },
  23: {
    position: 23,
    name: 'Indiana Avenue',
    type: 'property',
    color: 'red',
    price: 220,
    rent: [18, 90, 250, 700, 875, 1050],
    buildCost: 150,
    mortgageValue: 110,
  },
  24: {
    position: 24,
    name: 'Illinois Avenue',
    type: 'property',
    color: 'red',
    price: 240,
    rent: [20, 100, 300, 750, 925, 1100],
    buildCost: 150,
    mortgageValue: 120,
  },
  25: {
    position: 25,
    name: 'B&O Railroad',
    type: 'railroad',
    price: 200,
    rent: [25, 50, 100, 200],
    buildCost: 0,
    mortgageValue: 100,
  },
  26: {
    position: 26,
    name: 'Atlantic Avenue',
    type: 'property',
    color: 'yellow',
    price: 260,
    rent: [22, 110, 330, 800, 975, 1150],
    buildCost: 150,
    mortgageValue: 130,
  },
  27: {
    position: 27,
    name: 'Ventnor Avenue',
    type: 'property',
    color: 'yellow',
    price: 260,
    rent: [22, 110, 330, 800, 975, 1150],
    buildCost: 150,
    mortgageValue: 130,
  },
  28: {
    position: 28,
    name: 'Water Works',
    type: 'utility',
    price: 150,
    rent: [4, 10], // Multiplied by dice roll
    buildCost: 0,
    mortgageValue: 75,
  },
  29: {
    position: 29,
    name: 'Marvin Gardens',
    type: 'property',
    color: 'yellow',
    price: 280,
    rent: [24, 120, 360, 850, 1025, 1200],
    buildCost: 150,
    mortgageValue: 140,
  },
  31: {
    position: 31,
    name: 'Pacific Avenue',
    type: 'property',
    color: 'green',
    price: 300,
    rent: [26, 130, 390, 900, 1100, 1275],
    buildCost: 200,
    mortgageValue: 150,
  },
  32: {
    position: 32,
    name: 'North Carolina Avenue',
    type: 'property',
    color: 'green',
    price: 300,
    rent: [26, 130, 390, 900, 1100, 1275],
    buildCost: 200,
    mortgageValue: 150,
  },
  34: {
    position: 34,
    name: 'Pennsylvania Avenue',
    type: 'property',
    color: 'green',
    price: 320,
    rent: [28, 150, 450, 1000, 1200, 1400],
    buildCost: 200,
    mortgageValue: 160,
  },
  35: {
    position: 35,
    name: 'Short Line',
    type: 'railroad',
    price: 200,
    rent: [25, 50, 100, 200],
    buildCost: 0,
    mortgageValue: 100,
  },
  37: {
    position: 37,
    name: 'Park Place',
    type: 'property',
    color: 'blue',
    price: 350,
    rent: [35, 175, 500, 1100, 1300, 1500],
    buildCost: 200,
    mortgageValue: 175,
  },
  39: {
    position: 39,
    name: 'Boardwalk',
    type: 'property',
    color: 'blue',
    price: 400,
    rent: [50, 200, 600, 1400, 1700, 2000],
    buildCost: 200,
    mortgageValue: 200,
  },
};

/**
 * Get board space by position.
 */
export function getBoardSpace(position: number): BoardSpace | undefined {
  return BOARD_SPACES.find((space) => space.position === position);
}

/**
 * Get property info by position.
 */
export function getPropertyInfo(position: number): PropertyInfo | undefined {
  return PROPERTY_INFO[position];
}

/**
 * Get positions of all properties in a color group.
 */
export function getColorGroupPositions(color: PropertyColor): number[] {
  return BOARD_SPACES.filter((space) => space.color === color).map((space) => space.position);
}

/**
 * Check if a position is a corner space.
 */
export function isCornerSpace(position: number): boolean {
  return [0, 10, 20, 30].includes(position);
}

/**
 * Get side of board for a position.
 */
export function getBoardSide(position: number): 'bottom' | 'left' | 'top' | 'right' {
  if (position <= 10) return 'bottom';
  if (position <= 19) return 'left';
  if (position <= 30) return 'top';
  return 'right';
}

/**
 * Get CSS color class for property color.
 */
export function getColorClass(color?: PropertyColor): string {
  if (!color) return '';

  const colorMap: Record<PropertyColor, string> = {
    brown: 'bg-property-brown',
    lightblue: 'bg-property-lightblue',
    magenta: 'bg-property-magenta',
    orange: 'bg-property-orange',
    red: 'bg-property-red',
    yellow: 'bg-property-yellow',
    green: 'bg-property-green',
    blue: 'bg-property-blue',
    railroad: 'bg-property-railroad',
    utility: 'bg-property-utility',
  };

  return colorMap[color] || '';
}

/**
 * Get hex color for property color.
 */
export function getColorHex(color?: PropertyColor): string {
  if (!color) return '#808080';

  const colorMap: Record<PropertyColor, string> = {
    brown: '#8B4513',
    lightblue: '#87CEEB',
    magenta: '#FF00FF',
    orange: '#FFA500',
    red: '#FF0000',
    yellow: '#FFFF00',
    green: '#008000',
    blue: '#0000FF',
    railroad: '#000000',
    utility: '#808080',
  };

  return colorMap[color];
}
