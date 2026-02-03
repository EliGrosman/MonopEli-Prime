/**
 * Player colors for tokens and UI elements.
 */
export const PLAYER_COLORS = [
  '#E53935', // Red
  '#1E88E5', // Blue
  '#43A047', // Green
  '#FDD835', // Yellow
  '#8E24AA', // Purple
  '#FB8C00', // Orange
  '#00ACC1', // Cyan
  '#6D4C41', // Brown
];

/**
 * Get player color by index.
 */
export function getPlayerColor(playerIndex: number): string {
  return PLAYER_COLORS[playerIndex % PLAYER_COLORS.length];
}

/**
 * Get Tailwind class for player color.
 */
export function getPlayerColorClass(playerIndex: number): string {
  return `bg-player-${playerIndex}`;
}
