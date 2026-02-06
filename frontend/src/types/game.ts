import type { PlayerState } from './player';
import type { PropertyState } from './property';

export interface DiceRoll {
  die1: number;
  die2: number;
  isDoubles: boolean;
}

export interface GameState {
  // Note: Backend sends snake_case, so we use snake_case here
  game_id?: string;
  players: PlayerState[];
  properties: Record<string, PropertyState>;  // Backend sends string keys
  current_player: number;
  turn_number: number;
  last_roll: [number, number] | null;  // Backend sends tuple
  doubles_count: number;
  houses_remaining: number;
  hotels_remaining: number;
  game_over: boolean;
  winner: number | null;
  event_log: string[];

  // Aliased for easier access (optional)
  currentPlayer?: number;
  turnNumber?: number;
  gamePhase?: GamePhase;
  lastRoll?: DiceRoll | null;
  housesRemaining?: number;
  hotelsRemaining?: number;
  gameOver?: boolean;
  rolledDoubles?: boolean;
  doublesCount?: number;
}

export type GamePhase =
  | 'waiting'
  | 'pre_roll'
  | 'post_roll'
  | 'in_jail'
  | 'bankrupt'
  | 'game_over';

export interface GameEvent {
  id: string;
  timestamp: number;
  type: GameEventType;
  playerId: number;
  message: string;
  data?: Record<string, unknown>;
}

export type GameEventType =
  | 'roll'
  | 'move'
  | 'buy'
  | 'rent'
  | 'build'
  | 'mortgage'
  | 'jail'
  | 'card'
  | 'trade'
  | 'bankrupt'
  | 'win';
