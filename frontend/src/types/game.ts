import type { PlayerState } from './player';
import type { PropertyState } from './property';

export interface DiceRoll {
  die1: number;
  die2: number;
  isDoubles: boolean;
}

export interface GameState {
  // Core fields (always present after store transformation)
  players: PlayerState[];
  properties: Record<number, PropertyState>;
  currentPlayer: number;
  turnNumber: number;
  gamePhase: GamePhase;
  lastRoll: DiceRoll | null;
  doublesCount: number;
  housesRemaining: number;
  hotelsRemaining: number;
  gameOver: boolean;
  winner: number | null;
  rolledDoubles: boolean;

  // Backend snake_case fields (present from raw spread)
  game_id?: string;
  current_player?: number;
  decision_player?: number;
  rules_id?: string;
  debt?: { debtor: number; amount: number; creditor: number | null } | null;
  legal_actions?: { type: string; player_id: number; property_id?: number }[];
  turn_number?: number;
  last_roll?: [number, number] | null;
  doubles_count?: number;
  houses_remaining?: number;
  hotels_remaining?: number;
  game_over?: boolean;
  event_log?: string[];

  // Allow additional backend fields
  [key: string]: unknown;
}

export type GamePhase =
  | 'jail_decision'
  | 'purchase_decision'
  | 'asset_management'
  | 'debt_resolution'
  | 'terminal'
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
