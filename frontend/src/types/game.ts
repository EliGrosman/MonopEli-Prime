import type { PlayerState } from './player';
import type { PropertyState } from './property';

export interface DiceRoll {
  die1: number;
  die2: number;
  isDoubles: boolean;
}

export interface GameState {
  gameId: string;
  players: PlayerState[];
  properties: Record<number, PropertyState>;
  currentPlayer: number;
  turnNumber: number;
  lastRoll: DiceRoll | null;
  housesRemaining: number;
  hotelsRemaining: number;
  gamePhase: GamePhase;
  gameOver: boolean;
  winner: number | null;
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
