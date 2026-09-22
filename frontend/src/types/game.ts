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
  revision: number;
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
  decision_contract?: DecisionContract;

  // Allow additional backend fields
  [key: string]: unknown;
}

export type GamePhase =
  | 'jail_decision'
  | 'purchase_decision'
  | 'asset_management'
  | 'debt_resolution'
  | 'trade_response'
  | 'terminal'
  | 'waiting'
  | 'pre_roll'
  | 'post_roll'
  | 'in_jail'
  | 'bankrupt'
  | 'game_over';

export interface TradeOffer {
  trade_id: number;
  created_revision: number;
  from_player: number;
  to_player: number;
  give_properties: number[];
  give_money: number;
  want_properties: number[];
  want_money: number;
}

export interface DecisionContract {
  contract_version: 'decision-contract-v1';
  rules_id: string;
  revision: number;
  viewer_id: number | null;
  turn_owner: number;
  decision_player: number;
  phase: GamePhase;
  properties: {
    position: number;
    name: string;
    owner: number | null;
    houses: number;
    mortgaged: boolean;
    price: number;
    redemption_cost: number;
    group: string | null;
  }[];
  trade: {
    can_propose: boolean;
    proposals_remaining: number;
    used_recipients: number[];
    eligible_recipients: number[];
    tradeable_properties: number[];
  };
  pending_offer: TradeOffer | null;
}

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
