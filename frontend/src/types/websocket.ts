import type { GameState } from './game';

/**
 * WebSocket message types from server.
 */
export enum WSMessageType {
  // Game state
  STATE_UPDATE = 'state_update',
  GAME_OVER = 'game_over',

  // Player events
  PLAYER_JOINED = 'player_joined',
  PLAYER_LEFT = 'player_left',
  PLAYER_RECONNECTED = 'player_reconnected',

  // Action responses
  ACTION_RESULT = 'action_result',
  ACTION_ERROR = 'action_error',

  // Connection
  HEARTBEAT_ACK = 'heartbeat_ack',
  ERROR = 'error',

  // Chat
  CHAT_MESSAGE = 'chat_message',
}

/**
 * Base WebSocket message structure.
 */
export interface WSMessage {
  type: WSMessageType | string;
  data?: unknown;
  timestamp?: number;
}

/**
 * State update message from server.
 */
export interface StateUpdateMessage extends WSMessage {
  type: WSMessageType.STATE_UPDATE;
  data: GameState;
}

/**
 * Game over message from server.
 */
export interface GameOverMessage extends WSMessage {
  type: WSMessageType.GAME_OVER;
  data: {
    winner: number;
    reason: string;
    finalState: GameState;
  };
}

/**
 * Player joined message.
 */
export interface PlayerJoinedMessage extends WSMessage {
  type: WSMessageType.PLAYER_JOINED;
  data: {
    playerId: number;
    playerName: string;
  };
}

/**
 * Player left message.
 */
export interface PlayerLeftMessage extends WSMessage {
  type: WSMessageType.PLAYER_LEFT;
  data: {
    playerId: number;
    reason: string;
  };
}

/**
 * Action result message.
 */
export interface ActionResultMessage extends WSMessage {
  type: WSMessageType.ACTION_RESULT;
  data: {
    success: boolean;
    actionType: string;
    message?: string;
  };
}

/**
 * Error message from server.
 */
export interface ErrorMessage extends WSMessage {
  type: WSMessageType.ERROR;
  data: {
    code: string;
    message: string;
  };
}

/**
 * Chat message.
 */
export interface ChatMessage extends WSMessage {
  type: WSMessageType.CHAT_MESSAGE;
  data: {
    playerId: number;
    playerName: string;
    message: string;
    timestamp: number;
  };
}

/**
 * Client action message to send to server.
 */
export interface ClientActionMessage {
  type: 'action';
  data: {
    action_type: string;
    [key: string]: unknown;
  };
}

/**
 * Client heartbeat message.
 */
export interface ClientHeartbeatMessage {
  type: 'heartbeat';
}

/**
 * Client chat message.
 */
export interface ClientChatMessage {
  type: 'chat';
  data: {
    message: string;
  };
}

/**
 * Union type for all client messages.
 */
export type ClientMessage = ClientActionMessage | ClientHeartbeatMessage | ClientChatMessage;

/**
 * WebSocket connection state.
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
