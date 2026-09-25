import type { GameState } from './game';

/**
 * WebSocket message types from server.
 */
export const WSMessageType = {
  // Game state
  STATE_UPDATE: 'state_update',
  GAME_OVER: 'game_over',
  AGENT_UPDATE: 'agent_update',

  // Player events
  PLAYER_JOINED: 'player_joined',
  PLAYER_LEFT: 'player_left',
  PLAYER_RECONNECTED: 'player_reconnected',

  // Action responses
  ACTION_RESULT: 'action_result',
  ACTION_ERROR: 'action_error',

  // Connection
  IDENTITY: 'identity',
  HEARTBEAT_ACK: 'heartbeat_ack',
  ERROR: 'error',

  // Chat
  CHAT_MESSAGE: 'chat_message',

  // Lobby events
  LOBBY_UPDATE: 'lobby_update',
  LOBBY_PLAYER_JOINED: 'lobby_player_joined',
  LOBBY_PLAYER_LEFT: 'lobby_player_left',
  LOBBY_PLAYER_READY: 'lobby_player_ready',
  LOBBY_AI_ADDED: 'lobby_ai_added',
  LOBBY_AI_REMOVED: 'lobby_ai_removed',
  LOBBY_SETTINGS_CHANGED: 'lobby_settings_changed',
  LOBBY_GAME_STARTING: 'lobby_game_starting',
  LOBBY_GAME_STARTED: 'lobby_game_started',
} as const;

export type WSMessageType = (typeof WSMessageType)[keyof typeof WSMessageType];

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
  type: typeof WSMessageType.STATE_UPDATE;
  data: GameState;
}

/**
 * Game over message from server.
 */
export interface GameOverMessage extends WSMessage {
  type: typeof WSMessageType.GAME_OVER;
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
  type: typeof WSMessageType.PLAYER_JOINED;
  data: {
    playerId: number;
    playerName: string;
  };
}

/**
 * Player left message.
 */
export interface PlayerLeftMessage extends WSMessage {
  type: typeof WSMessageType.PLAYER_LEFT;
  data: {
    playerId: number;
    reason: string;
  };
}

/**
 * Action result message.
 */
export interface ActionResultMessage extends WSMessage {
  type: typeof WSMessageType.ACTION_RESULT;
  data: {
    success: boolean;
    actionType: string;
    message?: string;
    request_id?: string;
    revision?: number;
    error_code?: string;
  };
}

/**
 * Error message from server.
 */
export interface ErrorMessage extends WSMessage {
  type: typeof WSMessageType.ERROR;
  data: {
    code: string;
    message: string;
  };
}

/**
 * Chat message.
 */
export interface ChatMessage extends WSMessage {
  type: typeof WSMessageType.CHAT_MESSAGE;
  data: {
    playerId: number;
    playerName: string;
    message: string;
    timestamp: number;
  };
}

/**
 * Identity message - tells client their player_id.
 */
export interface IdentityMessage extends WSMessage {
  type: typeof WSMessageType.IDENTITY;
  data: {
    player_id: number | null;
    player_name: string;
  };
}

/**
 * Client action message to send to server.
 */
export interface ClientActionMessage {
  type: 'action';
  data: {
    action_type: string;
    request_id?: string;
    expected_revision?: number;
    contract_version?: string;
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

// Lobby WebSocket messages
import type { LobbyState, LobbyPlayer, LobbySettings } from './lobby';

/**
 * Lobby update message (full state sync).
 */
export interface LobbyUpdateMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_UPDATE;
  data: LobbyState;
}

/**
 * Player joined lobby message.
 */
export interface LobbyPlayerJoinedMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_PLAYER_JOINED;
  data: LobbyPlayer;
}

/**
 * Player left lobby message.
 */
export interface LobbyPlayerLeftMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_PLAYER_LEFT;
  data: {
    sessionId: string;
    reason: 'left' | 'kicked' | 'disconnected';
  };
}

/**
 * Player ready status changed message.
 */
export interface LobbyPlayerReadyMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_PLAYER_READY;
  data: {
    sessionId: string;
    isReady: boolean;
  };
}

/**
 * Lobby settings changed message.
 */
export interface LobbySettingsChangedMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_SETTINGS_CHANGED;
  data: Partial<LobbySettings>;
}

/**
 * Game starting from lobby message.
 */
export interface LobbyGameStartingMessage extends WSMessage {
  type: typeof WSMessageType.LOBBY_GAME_STARTING;
  data: {
    gameId: string;
    countdown: number;
  };
}
