/**
 * Lobby player slot.
 */
export interface LobbyPlayer {
  sessionId: string;
  displayName: string;
  isReady: boolean;
  isAi: boolean;
  aiType?: string;
  slot: number;
}

/**
 * Lobby settings.
 */
export interface LobbySettings {
  maxPlayers: number;
  startingMoney: number;
  turnTimeLimit: number | null;
  allowSpectators: boolean;
}

/**
 * Lobby state.
 */
export interface LobbyState {
  id: string;
  name: string;
  code: string;
  hostSessionId: string;
  players: LobbyPlayer[];
  settings: LobbySettings;
  isPrivate: boolean;
  gameId: string | null;
  createdAt: number;
}

/**
 * Lobby list item (for lobby browser).
 */
export interface LobbyListItem {
  id: string;
  name: string;
  code: string;
  playerCount: number;
  maxPlayers: number;
  hostName: string;
  isPrivate: boolean;
}
