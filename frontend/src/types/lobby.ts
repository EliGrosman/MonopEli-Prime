/**
 * Lobby player slot (matches backend snake_case).
 */
export interface LobbyPlayer {
  session_id: string;
  name: string;
  is_host: boolean;
  is_ready: boolean;
  is_ai: boolean;
  ai_type?: string;
  slot_id: number;
  joined_at: string;
}

/**
 * Lobby settings (matches backend snake_case).
 */
export interface LobbySettings {
  max_players: number;
  min_players: number;
  starting_money: number;
  go_salary: number;
  allow_spectators: boolean;
  private: boolean;
  rules_id: 'foundation-v1' | 'foundation-trade-v1';
}

/**
 * Lobby state (matches backend snake_case).
 */
export interface LobbyState {
  id: string;
  name: string;
  host_session_id: string;
  status: string;
  settings: LobbySettings;
  players: LobbyPlayer[];
  spectator_count: number;
  created_at: string;
  game_id: string | null;
  invite_code: string | null;
}

/**
 * Lobby list item (for lobby browser).
 */
export interface LobbyListItem {
  id: string;
  name: string;
  host_name: string;
  status: string;
  current_players: number;
  max_players: number;
  created_at: string;
}
