import { apiClient } from './client';
import type { LobbyState, LobbyListItem, LobbySettings } from '@/types';

/**
 * Create lobby request.
 */
export interface CreateLobbyRequest {
  name: string;
  host_name: string;
  settings?: Partial<LobbySettings>;
}

/**
 * Create lobby response from API.
 */
export interface CreateLobbyResponse {
  id: string;
  name: string;
  session_id: string;
  invite_code: string | null;
  websocket_url: string;
  created_at: string;
}

/**
 * Create a new lobby.
 * Note: The backend generates a new session_id and returns it.
 * The frontend should use this returned session_id for subsequent requests.
 */
export async function createLobby(request: CreateLobbyRequest): Promise<CreateLobbyResponse> {
  const response = await apiClient.post<CreateLobbyResponse>('/lobbies', request);
  return response.data;
}

/**
 * List available lobbies.
 */
export async function listLobbies(): Promise<LobbyListItem[]> {
  const response = await apiClient.get<LobbyListItem[]>('/lobbies');
  return response.data;
}

/**
 * Get lobby by ID.
 */
export async function getLobby(lobbyId: string): Promise<LobbyState> {
  const response = await apiClient.get<LobbyState>(`/lobbies/${lobbyId}`);
  return response.data;
}

/**
 * Join a lobby.
 */
export async function joinLobby(
  lobbyId: string,
  sessionId: string,
  playerName: string,
  inviteCode?: string
): Promise<LobbyState> {
  const response = await apiClient.post<LobbyState>(
    `/lobbies/${lobbyId}/join`,
    { player_name: playerName, invite_code: inviteCode },
    { headers: { 'X-Session-Id': sessionId } }
  );
  return response.data;
}

/**
 * Leave a lobby.
 */
export async function leaveLobby(lobbyId: string, sessionId: string): Promise<void> {
  await apiClient.post(`/lobbies/${lobbyId}/leave`, null, {
    headers: { 'X-Session-Id': sessionId },
  });
}

/**
 * Set ready status.
 */
export async function setReady(
  lobbyId: string,
  sessionId: string,
  isReady: boolean
): Promise<LobbyState> {
  const response = await apiClient.post<LobbyState>(
    `/lobbies/${lobbyId}/ready`,
    { is_ready: isReady },
    { headers: { 'X-Session-Id': sessionId } }
  );
  return response.data;
}

/**
 * Add AI player to lobby.
 */
export async function addAiPlayer(
  lobbyId: string,
  sessionId: string,
  aiType: string = 'rule_based'
): Promise<LobbyState> {
  const response = await apiClient.post<LobbyState>(
    `/lobbies/${lobbyId}/ai`,
    { ai_type: aiType },
    { headers: { 'X-Session-Id': sessionId } }
  );
  return response.data;
}

/**
 * Remove AI player from lobby.
 */
export async function removeAiPlayer(
  lobbyId: string,
  sessionId: string,
  slot: number
): Promise<LobbyState> {
  const response = await apiClient.delete<LobbyState>(`/lobbies/${lobbyId}/ai/${slot}`, {
    headers: { 'X-Session-Id': sessionId },
  });
  return response.data;
}

/**
 * Update lobby settings (host only).
 */
export async function updateLobbySettings(
  lobbyId: string,
  sessionId: string,
  settings: LobbySettings
): Promise<LobbyState> {
  const response = await apiClient.put<LobbyState>(
    `/lobbies/${lobbyId}/settings`,
    settings,
    { headers: { 'X-Session-Id': sessionId } }
  );
  return response.data;
}

/**
 * Start game from lobby.
 */
export async function startGame(
  lobbyId: string,
  sessionId: string
): Promise<{ game_id: string }> {
  const response = await apiClient.post<{ game_id: string }>(
    `/lobbies/${lobbyId}/start`,
    null,
    { headers: { 'X-Session-Id': sessionId } }
  );
  return response.data;
}
