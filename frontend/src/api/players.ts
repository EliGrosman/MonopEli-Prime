import { apiClient, setSessionId } from './client';

/**
 * Session response from API.
 */
export interface SessionResponse {
  session_id: string;
  display_name: string;
  created_at: string;
}

/**
 * Create a new player session.
 */
export async function createSession(displayName: string): Promise<SessionResponse> {
  const response = await apiClient.post<SessionResponse>('/players/session', {
    display_name: displayName,
  });
  // Set session ID for future requests
  setSessionId(response.data.session_id);
  return response.data;
}

/**
 * Get current session info.
 */
export async function getSession(sessionId: string): Promise<SessionResponse> {
  const response = await apiClient.get<SessionResponse>(`/players/session/${sessionId}`);
  return response.data;
}

/**
 * Update display name.
 */
export async function updateDisplayName(
  sessionId: string,
  displayName: string
): Promise<SessionResponse> {
  const response = await apiClient.patch<SessionResponse>(`/players/session/${sessionId}`, {
    display_name: displayName,
  });
  return response.data;
}
