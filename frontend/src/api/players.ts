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
 * Note: sessionId param is for setting the header before the call.
 */
export async function getSession(sessionId: string): Promise<SessionResponse> {
  // Ensure session ID header is set
  setSessionId(sessionId);
  const response = await apiClient.get<SessionResponse>('/players/me');
  return response.data;
}

/**
 * Update display name.
 */
export async function updateDisplayName(
  sessionId: string,
  displayName: string
): Promise<SessionResponse> {
  // Ensure session ID header is set
  setSessionId(sessionId);
  const response = await apiClient.put<SessionResponse>('/players/me', {
    display_name: displayName,
  });
  return response.data;
}
