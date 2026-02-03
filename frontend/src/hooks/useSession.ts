import { useCallback, useEffect } from 'react';
import { useSessionStore } from '@/store/sessionStore';
import { useUIStore } from '@/store/uiStore';
import { createSession, getSession } from '@/api/players';
import { setSessionId } from '@/api/client';

/**
 * Session management hook.
 *
 * Handles session creation, restoration, and state management.
 */
export function useSession() {
  const {
    sessionId,
    displayName,
    currentGameId,
    currentLobbyId,
    playerId,
    setSession,
    setDisplayName,
    setCurrentGame,
    setCurrentLobby,
    clearSession,
    isLoggedIn,
    isInGame,
    isInLobby,
  } = useSessionStore();

  const { addToast, setGlobalLoading } = useUIStore();

  // Restore session on mount
  useEffect(() => {
    if (sessionId) {
      // Set session ID for API calls
      setSessionId(sessionId);

      // Verify session is still valid
      getSession(sessionId).catch(() => {
        // Session invalid, clear it
        console.log('Session expired, clearing');
        clearSession();
        setSessionId(null);
      });
    }
  }, [sessionId, clearSession]);

  /**
   * Create a new session.
   */
  const login = useCallback(
    async (name: string) => {
      setGlobalLoading(true, 'Creating session...');
      try {
        const response = await createSession(name);
        setSession(response.session_id, response.display_name);
        addToast(`Welcome, ${response.display_name}!`, 'success');
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to create session';
        addToast(message, 'error');
        throw error;
      } finally {
        setGlobalLoading(false);
      }
    },
    [setSession, addToast, setGlobalLoading]
  );

  /**
   * End current session.
   */
  const logout = useCallback(() => {
    clearSession();
    setSessionId(null);
    addToast('Logged out', 'info');
  }, [clearSession, addToast]);

  /**
   * Update display name.
   */
  const updateName = useCallback(
    (name: string) => {
      setDisplayName(name);
    },
    [setDisplayName]
  );

  /**
   * Join a game.
   */
  const joinGame = useCallback(
    (gameId: string, playerIdx: number) => {
      setCurrentGame(gameId, playerIdx);
    },
    [setCurrentGame]
  );

  /**
   * Leave current game.
   */
  const leaveGame = useCallback(() => {
    setCurrentGame(null, null);
  }, [setCurrentGame]);

  /**
   * Join a lobby.
   */
  const joinLobby = useCallback(
    (lobbyId: string) => {
      setCurrentLobby(lobbyId);
    },
    [setCurrentLobby]
  );

  /**
   * Leave current lobby.
   */
  const leaveLobby = useCallback(() => {
    setCurrentLobby(null);
  }, [setCurrentLobby]);

  return {
    // State
    sessionId,
    displayName,
    currentGameId,
    currentLobbyId,
    playerId,

    // Computed
    isLoggedIn: isLoggedIn(),
    isInGame: isInGame(),
    isInLobby: isInLobby(),

    // Actions
    login,
    logout,
    updateName,
    joinGame,
    leaveGame,
    joinLobby,
    leaveLobby,
  };
}
