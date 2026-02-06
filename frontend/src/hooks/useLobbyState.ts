import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLobbyStore } from '@/store/lobbyStore';
import { useSessionStore } from '@/store/sessionStore';
import { useUIStore } from '@/store/uiStore';
import {
  createLobby,
  getLobby,
  joinLobby as joinLobbyApi,
  leaveLobby as leaveLobbyApi,
  listLobbies,
  setReady as setReadyApi,
  addAiPlayer as addAiPlayerApi,
  removeAiPlayer as removeAiPlayerApi,
  updateLobbySettings as updateSettingsApi,
  startGame as startGameApi,
  type CreateLobbyRequest,
} from '@/api/lobbies';
import type { LobbySettings } from '@/types';
import { WSMessageType, type ConnectionState } from '@/types/websocket';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const HEARTBEAT_INTERVAL = 30000;
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Lobby state management hook.
 *
 * Handles:
 * - Fetching lobby list and details
 * - WebSocket connection for real-time lobby updates
 * - Player actions (join, leave, ready, AI management)
 * - Game start flow
 */
export function useLobbyState(lobbyId?: string) {
  const navigate = useNavigate();
  const { sessionId, displayName, setSession, setCurrentLobby, setCurrentGame } = useSessionStore();
  const {
    currentLobby,
    lobbyList,
    isLoading,
    error,
    setCurrentLobby: setLobbyState,
    updateLobby,
    setLobbyList,
    setLoading,
    setError,
    addPlayer,
    removePlayer,
    removePlayerBySlot,
    updatePlayerReady,
    updateSettings,
    isHost,
    canStartGame,
    getMySlot,
    reset,
  } = useLobbyStore();
  const { addToast, setGlobalLoading } = useUIStore();

  const wsRef = useRef<WebSocket | null>(null);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');

  // WebSocket message handler
  const handleWsMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case WSMessageType.LOBBY_UPDATE:
            setLobbyState(message.data);
            break;

          case WSMessageType.LOBBY_PLAYER_JOINED:
            addPlayer(message.data);
            addToast(`${message.data.name} joined the lobby`, 'info');
            break;

          case WSMessageType.LOBBY_PLAYER_LEFT:
            removePlayer(message.data.sessionId);
            addToast(`A player left the lobby`, 'info');
            break;

          case WSMessageType.LOBBY_PLAYER_READY:
            updatePlayerReady(message.data.sessionId, message.data.isReady);
            break;

          case WSMessageType.LOBBY_AI_ADDED:
            addPlayer(message.data);
            addToast(`AI player "${message.data.name}" added`, 'info');
            break;

          case WSMessageType.LOBBY_AI_REMOVED:
            removePlayerBySlot(message.data.slot_id);
            addToast('AI player removed', 'info');
            break;

          case WSMessageType.LOBBY_SETTINGS_CHANGED:
            updateSettings(message.data);
            break;

          case WSMessageType.LOBBY_GAME_STARTING:
            addToast('Game starting...', 'info');
            break;

          case WSMessageType.LOBBY_GAME_STARTED:
            setCurrentGame(message.data.game_id, null); // Server will look up player by session
            navigate(`/game/${message.data.game_id}`);
            break;

          case WSMessageType.ERROR:
            setError(message.data.message);
            addToast(message.data.message, 'error');
            break;

          case WSMessageType.HEARTBEAT_ACK:
            // Heartbeat acknowledged
            break;

          default:
            console.log('Unknown lobby message:', message.type);
        }
      } catch (e) {
        console.error('Failed to parse lobby WebSocket message:', e);
      }
    },
    [
      setLobbyState,
      addPlayer,
      removePlayer,
      removePlayerBySlot,
      updatePlayerReady,
      updateSettings,
      setError,
      addToast,
      navigate,
      setCurrentGame,
    ]
  );

  // Connect to lobby WebSocket
  const connectWs = useCallback(() => {
    if (!sessionId || !lobbyId) return;

    setConnectionState('connecting');

    const wsUrl = `${WS_BASE_URL}/api/lobbies/ws/${lobbyId}?session_id=${sessionId}`;
    console.log('Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Lobby WebSocket connected');
      setConnectionState('connected');
      reconnectAttemptsRef.current = 0;

      // Start heartbeat
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'heartbeat' }));
        }
      }, HEARTBEAT_INTERVAL);
    };

    ws.onmessage = handleWsMessage;

    ws.onclose = (event) => {
      console.log('Lobby WebSocket closed:', event.code, event.reason);
      setConnectionState('disconnected');

      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }

      // Attempt reconnection for abnormal closures
      if (event.code !== 1000 && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        setConnectionState('reconnecting');
        reconnectAttemptsRef.current++;
        setTimeout(connectWs, RECONNECT_DELAY);
      }
    };

    ws.onerror = (error) => {
      console.error('Lobby WebSocket error:', error);
    };

    wsRef.current = ws;
  }, [sessionId, lobbyId, handleWsMessage]);

  // Disconnect WebSocket
  const disconnectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect');
      wsRef.current = null;
    }
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
    }
  }, []);

  // Fetch lobby list
  const fetchLobbyList = useCallback(async () => {
    setLoading(true);
    try {
      const lobbies = await listLobbies();
      setLobbyList(lobbies);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch lobbies';
      setError(message);
      addToast(message, 'error');
    } finally {
      setLoading(false);
    }
  }, [setLoading, setLobbyList, setError, addToast]);

  // Fetch specific lobby
  const fetchLobby = useCallback(
    async (id: string) => {
      setLoading(true);
      try {
        const lobby = await getLobby(id);
        setLobbyState(lobby);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch lobby';
        setError(message);
        addToast(message, 'error');
      } finally {
        setLoading(false);
      }
    },
    [setLoading, setLobbyState, setError, addToast]
  );

  // Create a new lobby
  const create = useCallback(
    async (request: CreateLobbyRequest) => {
      setGlobalLoading(true, 'Creating lobby...');
      try {
        const response = await createLobby(request);
        console.log('Lobby created:', response);
        // Backend generates session_id - update our session store with it
        setSession(response.session_id, request.host_name);
        console.log('Session set to:', response.session_id);
        setCurrentLobby(response.id);
        addToast(`Lobby "${response.name}" created!`, 'success');
        navigate(`/lobby/${response.id}`);
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to create lobby';
        addToast(message, 'error');
        return null;
      } finally {
        setGlobalLoading(false);
      }
    },
    [setGlobalLoading, setSession, setCurrentLobby, addToast, navigate]
  );

  // Join a lobby
  const join = useCallback(
    async (id: string, code?: string) => {
      if (!sessionId || !displayName) {
        addToast('Please log in first', 'error');
        return false;
      }

      setGlobalLoading(true, 'Joining lobby...');
      try {
        const lobby = await joinLobbyApi(id, sessionId, displayName, code);
        setLobbyState(lobby);
        setCurrentLobby(lobby.id);
        addToast(`Joined "${lobby.name}"!`, 'success');
        navigate(`/lobby/${lobby.id}`);
        return true;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to join lobby';
        addToast(message, 'error');
        return false;
      } finally {
        setGlobalLoading(false);
      }
    },
    [sessionId, displayName, setGlobalLoading, setLobbyState, setCurrentLobby, addToast, navigate]
  );

  // Leave current lobby
  const leave = useCallback(async () => {
    if (!sessionId || !currentLobby) return;

    setGlobalLoading(true, 'Leaving lobby...');
    try {
      await leaveLobbyApi(currentLobby.id, sessionId);
      disconnectWs();
      reset();
      setCurrentLobby(null);
      addToast('Left lobby', 'info');
      navigate('/lobby');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to leave lobby';
      addToast(message, 'error');
    } finally {
      setGlobalLoading(false);
    }
  }, [sessionId, currentLobby, setGlobalLoading, disconnectWs, reset, setCurrentLobby, addToast, navigate]);

  // Toggle ready status
  const toggleReady = useCallback(async () => {
    if (!sessionId || !currentLobby) return;

    const mySlot = getMySlot(sessionId);
    if (!mySlot) return;

    setLoading(true);
    try {
      const lobby = await setReadyApi(currentLobby.id, sessionId, !mySlot.is_ready);
      setLobbyState(lobby);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update ready status';
      addToast(message, 'error');
    } finally {
      setLoading(false);
    }
  }, [sessionId, currentLobby, getMySlot, setLoading, setLobbyState, addToast]);

  // Add AI player
  const addAi = useCallback(
    async (aiType: string = 'rule_based') => {
      if (!sessionId || !currentLobby) return;

      setLoading(true);
      try {
        const lobby = await addAiPlayerApi(currentLobby.id, sessionId, aiType);
        setLobbyState(lobby);
        addToast('AI player added', 'success');
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to add AI player';
        addToast(message, 'error');
      } finally {
        setLoading(false);
      }
    },
    [sessionId, currentLobby, setLoading, setLobbyState, addToast]
  );

  // Remove AI player
  const removeAi = useCallback(
    async (slot: number) => {
      if (!sessionId || !currentLobby) return;

      setLoading(true);
      try {
        const lobby = await removeAiPlayerApi(currentLobby.id, sessionId, slot);
        setLobbyState(lobby);
        addToast('AI player removed', 'info');
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to remove AI player';
        addToast(message, 'error');
      } finally {
        setLoading(false);
      }
    },
    [sessionId, currentLobby, setLoading, setLobbyState, addToast]
  );

  // Update lobby settings
  const changeSettings = useCallback(
    async (newSettings: Partial<LobbySettings>) => {
      if (!sessionId || !currentLobby) return;

      setLoading(true);
      try {
        // Merge with current settings for the full object
        const fullSettings: LobbySettings = {
          ...currentLobby.settings,
          ...newSettings,
        };
        const lobby = await updateSettingsApi(currentLobby.id, sessionId, fullSettings);
        setLobbyState(lobby);
        addToast('Settings updated', 'success');
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to update settings';
        addToast(message, 'error');
      } finally {
        setLoading(false);
      }
    },
    [sessionId, currentLobby, setLoading, setLobbyState, addToast]
  );

  // Start game
  const start = useCallback(async () => {
    if (!sessionId || !currentLobby) return;

    if (!canStartGame()) {
      addToast('Cannot start game: not all players are ready', 'error');
      return;
    }

    setGlobalLoading(true, 'Starting game...');
    try {
      const result = await startGameApi(currentLobby.id, sessionId);
      updateLobby({ game_id: result.game_id });
      setCurrentGame(result.game_id, null); // Server will look up player by session
      addToast('Game starting!', 'success');
      navigate(`/game/${result.game_id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start game';
      addToast(message, 'error');
    } finally {
      setGlobalLoading(false);
    }
  }, [sessionId, currentLobby, canStartGame, setGlobalLoading, updateLobby, setCurrentGame, addToast, navigate]);

  // Connect WebSocket when lobbyId is provided
  useEffect(() => {
    if (lobbyId && sessionId) {
      fetchLobby(lobbyId);
      connectWs();
    }

    return () => {
      disconnectWs();
    };
  }, [lobbyId, sessionId, fetchLobby, connectWs, disconnectWs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWs();
    };
  }, [disconnectWs]);

  return {
    // State
    currentLobby,
    lobbyList,
    isLoading,
    error,
    connectionState,

    // Computed
    isHost: sessionId ? isHost(sessionId) : false,
    canStart: canStartGame(),
    mySlot: sessionId ? getMySlot(sessionId) : null,

    // Actions
    fetchLobbyList,
    fetchLobby,
    create,
    join,
    leave,
    toggleReady,
    addAi,
    removeAi,
    changeSettings,
    start,
    reset,
  };
}
