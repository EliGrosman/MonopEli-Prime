import { useEffect, useRef, useCallback, useState } from 'react';
import { useSessionStore } from '@/store/sessionStore';
import { useGameStore } from '@/store/gameStore';
import type {
  WSMessage,
  WSMessageType,
  ConnectionState,
  ClientMessage,
  IdentityMessage,
} from '@/types';

function getWsBaseUrl(): string {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  if (typeof window !== 'undefined' && window.location.host) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}`;
  }
  return 'ws://localhost:8000';
}
const WS_BASE_URL = getWsBaseUrl();
const HEARTBEAT_INTERVAL = 30000;
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

interface UseWebSocketOptions {
  gameId: string;
  playerId?: number;
  onMessage?: (message: WSMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoConnect?: boolean;
}

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  send: (message: ClientMessage) => void;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
}

/**
 * WebSocket connection hook with auto-reconnection.
 *
 * Features:
 * - Automatic connection management
 * - Heartbeat handling
 * - Auto-reconnection on disconnect
 * - Message queue during disconnect
 * - Type-safe message handling
 */
export function useWebSocket({
  gameId,
  playerId,
  onMessage,
  onConnect,
  onDisconnect,
  autoConnect = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  const sessionId = useSessionStore((s) => s.sessionId);
  const { setConnected, updateGameState, setError } = useGameStore();

  const wsRef = useRef<WebSocket | null>(null);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const messageQueueRef = useRef<string[]>([]);
  const isManualDisconnectRef = useRef(false);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');

  const clearHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(
    (ws: WebSocket) => {
      clearHeartbeat();
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'heartbeat' }));
        }
      }, HEARTBEAT_INTERVAL);
    },
    [clearHeartbeat]
  );

  // Store connect function in a ref to allow self-reference for reconnection
  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (!sessionId || !gameId) {
      console.warn('Cannot connect: missing sessionId or gameId');
      return;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    isManualDisconnectRef.current = false;
    setConnectionState('connecting');

    const params = new URLSearchParams({
      session_id: sessionId,
      ...(playerId !== undefined && { player_id: String(playerId) }),
    });

    const ws = new WebSocket(`${WS_BASE_URL}/ws/games/${gameId}?${params}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionState('connected');
      setConnected(true);
      reconnectAttemptsRef.current = 0;

      // Flush message queue
      while (messageQueueRef.current.length > 0) {
        const msg = messageQueueRef.current.shift();
        if (msg) ws.send(msg);
      }

      // Start heartbeat
      startHeartbeat(ws);
      onConnect?.();
    };

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);

        // Handle specific message types
        switch (message.type as WSMessageType) {
          case 'identity':
            // Server tells us our player_id
            if (message.data && typeof message.data === 'object') {
              const identityData = message.data as IdentityMessage['data'];
              useSessionStore.getState().setCurrentGame(gameId, identityData.player_id);
            }
            break;
          case 'state_update':
            if (message.data) {
              updateGameState(message.data as Parameters<typeof updateGameState>[0]);
            }
            break;
          case 'error':
            if (message.data && typeof message.data === 'object' && 'message' in message.data) {
              setError((message.data as { message: string }).message);
            }
            break;
          case 'heartbeat_ack':
            // Heartbeat acknowledged - connection is healthy
            break;
        }

        // Call custom handler
        onMessage?.(message);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setConnectionState('disconnected');
      setConnected(false);
      clearHeartbeat();
      onDisconnect?.();

      // Attempt reconnection if not manual disconnect
      if (
        !isManualDisconnectRef.current &&
        event.code !== 1000 && // Normal closure
        reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
      ) {
        setConnectionState('reconnecting');
        reconnectAttemptsRef.current++;
        console.log(
          `Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`
        );
        // Use ref to call connect to avoid circular dependency
        reconnectTimeoutRef.current = setTimeout(() => connectRef.current(), RECONNECT_DELAY);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error');
    };

    wsRef.current = ws;
  }, [
    sessionId,
    gameId,
    playerId,
    setConnected,
    updateGameState,
    setError,
    onMessage,
    onConnect,
    onDisconnect,
    startHeartbeat,
    clearHeartbeat,
  ]);

  // Keep ref in sync with latest connect function
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect');
      wsRef.current = null;
    }
    clearHeartbeat();
    setConnectionState('disconnected');
    setConnected(false);
  }, [clearHeartbeat, setConnected]);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(connect, 100);
  }, [disconnect, connect]);

  const send = useCallback((message: ClientMessage) => {
    const data = JSON.stringify(message);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    } else {
      // Queue message for when connection is restored
      messageQueueRef.current.push(data);
      console.log('Message queued (not connected)');
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (autoConnect && sessionId && gameId) {
      // Defer connection to next tick to avoid state update during effect
      const timeoutId = setTimeout(() => connectRef.current(), 0);
      return () => {
        clearTimeout(timeoutId);
        disconnect();
      };
    }
    return () => {
      disconnect();
    };
  }, [autoConnect, sessionId, gameId, disconnect]);

  return {
    connectionState,
    send,
    connect,
    disconnect,
    reconnect,
  };
}
