import type { ConnectionState } from '@/types';

interface ConnectionStatusProps {
  state: ConnectionState;
  onReconnect?: () => void;
}

/**
 * Visual indicator for WebSocket connection status.
 */
export function ConnectionStatus({ state, onReconnect }: ConnectionStatusProps) {
  const getStatusColor = () => {
    switch (state) {
      case 'connected':
        return 'bg-green-500';
      case 'connecting':
        return 'bg-yellow-500 animate-pulse';
      case 'reconnecting':
        return 'bg-orange-500 animate-pulse';
      case 'disconnected':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (state) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return 'Reconnecting...';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="flex items-center gap-2 text-sm">
      <div
        className={`w-2 h-2 rounded-full ${getStatusColor()}`}
        title={getStatusText()}
        aria-label={`Connection status: ${getStatusText()}`}
      />
      <span className="text-gray-600">{getStatusText()}</span>
      {state === 'disconnected' && onReconnect && (
        <button
          onClick={onReconnect}
          className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Reconnect
        </button>
      )}
    </div>
  );
}
