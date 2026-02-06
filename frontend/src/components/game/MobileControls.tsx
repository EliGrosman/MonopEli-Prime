import { useResponsive } from '@/hooks';
import type { MobileTab } from '@/hooks/useMobileControls';
import type { GameState, PlayerState } from '@/types';

interface MobileControlsProps {
  /** Current game state */
  gameState?: GameState | null;
  /** Current player (the user) */
  currentPlayer?: PlayerState | null;
  /** Whether it's the user's turn */
  isMyTurn?: boolean;
  /** Callback when roll dice is clicked */
  onRollDice?: () => void;
  /** Callback when end turn is clicked */
  onEndTurn?: () => void;
  /** Callback when a tab is selected */
  onTabChange?: (tab: MobileTab) => void;
  /** Which tab is currently active */
  activeTab?: MobileTab;
  /** Number of unread events in log */
  unreadEvents?: number;
  /** Whether dice can be rolled */
  canRoll?: boolean;
  /** Whether turn can be ended */
  canEndTurn?: boolean;
  /** Whether controls are disabled */
  disabled?: boolean;
}

/**
 * Mobile-optimized bottom controls for the game.
 * Only renders on mobile devices (< 640px).
 *
 * Features:
 * - Bottom navigation bar with tab icons
 * - Quick action buttons (roll dice, end turn)
 * - Badge indicators for notifications
 * - Touch-optimized sizing
 */
export function MobileControls({
  isMyTurn = false,
  onRollDice,
  onEndTurn,
  onTabChange,
  activeTab = 'actions',
  unreadEvents = 0,
  canRoll = false,
  canEndTurn = false,
  disabled = false,
}: MobileControlsProps) {
  const { isMobile } = useResponsive();

  // Don't render on non-mobile devices
  if (!isMobile) {
    return null;
  }

  const handleTabClick = (tab: MobileTab) => {
    onTabChange?.(tab);
  };

  return (
    <div
      className="mobile-controls fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg z-40 safe-area-pb"
      role="navigation"
      aria-label="Game controls"
    >
      {/* Quick actions bar */}
      {isMyTurn && (
        <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-100 bg-gray-50">
          <button
            onClick={onRollDice}
            disabled={disabled || !canRoll}
            className={`
              flex-1 py-2 px-4 rounded-lg font-semibold text-sm transition-colors
              ${
                !disabled && canRoll
                  ? 'bg-board-border text-white active:bg-green-800'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }
            `}
            aria-label="Roll dice"
          >
            🎲 Roll
          </button>
          <button
            onClick={onEndTurn}
            disabled={disabled || !canEndTurn}
            className={`
              flex-1 py-2 px-4 rounded-lg font-semibold text-sm transition-colors
              ${
                !disabled && canEndTurn
                  ? 'bg-blue-600 text-white active:bg-blue-700'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }
            `}
            aria-label="End turn"
          >
            ✓ End Turn
          </button>
        </div>
      )}

      {/* Tab bar */}
      <div className="flex items-stretch" role="tablist">
        <TabButton
          icon="⚡"
          label="Actions"
          isActive={activeTab === 'actions'}
          onClick={() => handleTabClick('actions')}
        />
        <TabButton
          icon="🏠"
          label="Properties"
          isActive={activeTab === 'properties'}
          onClick={() => handleTabClick('properties')}
        />
        <TabButton
          icon="👥"
          label="Players"
          isActive={activeTab === 'players'}
          onClick={() => handleTabClick('players')}
        />
        <TabButton
          icon="📜"
          label="Log"
          isActive={activeTab === 'log'}
          onClick={() => handleTabClick('log')}
          badge={unreadEvents > 0 ? unreadEvents : undefined}
        />
      </div>
    </div>
  );
}

interface TabButtonProps {
  icon: string;
  label: string;
  isActive: boolean;
  onClick: () => void;
  badge?: number;
}

function TabButton({ icon, label, isActive, onClick, badge }: TabButtonProps) {
  return (
    <button
      role="tab"
      aria-selected={isActive}
      onClick={onClick}
      className={`
        flex-1 flex flex-col items-center justify-center py-2 relative
        transition-colors min-h-[56px]
        ${isActive ? 'text-board-border bg-green-50' : 'text-gray-500 active:bg-gray-100'}
      `}
    >
      <span className="text-xl" role="img" aria-hidden="true">
        {icon}
      </span>
      <span className="text-[10px] mt-0.5">{label}</span>
      {badge !== undefined && badge > 0 && (
        <span
          className="absolute top-1 right-1/4 min-w-[16px] h-4 px-1 flex items-center justify-center bg-red-500 text-white text-[10px] font-bold rounded-full"
          aria-label={`${badge} unread`}
        >
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  );
}

export default MobileControls;
