import { useEffect, useCallback, useRef } from 'react';

export type KeyboardShortcut = {
  /** The key to listen for (e.g., 'r', 'Enter', 'Escape') */
  key: string;
  /** Optional modifier keys */
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };
  /** Callback when the shortcut is triggered */
  action: () => void;
  /** Whether the shortcut is currently enabled */
  enabled?: boolean;
  /** Description for accessibility */
  description?: string;
};

export interface UseKeyboardNavigationOptions {
  /** Shortcuts to register */
  shortcuts: KeyboardShortcut[];
  /** Whether keyboard navigation is enabled globally */
  enabled?: boolean;
  /** Whether to prevent default behavior on matched shortcuts */
  preventDefault?: boolean;
  /** Element to attach listeners to (defaults to window) */
  target?: HTMLElement | Window | null;
}

/**
 * Hook for keyboard navigation and shortcuts.
 * Handles keyboard events and triggers corresponding actions.
 *
 * @example
 * ```tsx
 * function GameControls({ onRollDice, onEndTurn }) {
 *   useKeyboardNavigation({
 *     shortcuts: [
 *       { key: 'r', action: onRollDice, description: 'Roll dice' },
 *       { key: 'e', action: onEndTurn, description: 'End turn' },
 *       { key: 'Escape', action: () => setModalOpen(false) },
 *     ],
 *   });
 *
 *   return <div>...</div>;
 * }
 * ```
 */
export function useKeyboardNavigation({
  shortcuts,
  enabled = true,
  preventDefault = true,
  target = typeof window !== 'undefined' ? window : null,
}: UseKeyboardNavigationOptions): void {
  // Use refs to avoid stale closures
  const shortcutsRef = useRef(shortcuts);
  shortcutsRef.current = shortcuts;

  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabledRef.current) return;

      // Don't trigger shortcuts when typing in input fields
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return;
      }

      for (const shortcut of shortcutsRef.current) {
        if (shortcut.enabled === false) continue;

        const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const modifiersMatch = checkModifiers(event, shortcut.modifiers);

        if (keyMatches && modifiersMatch) {
          if (preventDefault) {
            event.preventDefault();
          }
          shortcut.action();
          return;
        }
      }
    },
    [preventDefault]
  );

  useEffect(() => {
    if (!target) return;

    target.addEventListener('keydown', handleKeyDown as EventListener);

    return () => {
      target.removeEventListener('keydown', handleKeyDown as EventListener);
    };
  }, [target, handleKeyDown]);
}

function checkModifiers(
  event: KeyboardEvent,
  modifiers?: KeyboardShortcut['modifiers']
): boolean {
  if (!modifiers) {
    // If no modifiers specified, require none to be pressed
    return !event.ctrlKey && !event.altKey && !event.shiftKey && !event.metaKey;
  }

  const ctrlMatch = modifiers.ctrl ? event.ctrlKey : !event.ctrlKey;
  const altMatch = modifiers.alt ? event.altKey : !event.altKey;
  const shiftMatch = modifiers.shift ? event.shiftKey : !event.shiftKey;
  const metaMatch = modifiers.meta ? event.metaKey : !event.metaKey;

  return ctrlMatch && altMatch && shiftMatch && metaMatch;
}

/**
 * Standard game keyboard shortcuts.
 * Use this with useKeyboardNavigation for consistent controls.
 */
export const GAME_SHORTCUTS = {
  ROLL_DICE: 'r',
  END_TURN: 'e',
  BUY_PROPERTY: 'b',
  BUILD_HOUSE: 'h',
  MORTGAGE: 'm',
  TRADE: 't',
  ESCAPE: 'Escape',
  CONFIRM: 'Enter',
  HELP: '?',
} as const;

/**
 * Creates a standard set of game shortcuts.
 * Helper function to quickly set up common game controls.
 */
export function createGameShortcuts(handlers: {
  onRollDice?: () => void;
  onEndTurn?: () => void;
  onBuyProperty?: () => void;
  onBuildHouse?: () => void;
  onMortgage?: () => void;
  onTrade?: () => void;
  onEscape?: () => void;
  onConfirm?: () => void;
  onHelp?: () => void;
  canRoll?: boolean;
  canEndTurn?: boolean;
  canBuy?: boolean;
  canBuild?: boolean;
  canMortgage?: boolean;
  canTrade?: boolean;
}): KeyboardShortcut[] {
  const shortcuts: KeyboardShortcut[] = [];

  if (handlers.onRollDice) {
    shortcuts.push({
      key: GAME_SHORTCUTS.ROLL_DICE,
      action: handlers.onRollDice,
      enabled: handlers.canRoll !== false,
      description: 'Roll dice',
    });
  }

  if (handlers.onEndTurn) {
    shortcuts.push({
      key: GAME_SHORTCUTS.END_TURN,
      action: handlers.onEndTurn,
      enabled: handlers.canEndTurn !== false,
      description: 'End turn',
    });
  }

  if (handlers.onBuyProperty) {
    shortcuts.push({
      key: GAME_SHORTCUTS.BUY_PROPERTY,
      action: handlers.onBuyProperty,
      enabled: handlers.canBuy !== false,
      description: 'Buy property',
    });
  }

  if (handlers.onBuildHouse) {
    shortcuts.push({
      key: GAME_SHORTCUTS.BUILD_HOUSE,
      action: handlers.onBuildHouse,
      enabled: handlers.canBuild !== false,
      description: 'Build house',
    });
  }

  if (handlers.onMortgage) {
    shortcuts.push({
      key: GAME_SHORTCUTS.MORTGAGE,
      action: handlers.onMortgage,
      enabled: handlers.canMortgage !== false,
      description: 'Mortgage property',
    });
  }

  if (handlers.onTrade) {
    shortcuts.push({
      key: GAME_SHORTCUTS.TRADE,
      action: handlers.onTrade,
      enabled: handlers.canTrade !== false,
      description: 'Open trade',
    });
  }

  if (handlers.onEscape) {
    shortcuts.push({
      key: GAME_SHORTCUTS.ESCAPE,
      action: handlers.onEscape,
      description: 'Close/Cancel',
    });
  }

  if (handlers.onConfirm) {
    shortcuts.push({
      key: GAME_SHORTCUTS.CONFIRM,
      action: handlers.onConfirm,
      description: 'Confirm',
    });
  }

  if (handlers.onHelp) {
    shortcuts.push({
      key: GAME_SHORTCUTS.HELP,
      modifiers: { shift: true },
      action: handlers.onHelp,
      description: 'Show keyboard shortcuts',
    });
  }

  return shortcuts;
}

export default useKeyboardNavigation;
