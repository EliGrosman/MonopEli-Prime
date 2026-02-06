import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import {
  useKeyboardNavigation,
  createGameShortcuts,
  GAME_SHORTCUTS,
} from './useKeyboardNavigation';

function fireKeyDown(key: string, options: Partial<KeyboardEventInit> = {}) {
  const event = new KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
    ...options,
  });
  window.dispatchEvent(event);
  return event;
}

describe('useKeyboardNavigation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('basic shortcuts', () => {
    it('triggers action on matching key', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      fireKeyDown('r');
      expect(action).toHaveBeenCalledTimes(1);
    });

    it('is case insensitive', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'R', action }],
        })
      );

      fireKeyDown('r');
      expect(action).toHaveBeenCalledTimes(1);
    });

    it('does not trigger on non-matching key', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      fireKeyDown('e');
      expect(action).not.toHaveBeenCalled();
    });

    it('handles multiple shortcuts', () => {
      const rollAction = vi.fn();
      const endAction = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [
            { key: 'r', action: rollAction },
            { key: 'e', action: endAction },
          ],
        })
      );

      fireKeyDown('r');
      expect(rollAction).toHaveBeenCalledTimes(1);
      expect(endAction).not.toHaveBeenCalled();

      fireKeyDown('e');
      expect(endAction).toHaveBeenCalledTimes(1);
    });

    it('handles special keys like Escape', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'Escape', action }],
        })
      );

      fireKeyDown('Escape');
      expect(action).toHaveBeenCalledTimes(1);
    });
  });

  describe('modifier keys', () => {
    it('requires ctrl when specified', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 's', modifiers: { ctrl: true }, action }],
        })
      );

      // Without ctrl
      fireKeyDown('s');
      expect(action).not.toHaveBeenCalled();

      // With ctrl
      fireKeyDown('s', { ctrlKey: true });
      expect(action).toHaveBeenCalledTimes(1);
    });

    it('requires shift when specified', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: '?', modifiers: { shift: true }, action }],
        })
      );

      // Without shift
      fireKeyDown('?');
      expect(action).not.toHaveBeenCalled();

      // With shift
      fireKeyDown('?', { shiftKey: true });
      expect(action).toHaveBeenCalledTimes(1);
    });

    it('requires multiple modifiers', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 's', modifiers: { ctrl: true, shift: true }, action }],
        })
      );

      // Only ctrl
      fireKeyDown('s', { ctrlKey: true });
      expect(action).not.toHaveBeenCalled();

      // Only shift
      fireKeyDown('s', { shiftKey: true });
      expect(action).not.toHaveBeenCalled();

      // Both
      fireKeyDown('s', { ctrlKey: true, shiftKey: true });
      expect(action).toHaveBeenCalledTimes(1);
    });
  });

  describe('enabled state', () => {
    it('does not trigger when globally disabled', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
          enabled: false,
        })
      );

      fireKeyDown('r');
      expect(action).not.toHaveBeenCalled();
    });

    it('does not trigger when shortcut is disabled', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action, enabled: false }],
        })
      );

      fireKeyDown('r');
      expect(action).not.toHaveBeenCalled();
    });

    it('respects enabled changes', () => {
      const action = vi.fn();

      const { rerender } = renderHook(
        ({ enabled }) =>
          useKeyboardNavigation({
            shortcuts: [{ key: 'r', action, enabled }],
          }),
        { initialProps: { enabled: false } }
      );

      fireKeyDown('r');
      expect(action).not.toHaveBeenCalled();

      rerender({ enabled: true });
      fireKeyDown('r');
      expect(action).toHaveBeenCalledTimes(1);
    });
  });

  describe('input field handling', () => {
    it('does not trigger when focused on input', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      // Create and focus an input
      const input = document.createElement('input');
      document.body.appendChild(input);

      const event = new KeyboardEvent('keydown', {
        key: 'r',
        bubbles: true,
      });
      Object.defineProperty(event, 'target', { value: input });
      window.dispatchEvent(event);

      expect(action).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it('does not trigger when focused on textarea', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      const textarea = document.createElement('textarea');
      document.body.appendChild(textarea);

      const event = new KeyboardEvent('keydown', {
        key: 'r',
        bubbles: true,
      });
      Object.defineProperty(event, 'target', { value: textarea });
      window.dispatchEvent(event);

      expect(action).not.toHaveBeenCalled();

      document.body.removeChild(textarea);
    });
  });

  describe('preventDefault', () => {
    it('prevents default by default', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      const preventDefaultSpy = vi.fn();
      const event = new KeyboardEvent('keydown', {
        key: 'r',
        bubbles: true,
        cancelable: true,
      });
      Object.defineProperty(event, 'preventDefault', { value: preventDefaultSpy });
      window.dispatchEvent(event);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('does not prevent default when option is false', () => {
      const action = vi.fn();

      renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
          preventDefault: false,
        })
      );

      const preventDefaultSpy = vi.fn();
      const event = new KeyboardEvent('keydown', {
        key: 'r',
        bubbles: true,
        cancelable: true,
      });
      Object.defineProperty(event, 'preventDefault', { value: preventDefaultSpy });
      window.dispatchEvent(event);

      expect(preventDefaultSpy).not.toHaveBeenCalled();
    });
  });

  describe('cleanup', () => {
    it('removes event listener on unmount', () => {
      const action = vi.fn();
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      const { unmount } = renderHook(() =>
        useKeyboardNavigation({
          shortcuts: [{ key: 'r', action }],
        })
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));
    });
  });
});

describe('createGameShortcuts', () => {
  it('creates roll dice shortcut', () => {
    const onRollDice = vi.fn();
    const shortcuts = createGameShortcuts({ onRollDice });

    expect(shortcuts).toHaveLength(1);
    expect(shortcuts[0].key).toBe(GAME_SHORTCUTS.ROLL_DICE);
    expect(shortcuts[0].description).toBe('Roll dice');
  });

  it('creates multiple shortcuts', () => {
    const handlers = {
      onRollDice: vi.fn(),
      onEndTurn: vi.fn(),
      onBuyProperty: vi.fn(),
    };
    const shortcuts = createGameShortcuts(handlers);

    expect(shortcuts).toHaveLength(3);
    expect(shortcuts.map((s) => s.key)).toContain(GAME_SHORTCUTS.ROLL_DICE);
    expect(shortcuts.map((s) => s.key)).toContain(GAME_SHORTCUTS.END_TURN);
    expect(shortcuts.map((s) => s.key)).toContain(GAME_SHORTCUTS.BUY_PROPERTY);
  });

  it('respects canRoll flag', () => {
    const onRollDice = vi.fn();
    const shortcuts = createGameShortcuts({ onRollDice, canRoll: false });

    expect(shortcuts[0].enabled).toBe(false);
  });

  it('creates help shortcut with shift modifier', () => {
    const onHelp = vi.fn();
    const shortcuts = createGameShortcuts({ onHelp });

    const helpShortcut = shortcuts.find((s) => s.key === GAME_SHORTCUTS.HELP);
    expect(helpShortcut?.modifiers?.shift).toBe(true);
  });
});

describe('GAME_SHORTCUTS', () => {
  it('has correct key bindings', () => {
    expect(GAME_SHORTCUTS.ROLL_DICE).toBe('r');
    expect(GAME_SHORTCUTS.END_TURN).toBe('e');
    expect(GAME_SHORTCUTS.BUY_PROPERTY).toBe('b');
    expect(GAME_SHORTCUTS.BUILD_HOUSE).toBe('h');
    expect(GAME_SHORTCUTS.MORTGAGE).toBe('m');
    expect(GAME_SHORTCUTS.TRADE).toBe('t');
    expect(GAME_SHORTCUTS.ESCAPE).toBe('Escape');
    expect(GAME_SHORTCUTS.CONFIRM).toBe('Enter');
    expect(GAME_SHORTCUTS.HELP).toBe('?');
  });
});
