import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { renderHook, act } from '@testing-library/react';
import MobileControls from './MobileControls';
import { useMobileControls } from '@/hooks/useMobileControls';

describe('MobileControls', () => {
  const originalInnerWidth = window.innerWidth;

  function setMobileView() {
    Object.defineProperty(window, 'innerWidth', { value: 400, writable: true });
    window.dispatchEvent(new Event('resize'));
  }

  function setDesktopView() {
    Object.defineProperty(window, 'innerWidth', { value: 1024, writable: true });
    window.dispatchEvent(new Event('resize'));
  }

  beforeEach(() => {
    vi.useFakeTimers();
    setMobileView();
    // Allow debounced resize to complete
    vi.advanceTimersByTime(200);
  });

  afterEach(() => {
    vi.useRealTimers();
    Object.defineProperty(window, 'innerWidth', { value: originalInnerWidth, writable: true });
  });

  describe('visibility', () => {
    it('renders on mobile devices', () => {
      render(<MobileControls />);
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });

    it('does not render on desktop', () => {
      setDesktopView();
      vi.advanceTimersByTime(200);

      const { container } = render(<MobileControls />);
      expect(container).toBeEmptyDOMElement();
    });
  });

  describe('tab navigation', () => {
    it('renders all four tabs', () => {
      render(<MobileControls />);

      expect(screen.getByRole('tab', { name: /actions/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /properties/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /players/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /log/i })).toBeInTheDocument();
    });

    it('highlights active tab', () => {
      render(<MobileControls activeTab="properties" />);

      const propertiesTab = screen.getByRole('tab', { name: /properties/i });
      expect(propertiesTab).toHaveAttribute('aria-selected', 'true');
    });

    it('calls onTabChange when tab is clicked', () => {
      const onTabChange = vi.fn();
      render(<MobileControls onTabChange={onTabChange} />);

      fireEvent.click(screen.getByRole('tab', { name: /players/i }));
      expect(onTabChange).toHaveBeenCalledWith('players');
    });
  });

  describe('quick actions', () => {
    it('shows quick actions when it is user turn', () => {
      render(<MobileControls isMyTurn={true} canRoll={true} canEndTurn={true} />);

      expect(screen.getByRole('button', { name: /roll dice/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /end turn/i })).toBeInTheDocument();
    });

    it('hides quick actions when not user turn', () => {
      render(<MobileControls isMyTurn={false} />);

      expect(screen.queryByRole('button', { name: /roll dice/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /end turn/i })).not.toBeInTheDocument();
    });

    it('enables roll button when canRoll is true', () => {
      render(<MobileControls isMyTurn={true} canRoll={true} />);

      const rollButton = screen.getByRole('button', { name: /roll dice/i });
      expect(rollButton).not.toBeDisabled();
    });

    it('disables roll button when canRoll is false', () => {
      render(<MobileControls isMyTurn={true} canRoll={false} />);

      const rollButton = screen.getByRole('button', { name: /roll dice/i });
      expect(rollButton).toBeDisabled();
    });

    it('enables end turn button when canEndTurn is true', () => {
      render(<MobileControls isMyTurn={true} canEndTurn={true} />);

      const endTurnButton = screen.getByRole('button', { name: /end turn/i });
      expect(endTurnButton).not.toBeDisabled();
    });

    it('disables end turn button when canEndTurn is false', () => {
      render(<MobileControls isMyTurn={true} canEndTurn={false} />);

      const endTurnButton = screen.getByRole('button', { name: /end turn/i });
      expect(endTurnButton).toBeDisabled();
    });

    it('calls onRollDice when roll button is clicked', () => {
      const onRollDice = vi.fn();
      render(<MobileControls isMyTurn={true} canRoll={true} onRollDice={onRollDice} />);

      fireEvent.click(screen.getByRole('button', { name: /roll dice/i }));
      expect(onRollDice).toHaveBeenCalled();
    });

    it('calls onEndTurn when end turn button is clicked', () => {
      const onEndTurn = vi.fn();
      render(<MobileControls isMyTurn={true} canEndTurn={true} onEndTurn={onEndTurn} />);

      fireEvent.click(screen.getByRole('button', { name: /end turn/i }));
      expect(onEndTurn).toHaveBeenCalled();
    });

    it('disables all buttons when disabled prop is true', () => {
      render(
        <MobileControls isMyTurn={true} canRoll={true} canEndTurn={true} disabled={true} />
      );

      expect(screen.getByRole('button', { name: /roll dice/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /end turn/i })).toBeDisabled();
    });
  });

  describe('badge notifications', () => {
    it('shows badge on log tab when there are unread events', () => {
      render(<MobileControls unreadEvents={5} />);

      expect(screen.getByText('5')).toBeInTheDocument();
    });

    it('shows 99+ for large numbers', () => {
      render(<MobileControls unreadEvents={150} />);

      expect(screen.getByText('99+')).toBeInTheDocument();
    });

    it('does not show badge when unreadEvents is 0', () => {
      render(<MobileControls unreadEvents={0} />);

      expect(screen.queryByLabelText(/unread/i)).not.toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has navigation role', () => {
      render(<MobileControls />);

      expect(screen.getByRole('navigation')).toHaveAttribute('aria-label', 'Game controls');
    });

    it('tabs have proper ARIA attributes', () => {
      render(<MobileControls activeTab="actions" />);

      const tabs = screen.getAllByRole('tab');
      expect(tabs).toHaveLength(4);

      const actionsTab = screen.getByRole('tab', { name: /actions/i });
      expect(actionsTab).toHaveAttribute('aria-selected', 'true');
    });
  });
});

describe('useMobileControls', () => {
  it('has initial state', () => {
    const { result } = renderHook(() => useMobileControls());

    expect(result.current.activeTab).toBe('actions');
    expect(result.current.isPanelOpen).toBe(false);
  });

  it('opens panel with correct tab', () => {
    const { result } = renderHook(() => useMobileControls());

    act(() => {
      result.current.openPanel('properties');
    });

    expect(result.current.activeTab).toBe('properties');
    expect(result.current.isPanelOpen).toBe(true);
  });

  it('closes panel', () => {
    const { result } = renderHook(() => useMobileControls());

    act(() => {
      result.current.openPanel('players');
    });

    act(() => {
      result.current.closePanel();
    });

    expect(result.current.isPanelOpen).toBe(false);
    expect(result.current.activeTab).toBe('players');
  });

  it('toggles panel on same tab click', () => {
    const { result } = renderHook(() => useMobileControls());

    // Open panel
    act(() => {
      result.current.handleTabChange('log');
    });

    expect(result.current.isPanelOpen).toBe(true);
    expect(result.current.activeTab).toBe('log');

    // Click same tab again - should close
    act(() => {
      result.current.handleTabChange('log');
    });

    expect(result.current.isPanelOpen).toBe(false);
  });

  it('switches tabs when panel is open', () => {
    const { result } = renderHook(() => useMobileControls());

    act(() => {
      result.current.openPanel('actions');
    });

    act(() => {
      result.current.handleTabChange('properties');
    });

    expect(result.current.activeTab).toBe('properties');
    expect(result.current.isPanelOpen).toBe(true);
  });
});
