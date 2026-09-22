import { act, render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DiceRoll, DiceDisplay } from './DiceRoll';
import type { DiceRoll as DiceRollType } from '@/types';

describe('DiceRoll', () => {
  const mockRoll: DiceRollType = {
    die1: 3,
    die2: 4,
    isDoubles: false,
  };

  const doublesRoll: DiceRollType = {
    die1: 5,
    die2: 5,
    isDoubles: true,
  };

  it('renders dice when roll is provided', () => {
    render(<DiceRoll roll={mockRoll} />);
    // Should show total
    expect(screen.getByText('7')).toBeInTheDocument();
  });

  it('shows Doubles indicator when doubles are rolled', () => {
    render(<DiceRoll roll={doublesRoll} />);
    expect(screen.getByText('Doubles!')).toBeInTheDocument();
  });

  it('shows roll button when canRoll is true', () => {
    render(<DiceRoll roll={null} canRoll={true} onRoll={() => {}} />);
    expect(screen.getByText('Roll Dice')).toBeInTheDocument();
  });

  it('hides roll button when canRoll is false', () => {
    render(<DiceRoll roll={null} canRoll={false} />);
    expect(screen.queryByText('Roll Dice')).not.toBeInTheDocument();
  });

  it('calls onRoll when roll button is clicked', () => {
    const onRoll = vi.fn();
    render(<DiceRoll roll={null} canRoll={true} onRoll={onRoll} />);

    fireEvent.click(screen.getByText('Roll Dice'));
    expect(onRoll).toHaveBeenCalled();
  });

  it('disables roll button when isRolling is true', () => {
    render(<DiceRoll roll={null} canRoll={true} onRoll={() => {}} isRolling={true} />);
    const button = screen.getByText('Rolling...');
    expect(button).toBeDisabled();
  });

  it('finishes an active animation when refreshed with the same roll', () => {
    vi.useFakeTimers();
    const onRoll = vi.fn();
    const { rerender } = render(<DiceRoll roll={doublesRoll} canRoll={true} onRoll={onRoll} />);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByRole('button', { name: 'Roll Dice' })).toBeDisabled();

    rerender(<DiceRoll roll={{ ...doublesRoll }} canRoll={true} onRoll={onRoll} />);
    act(() => {
      vi.runOnlyPendingTimers();
    });

    expect(screen.getByRole('button', { name: 'Roll Dice' })).toBeEnabled();
    vi.useRealTimers();
  });

  it('shows question marks when no roll', () => {
    render(<DiceRoll roll={null} canRoll={false} />);
    // Question marks are shown for empty dice
    expect(screen.getAllByText('?')).toHaveLength(2);
  });
});

describe('DiceDisplay', () => {
  it('renders nothing when roll is null', () => {
    const { container } = render(<DiceDisplay roll={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows dice values and total', () => {
    const roll: DiceRollType = { die1: 2, die2: 6, isDoubles: false };
    render(<DiceDisplay roll={roll} />);

    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
  });

  it('shows Doubles badge when doubles rolled', () => {
    const roll: DiceRollType = { die1: 3, die2: 3, isDoubles: true };
    render(<DiceDisplay roll={roll} />);

    expect(screen.getByText('Doubles')).toBeInTheDocument();
  });
});
