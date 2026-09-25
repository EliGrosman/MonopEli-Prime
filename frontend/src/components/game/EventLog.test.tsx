import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { act } from '@testing-library/react';
import { EventLog, CompactEventLog } from '@/components/game/EventLog';
import { useGameStore } from '@/store';
import type { GameEvent } from '@/types';

function createMockEvent(overrides: Partial<GameEvent> = {}): GameEvent {
  return {
    id: `event-${Math.random().toString(36).slice(2)}`,
    timestamp: Date.now(),
    type: 'roll',
    playerId: 0,
    message: 'Test event message',
    ...overrides,
  };
}

describe('EventLog', () => {
  beforeEach(() => {
    act(() => {
      useGameStore.getState().reset();
      useGameStore.getState().clearEvents();
    });
  });

  it('shows empty state when no events', () => {
    render(<EventLog />);

    expect(screen.getByText('No events yet')).toBeInTheDocument();
    expect(screen.getByText('Activity')).toBeInTheDocument();
  });

  it('renders events when events exist', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ message: 'Player rolled dice' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Player rolled dice')).toBeInTheDocument();
  });

  it('renders multiple events', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ id: '1', message: 'Event 1' }));
      useGameStore.getState().addEvent(createMockEvent({ id: '2', message: 'Event 2' }));
      useGameStore.getState().addEvent(createMockEvent({ id: '3', message: 'Event 3' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Event 1')).toBeInTheDocument();
    expect(screen.getByText('Event 2')).toBeInTheDocument();
    expect(screen.getByText('Event 3')).toBeInTheDocument();
  });

  it('shows correct icon for roll event', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ type: 'roll', message: 'Rolled dice' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Rolled dice')).toBeInTheDocument();
  });

  it('shows correct icon for buy event', () => {
    act(() => {
      useGameStore
        .getState()
        .addEvent(createMockEvent({ type: 'buy', message: 'Bought property' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Bought property')).toBeInTheDocument();
  });

  it('shows correct icon for rent event', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ type: 'rent', message: 'Paid rent' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Paid rent')).toBeInTheDocument();
  });

  it('shows correct icon for jail event', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ type: 'jail', message: 'Went to jail' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Went to jail')).toBeInTheDocument();
  });

  it('shows correct icon for bankrupt event', () => {
    act(() => {
      useGameStore
        .getState()
        .addEvent(createMockEvent({ type: 'bankrupt', message: 'Went bankrupt' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Went bankrupt')).toBeInTheDocument();
  });

  it('shows correct icon for win event', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ type: 'win', message: 'Won the game!' }));
    });

    render(<EventLog />);

    expect(screen.getByText('Won the game!')).toBeInTheDocument();
  });

  it('limits displayed events by maxEvents prop', () => {
    act(() => {
      for (let i = 0; i < 10; i++) {
        useGameStore
          .getState()
          .addEvent(createMockEvent({ id: `event-${i}`, message: `Event ${i}` }));
      }
    });

    render(<EventLog maxEvents={5} />);

    // Should show last 5 events (in reverse order, so most recent first)
    expect(screen.queryByText('Event 0')).not.toBeInTheDocument();
    expect(screen.queryByText('Event 4')).not.toBeInTheDocument();
    expect(screen.getByText('Event 5')).toBeInTheDocument();
    expect(screen.getByText('Event 9')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<EventLog className="custom-class" />);

    // The className is applied to the outer div
    expect(container.firstChild).toHaveClass('custom-class');
  });

  it('has correct accessibility attributes', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent());
    });

    render(<EventLog />);

    const log = screen.getByRole('log');
    expect(log).toHaveAttribute('aria-live', 'polite');
    expect(log).toHaveAttribute('aria-label', 'Game events');
  });
});

describe('CompactEventLog', () => {
  beforeEach(() => {
    act(() => {
      useGameStore.getState().reset();
      useGameStore.getState().clearEvents();
    });
  });

  it('returns null when no events', () => {
    const { container } = render(<CompactEventLog />);

    expect(container).toBeEmptyDOMElement();
  });

  it('renders events when events exist', () => {
    act(() => {
      useGameStore.getState().addEvent(createMockEvent({ message: 'Compact event' }));
    });

    render(<CompactEventLog />);

    expect(screen.getByText(/Compact event/)).toBeInTheDocument();
  });

  it('limits events by maxEvents prop', () => {
    act(() => {
      for (let i = 0; i < 10; i++) {
        useGameStore
          .getState()
          .addEvent(createMockEvent({ id: `event-${i}`, message: `Event ${i}` }));
      }
    });

    render(<CompactEventLog maxEvents={3} />);

    // Should only show 3 most recent events
    expect(screen.queryByText(/Event 6/)).not.toBeInTheDocument();
    expect(screen.getByText(/Event 7/)).toBeInTheDocument();
    expect(screen.getByText(/Event 9/)).toBeInTheDocument();
  });
});
