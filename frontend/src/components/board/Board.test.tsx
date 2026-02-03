import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Board } from './Board';
import type { PlayerState, PropertyState } from '@/types';

// Mock players
const mockPlayers: PlayerState[] = [
  {
    id: 0,
    name: 'Alice',
    money: 1500,
    position: 0,
    inJail: false,
    jailTurns: 0,
    jailCards: 0,
    bankrupt: false,
    isAi: false,
    color: '#E53935',
  },
  {
    id: 1,
    name: 'Bob',
    money: 1500,
    position: 5,
    inJail: false,
    jailTurns: 0,
    jailCards: 0,
    bankrupt: false,
    isAi: false,
    color: '#1E88E5',
  },
];

// Mock properties
const mockProperties: Record<number, PropertyState> = {
  1: { position: 1, owner: 0, houses: 2, mortgaged: false },
  3: { position: 3, owner: 0, houses: 0, mortgaged: false },
  5: { position: 5, owner: 1, houses: 0, mortgaged: true },
};

describe('Board', () => {
  it('renders the board component', () => {
    render(<Board />);
    expect(screen.getByTestId('board')).toBeInTheDocument();
  });

  it('renders all 40 spaces', () => {
    render(<Board />);

    // Check some specific spaces exist
    expect(screen.getByTestId('space-0')).toBeInTheDocument(); // GO
    expect(screen.getByTestId('space-10')).toBeInTheDocument(); // Jail
    expect(screen.getByTestId('space-20')).toBeInTheDocument(); // Free Parking
    expect(screen.getByTestId('space-30')).toBeInTheDocument(); // Go To Jail
    expect(screen.getByTestId('space-39')).toBeInTheDocument(); // Boardwalk
  });

  it('renders corner spaces with correct content', () => {
    render(<Board />);

    // Check GO space
    const goSpace = screen.getByTestId('space-0');
    expect(goSpace).toHaveTextContent('GO');

    // Check Jail space
    const jailSpace = screen.getByTestId('space-10');
    expect(jailSpace).toHaveTextContent('JAIL');

    // Check Free Parking
    const parkingSpace = screen.getByTestId('space-20');
    expect(parkingSpace).toHaveTextContent('FREE');
    expect(parkingSpace).toHaveTextContent('PARKING');

    // Check Go To Jail
    const goToJailSpace = screen.getByTestId('space-30');
    expect(goToJailSpace).toHaveTextContent('GO TO');
    expect(goToJailSpace).toHaveTextContent('JAIL');
  });

  it('renders player tokens at correct positions', () => {
    render(<Board players={mockPlayers} />);

    // Player 0 at position 0 (GO)
    const token0 = screen.getByTestId('token-player-0');
    expect(token0).toBeInTheDocument();

    // Player 1 at position 5 (Reading Railroad)
    const token1 = screen.getByTestId('token-player-1');
    expect(token1).toBeInTheDocument();
  });

  it('does not render tokens for bankrupt players', () => {
    const playersWithBankrupt: PlayerState[] = [
      ...mockPlayers,
      {
        id: 2,
        name: 'Charlie',
        money: 0,
        position: 10,
        inJail: false,
        jailTurns: 0,
        jailCards: 0,
        bankrupt: true,
        isAi: false,
        color: '#43A047',
      },
    ];

    render(<Board players={playersWithBankrupt} />);

    expect(screen.queryByTestId('token-player-2')).not.toBeInTheDocument();
  });

  it('calls onSpaceClick when a space is clicked', () => {
    const onSpaceClick = vi.fn();
    render(<Board onSpaceClick={onSpaceClick} />);

    fireEvent.click(screen.getByTestId('space-1'));

    expect(onSpaceClick).toHaveBeenCalledWith(1);
  });

  it('highlights specified spaces', () => {
    render(<Board highlightedSpaces={[1, 3, 5]} />);

    const space1 = screen.getByTestId('space-1');
    const space3 = screen.getByTestId('space-3');
    const space5 = screen.getByTestId('space-5');

    // Highlighted spaces should have the highlight animation
    expect(space1.className).toContain('animate-pulse-highlight');
    expect(space3.className).toContain('animate-pulse-highlight');
    expect(space5.className).toContain('animate-pulse-highlight');
  });

  it('renders the center area with title', () => {
    render(<Board />);

    expect(screen.getByText('MONOPOLY')).toBeInTheDocument();
  });

  it('supports keyboard navigation on spaces', () => {
    const onSpaceClick = vi.fn();
    render(<Board onSpaceClick={onSpaceClick} />);

    const space = screen.getByTestId('space-5');
    fireEvent.keyDown(space, { key: 'Enter' });

    expect(onSpaceClick).toHaveBeenCalledWith(5);
  });

  it('renders property information on property spaces', () => {
    render(<Board />);

    // Check that Mediterranean Avenue shows price
    const space1 = screen.getByTestId('space-1');
    expect(space1).toHaveTextContent('$60');
  });
});
