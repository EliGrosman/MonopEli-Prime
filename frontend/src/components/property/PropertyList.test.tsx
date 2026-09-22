import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PropertyList, PropertySummary } from './PropertyList';
import { useGameStore } from '@/store/gameStore';
import type { GameState, PlayerState, PropertyState } from '@/types';

// Mock the game store
vi.mock('@/store/gameStore', () => ({
  useGameStore: vi.fn(),
}));

const mockUseGameStore = vi.mocked(useGameStore);

describe('PropertyList', () => {
  const mockPlayers: PlayerState[] = [
    {
      id: 0,
      name: 'Player 1',
      color: '#FF0000',
      position: 0,
      money: 1500,
      jailTurns: 0,
      bankrupt: false,
      jailCards: 0,
      inJail: false,
      isAi: false,
    },
    {
      id: 1,
      name: 'Player 2',
      color: '#0000FF',
      position: 10,
      money: 1500,
      jailTurns: 0,
      bankrupt: false,
      jailCards: 0,
      inJail: false,
      isAi: false,
    },
  ];

  const mockProperties: Record<number, PropertyState> = {
    1: { position: 1, owner: 0, houses: 0, mortgaged: false }, // Brown
    3: { position: 3, owner: 0, houses: 0, mortgaged: false }, // Brown
    6: { position: 6, owner: 0, houses: 2, mortgaged: false }, // Light blue
    8: { position: 8, owner: 1, houses: 0, mortgaged: false }, // Light blue
    9: { position: 9, owner: 0, houses: 0, mortgaged: false }, // Light blue
  };

  const mockGameState: GameState = {
    revision: 0,
    players: mockPlayers,
    properties: mockProperties,
    currentPlayer: 0,
    turnNumber: 1,
    gamePhase: 'pre_roll' as const,
    lastRoll: null,
    doublesCount: 0,
    housesRemaining: 32,
    hotelsRemaining: 12,
    gameOver: false,
    winner: null,
    rolledDoubles: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseGameStore.mockReturnValue({
      gameState: mockGameState,
      hasMonopoly: (playerId: number, color: string) => {
        // Player 0 has monopoly on brown (positions 1, 3)
        if (playerId === 0 && color === 'brown') return true;
        return false;
      },
      getPlayerProperties: (playerId: number) => {
        return Object.values(mockProperties).filter((p) => p.owner === playerId);
      },
    } as ReturnType<typeof useGameStore>);
  });

  it('renders message when no properties', () => {
    mockUseGameStore.mockReturnValue({
      gameState: mockGameState,
      hasMonopoly: () => false,
      getPlayerProperties: () => [],
    } as ReturnType<typeof useGameStore>);

    render(<PropertyList playerId={1} />);
    expect(screen.getByText('No properties owned')).toBeInTheDocument();
  });

  it('renders nothing when no game state', () => {
    mockUseGameStore.mockReturnValue({
      gameState: null,
      hasMonopoly: () => false,
      getPlayerProperties: () => [],
    } as ReturnType<typeof useGameStore>);

    render(<PropertyList playerId={0} />);
    expect(screen.getByText('No properties owned')).toBeInTheDocument();
  });

  it('groups properties by color', () => {
    render(<PropertyList playerId={0} />);
    // Should have brown group header
    expect(screen.getByText('brown')).toBeInTheDocument();
    // Should have lightblue group header
    expect(screen.getByText('lightblue')).toBeInTheDocument();
  });

  it('shows property count in group header', () => {
    render(<PropertyList playerId={0} />);
    // Brown has 2 of 2 properties
    expect(screen.getByText('(2/2)')).toBeInTheDocument();
    // Light blue has 2 of 3 properties
    expect(screen.getByText('(2/3)')).toBeInTheDocument();
  });

  it('shows monopoly indicator when player has monopoly', () => {
    render(<PropertyList playerId={0} />);
    expect(screen.getByText('Monopoly')).toBeInTheDocument();
  });

  it('hides monopoly indicator when showMonopolyIndicator is false', () => {
    render(<PropertyList playerId={0} showMonopolyIndicator={false} />);
    expect(screen.queryByText('Monopoly')).not.toBeInTheDocument();
  });

  it('shows summary statistics', () => {
    render(<PropertyList playerId={0} />);
    expect(screen.getByText('Total Properties')).toBeInTheDocument();
    expect(screen.getByText('Monopolies')).toBeInTheDocument();
  });

  it('calls onPropertyClick when property is clicked', () => {
    const onPropertyClick = vi.fn();
    render(<PropertyList playerId={0} onPropertyClick={onPropertyClick} />);

    // Click on a property card
    const propertyCards = screen.getAllByRole('button');
    fireEvent.click(propertyCards[0]);
    expect(onPropertyClick).toHaveBeenCalled();
  });
});

describe('PropertySummary', () => {
  const mockPlayers: PlayerState[] = [
    {
      id: 0,
      name: 'Player 1',
      color: '#FF0000',
      position: 0,
      money: 1500,
      jailTurns: 0,
      bankrupt: false,
      jailCards: 0,
      inJail: false,
      isAi: false,
    },
  ];

  const mockProperties: Record<number, PropertyState> = {
    1: { position: 1, owner: 0, houses: 2, mortgaged: false },
    3: { position: 3, owner: 0, houses: 1, mortgaged: false },
    6: { position: 6, owner: 0, houses: 5, mortgaged: false }, // Hotel
    8: { position: 8, owner: 0, houses: 0, mortgaged: false },
  };

  const mockGameState: GameState = {
    revision: 0,
    players: mockPlayers,
    properties: mockProperties,
    currentPlayer: 0,
    turnNumber: 1,
    gamePhase: 'pre_roll' as const,
    lastRoll: null,
    doublesCount: 0,
    housesRemaining: 32,
    hotelsRemaining: 12,
    gameOver: false,
    winner: null,
    rolledDoubles: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseGameStore.mockReturnValue({
      gameState: mockGameState,
      hasMonopoly: (playerId: number, color: string) => {
        if (playerId === 0 && color === 'brown') return true;
        return false;
      },
      getPlayerProperties: (playerId: number) => {
        return Object.values(mockProperties).filter((p) => p.owner === playerId);
      },
    } as ReturnType<typeof useGameStore>);
  });

  it('shows property count', () => {
    render(<PropertySummary playerId={0} />);
    expect(screen.getByText('4 props')).toBeInTheDocument();
  });

  it('shows monopoly count', () => {
    render(<PropertySummary playerId={0} />);
    expect(screen.getByText('1 mono')).toBeInTheDocument();
  });

  it('shows house count', () => {
    render(<PropertySummary playerId={0} />);
    // 2 + 1 = 3 houses (hotel doesn't count as houses)
    expect(screen.getByText('3 houses')).toBeInTheDocument();
  });

  it('shows hotel count', () => {
    render(<PropertySummary playerId={0} />);
    expect(screen.getByText('1 hotels')).toBeInTheDocument();
  });

  it('hides monopolies section when zero', () => {
    mockUseGameStore.mockReturnValue({
      gameState: mockGameState,
      hasMonopoly: () => false,
      getPlayerProperties: (playerId: number) => {
        return Object.values(mockProperties).filter((p) => p.owner === playerId);
      },
    } as ReturnType<typeof useGameStore>);

    render(<PropertySummary playerId={0} />);
    expect(screen.queryByText(/mono/)).not.toBeInTheDocument();
  });
});
