import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PropertyModal } from './PropertyModal';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { useActions } from '@/hooks/useActions';
import type { GameState, PlayerState, PropertyState } from '@/types';

// Mock the stores and hooks
vi.mock('@/store/gameStore', () => ({
  useGameStore: vi.fn(),
}));

vi.mock('@/store/sessionStore', () => ({
  useSessionStore: vi.fn(),
}));

vi.mock('@/hooks/useActions', () => ({
  useActions: vi.fn(),
}));

const mockUseGameStore = vi.mocked(useGameStore);
const mockUseSessionStore = vi.mocked(useSessionStore);
const mockUseActions = vi.mocked(useActions);

describe('PropertyModal', () => {
  const mockSend = vi.fn();

  const mockPlayers: PlayerState[] = [
    { id: 0, name: 'Player 1', color: '#FF0000', position: 0, money: 1500, jailTurns: 0, bankrupt: false, getOutOfJailCards: 0 },
    { id: 1, name: 'Player 2', color: '#0000FF', position: 10, money: 1500, jailTurns: 0, bankrupt: false, getOutOfJailCards: 0 },
  ];

  const mockProperties: Record<number, PropertyState> = {
    1: { position: 1, owner: 0, houses: 0, mortgaged: false },
    3: { position: 3, owner: 0, houses: 0, mortgaged: false },
    6: { position: 6, owner: 1, houses: 0, mortgaged: false },
  };

  const mockGameState: GameState = {
    players: mockPlayers,
    properties: mockProperties,
    currentPlayer: 0,
    phase: 'playing',
    dice: [1, 2],
    doublesCount: 0,
    housesRemaining: 32,
    hotelsRemaining: 12,
    gameOver: false,
    winner: null,
    turnCount: 1,
  };

  const mockActions = {
    buildHouse: vi.fn(),
    buildHotel: vi.fn(),
    sellHouse: vi.fn(),
    sellHotel: vi.fn(),
    mortgageProperty: vi.fn(),
    unmortgageProperty: vi.fn(),
    isActionPending: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();

    mockUseGameStore.mockReturnValue({
      gameState: mockGameState,
      hasMonopoly: (playerId: number, color: string) => {
        if (playerId === 0 && color === 'brown') return true;
        return false;
      },
    } as ReturnType<typeof useGameStore>);

    mockUseSessionStore.mockReturnValue({
      playerId: 0,
    } as ReturnType<typeof useSessionStore>);

    mockUseActions.mockReturnValue(mockActions as ReturnType<typeof useActions>);
  });

  it('returns null when position is null', () => {
    const { container } = render(
      <PropertyModal position={null} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('returns null when no game state', () => {
    mockUseGameStore.mockReturnValue({
      gameState: null,
      hasMonopoly: () => false,
    } as ReturnType<typeof useGameStore>);

    const { container } = render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders property name as modal title', () => {
    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    // Property name appears in title, card, and color group section
    const elements = screen.getAllByText('Mediterranean Avenue');
    expect(elements.length).toBeGreaterThanOrEqual(1);
    // Check the modal title specifically
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Mediterranean Avenue');
  });

  it('shows ownership section', () => {
    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('Ownership')).toBeInTheDocument();
    // Player 1 appears multiple times (owner, color group)
    const playerElements = screen.getAllByText('Player 1');
    expect(playerElements.length).toBeGreaterThanOrEqual(1);
    // (You) is in the same span as player name
    expect(screen.getByText(/\(You\)/)).toBeInTheDocument();
  });

  it('shows unowned status for unowned property', () => {
    const propsWithUnowned = {
      ...mockProperties,
      11: { position: 11, owner: null, houses: 0, mortgaged: false } as PropertyState,
    };

    mockUseGameStore.mockReturnValue({
      gameState: { ...mockGameState, properties: propsWithUnowned },
      hasMonopoly: () => false,
    } as ReturnType<typeof useGameStore>);

    render(
      <PropertyModal position={11} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('Not owned')).toBeInTheDocument();
  });

  it('shows color group section for properties', () => {
    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('Brown Group')).toBeInTheDocument();
  });

  it('shows monopoly message when player has monopoly', () => {
    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('You have a monopoly! Double rent applies.')).toBeInTheDocument();
  });

  it('shows development section for owned properties', () => {
    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('Development')).toBeInTheDocument();
    expect(screen.getByText('No buildings')).toBeInTheDocument();
  });

  it('shows house indicators when houses built', () => {
    const propsWithHouses = {
      ...mockProperties,
      1: { ...mockProperties[1], houses: 3 },
      3: { ...mockProperties[3], houses: 3 },
    };

    mockUseGameStore.mockReturnValue({
      gameState: { ...mockGameState, properties: propsWithHouses },
      hasMonopoly: () => true,
    } as ReturnType<typeof useGameStore>);

    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    const houses = screen.getAllByTitle('House');
    expect(houses).toHaveLength(3);
  });

  it('shows hotel indicator when 5 houses', () => {
    const propsWithHotel = {
      ...mockProperties,
      1: { ...mockProperties[1], houses: 5 },
      3: { ...mockProperties[3], houses: 5 },
    };

    mockUseGameStore.mockReturnValue({
      gameState: { ...mockGameState, properties: propsWithHotel },
      hasMonopoly: () => true,
    } as ReturnType<typeof useGameStore>);

    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByTitle('Hotel')).toBeInTheDocument();
  });

  it('shows mortgage status when mortgaged', () => {
    const propsWithMortgage = {
      ...mockProperties,
      1: { ...mockProperties[1], mortgaged: true },
    };

    mockUseGameStore.mockReturnValue({
      gameState: { ...mockGameState, properties: propsWithMortgage },
      hasMonopoly: () => false,
    } as ReturnType<typeof useGameStore>);

    render(
      <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
    );
    expect(screen.getByText('This property is mortgaged')).toBeInTheDocument();
  });

  describe('action buttons', () => {
    it('shows build button when can build', () => {
      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      // Player has monopoly on brown, no houses yet, can build
      expect(screen.getByRole('button', { name: /Build House/ })).toBeInTheDocument();
    });

    it('shows mortgage button when can mortgage', () => {
      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      expect(screen.getByRole('button', { name: /Mortgage/ })).toBeInTheDocument();
    });

    it('does not show buttons for properties not owned by player', () => {
      render(
        <PropertyModal position={6} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      expect(screen.queryByRole('button', { name: /Build/ })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Mortgage/ })).not.toBeInTheDocument();
    });

    it('calls buildHouse when build house button clicked', () => {
      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      fireEvent.click(screen.getByRole('button', { name: /Build House/ }));
      expect(mockActions.buildHouse).toHaveBeenCalledWith(1);
    });

    it('calls buildHotel when at 4 houses', () => {
      const propsWith4Houses = {
        ...mockProperties,
        1: { ...mockProperties[1], houses: 4 },
        3: { ...mockProperties[3], houses: 4 },
      };

      mockUseGameStore.mockReturnValue({
        gameState: { ...mockGameState, properties: propsWith4Houses },
        hasMonopoly: () => true,
      } as ReturnType<typeof useGameStore>);

      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      fireEvent.click(screen.getByRole('button', { name: /Build Hotel/ }));
      expect(mockActions.buildHotel).toHaveBeenCalledWith(1);
    });

    it('calls mortgageProperty when mortgage button clicked', () => {
      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      fireEvent.click(screen.getByRole('button', { name: /Mortgage/ }));
      expect(mockActions.mortgageProperty).toHaveBeenCalledWith(1);
    });

    it('disables buttons when action is pending', () => {
      mockUseActions.mockReturnValue({
        ...mockActions,
        isActionPending: true,
      } as ReturnType<typeof useActions>);

      render(
        <PropertyModal position={1} isOpen={true} onClose={() => {}} send={mockSend} />
      );
      expect(screen.getByRole('button', { name: /Build House/ })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Mortgage/ })).toBeDisabled();
    });
  });

  it('calls onClose when modal is closed', () => {
    const onClose = vi.fn();
    render(
      <PropertyModal position={1} isOpen={true} onClose={onClose} send={mockSend} />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Close modal' }));
    expect(onClose).toHaveBeenCalled();
  });
});
