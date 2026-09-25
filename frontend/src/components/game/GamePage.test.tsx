import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import type { DecisionContract, GameState } from '@/types';
import { useGameStore } from '@/store/gameStore';
import { useSessionStore } from '@/store/sessionStore';
import { GamePage } from './GamePage';

const { mockUseGameState } = vi.hoisted(() => ({
  mockUseGameState: vi.fn(),
}));

vi.mock('@/hooks/useGameState', () => ({
  useGameState: () => mockUseGameState(),
}));

vi.mock('@/components/board/Board', () => ({
  Board: ({
    centerContent,
    onSpaceClick,
  }: {
    centerContent?: ReactNode;
    onSpaceClick?: (position: number) => void;
  }) => (
    <div data-testid="mock-board">
      <div data-testid="mock-board-center">{centerContent}</div>
      <button onClick={() => onSpaceClick?.(1)}>Inspect property</button>
    </div>
  ),
}));

vi.mock('@/components/property', () => ({
  PropertyModal: ({ isOpen }: { isOpen: boolean }) =>
    isOpen ? (
      <div role="dialog" aria-label="Property details">
        Property details
      </div>
    ) : null,
  PropertyList: () => <div>Player holdings</div>,
}));

vi.mock('./PlayerPanel', () => ({
  PlayerPanel: ({ onSelect }: { onSelect: (id: number) => void }) => (
    <aside>
      Players panel<button onClick={() => onSelect(1)}>Opponent details</button>
    </aside>
  ),
}));

vi.mock('./PlayerCard', () => ({
  PlayerCard: ({ player }: { player: { name: string } }) => <div>{player.name} profile</div>,
}));

vi.mock('./EventLog', () => ({
  EventLog: () => <aside>Game log</aside>,
  LatestActivity: ({ onOpenHistory }: { onOpenHistory: () => void }) => (
    <button onClick={onOpenHistory}>Open activity history</button>
  ),
}));

vi.mock('./ActionPanel', () => ({
  ActionPanel: ({ headerAction }: { headerAction?: ReactNode }) => (
    <section aria-label="Game actions">
      <span>Centered actions</span>
      {headerAction}
    </section>
  ),
}));

vi.mock('./TradePanel', () => ({
  TradePanel: () => <section aria-label="Compose trade">Trade panel content</section>,
}));

function makeContract(pendingOffer: DecisionContract['pending_offer'] = null): DecisionContract {
  return {
    contract_version: 'decision-contract-v1',
    rules_id: 'foundation-trade-v1',
    revision: 1,
    viewer_id: 0,
    turn_owner: 0,
    decision_player: 0,
    phase: pendingOffer ? 'trade_response' : 'pre_roll',
    properties: [],
    trade: {
      can_propose: pendingOffer === null,
      proposals_remaining: 2,
      used_recipients: [],
      eligible_recipients: [1],
      tradeable_properties: [],
    },
    pending_offer: pendingOffer,
  };
}

function makeGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    players: [
      {
        id: 0,
        name: 'Player',
        money: 1500,
        position: 0,
        inJail: false,
        jailTurns: 0,
        jailCards: 0,
        bankrupt: false,
        isAi: false,
        color: '#e53935',
      },
      {
        id: 1,
        name: 'Opponent',
        money: 1500,
        position: 0,
        inJail: false,
        jailTurns: 0,
        jailCards: 0,
        bankrupt: false,
        isAi: false,
        color: '#1e88e5',
      },
    ],
    properties: {},
    currentPlayer: 0,
    turnNumber: 1,
    gamePhase: 'pre_roll',
    lastRoll: null,
    doublesCount: 0,
    housesRemaining: 32,
    hotelsRemaining: 12,
    gameOver: false,
    winner: null,
    revision: 1,
    rolledDoubles: false,
    decision_player: 0,
    rules_id: 'foundation-v1',
    legal_actions: [],
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/game/test-game']}>
      <Routes>
        <Route path="/game/:gameId" element={<GamePage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('GamePage board controls', () => {
  beforeEach(() => {
    useSessionStore.setState({ playerId: 0, currentGameId: 'test-game' });
    useGameStore.setState({
      gameState: makeGameState(),
      isConnected: true,
      isLoading: false,
      error: null,
      pendingRequestId: null,
      events: [],
    });
    mockUseGameState.mockReturnValue({
      isLoading: false,
      error: null,
      connectionState: 'connected',
      reconnect: vi.fn(),
      isMyTurn: true,
      send: vi.fn(),
    });
  });

  it('renders actions and the log in the board center while leaving players in the sidebar', () => {
    renderPage();

    expect(screen.getByTestId('mock-board-center')).toHaveTextContent('Centered actions');
    expect(screen.getByTestId('mock-board-center')).toHaveTextContent('Game log');
    expect(screen.getByText('Players panel')).toBeInTheDocument();
    expect(screen.getByText('Game log')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open trading' })).not.toBeInTheDocument();
  });

  it('opens trading from the board-center control', () => {
    const contract = makeContract();
    useGameStore.setState({
      gameState: makeGameState({
        rules_id: 'foundation-trade-v1',
        decision_contract: contract,
      }),
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Open trading' }));

    expect(screen.getByRole('dialog', { name: 'Trade' })).toBeInTheDocument();
    expect(screen.getByText('Trade panel content')).toBeInTheDocument();
  });

  it('automatically opens each incoming offer only once', async () => {
    const firstOffer = {
      trade_id: 41,
      created_revision: 2,
      from_player: 1,
      to_player: 0,
      give_properties: [],
      give_money: 100,
      want_properties: [],
      want_money: 0,
    };
    useGameStore.setState({
      gameState: makeGameState({
        revision: 2,
        gamePhase: 'trade_response',
        rules_id: 'foundation-trade-v1',
        decision_contract: makeContract(firstOffer),
      }),
    });

    renderPage();
    expect(await screen.findByRole('dialog', { name: 'Incoming trade offer' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Close modal' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    act(() => {
      useGameStore.setState((state) => ({
        gameState: state.gameState ? { ...state.gameState, revision: 3 } : null,
      }));
    });
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    act(() => {
      useGameStore.setState({
        gameState: makeGameState({
          revision: 4,
          gamePhase: 'trade_response',
          rules_id: 'foundation-trade-v1',
          decision_contract: makeContract({ ...firstOffer, trade_id: 42 }),
        }),
      });
    });

    expect(await screen.findByRole('dialog', { name: 'Incoming trade offer' })).toBeInTheDocument();
  });

  it('gives a new incoming trade priority over another dialog', async () => {
    const firstOffer = {
      trade_id: 41,
      created_revision: 2,
      from_player: 1,
      to_player: 0,
      give_properties: [],
      give_money: 100,
      want_properties: [],
      want_money: 0,
    };
    useGameStore.setState({
      gameState: makeGameState({
        properties: { 1: { position: 1, owner: null, houses: 0, mortgaged: false } },
      }),
    });
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Inspect property' }));
    expect(screen.getByRole('dialog', { name: 'Property details' })).toBeInTheDocument();

    act(() => {
      useGameStore.setState({
        gameState: makeGameState({
          revision: 2,
          gamePhase: 'trade_response',
          rules_id: 'foundation-trade-v1',
          decision_contract: makeContract(firstOffer),
        }),
      });
    });

    expect(await screen.findByRole('dialog', { name: 'Incoming trade offer' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Property details' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close modal' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('opens compact player rows into detailed holdings', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Opponent details' }));

    expect(screen.getByRole('dialog', { name: 'Opponent' })).toBeInTheDocument();
    expect(screen.getByText('Opponent profile')).toBeInTheDocument();
    expect(screen.getByText('Player holdings')).toBeInTheDocument();
  });

  it('prioritizes debt controls in the board center', () => {
    useGameStore.setState({
      gameState: makeGameState({
        gamePhase: 'debt_resolution',
        debt: { debtor: 0, amount: 200, creditor: 1 },
        legal_actions: [{ type: 'MortgageProperty', player_id: 0, property_id: 1 }],
      }),
    });

    renderPage();

    expect(screen.getByTestId('mock-board-center')).toHaveTextContent('Debt resolution');
    expect(screen.getByRole('button', { name: /mortgage property/i })).toBeInTheDocument();
    expect(screen.queryByText('Centered actions')).not.toBeInTheDocument();
  });
});
