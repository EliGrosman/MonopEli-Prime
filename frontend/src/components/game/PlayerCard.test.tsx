import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PlayerCard } from './PlayerCard';
import type { PlayerState, PropertyState } from '@/types';

const mockPlayer: PlayerState = {
  id: 0,
  name: 'Alice',
  money: 1500,
  position: 5,
  inJail: false,
  jailTurns: 0,
  jailCards: 1,
  bankrupt: false,
  isAi: false,
  color: '#E53935',
};

const mockProperties: Record<number, PropertyState> = {
  1: { position: 1, owner: 0, houses: 2, mortgaged: false },
  3: { position: 3, owner: 0, houses: 0, mortgaged: false },
  5: { position: 5, owner: 1, houses: 0, mortgaged: true },
};

describe('PlayerCard', () => {
  it('renders player name', () => {
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('Alice')).toBeInTheDocument();
  });

  it('shows (You) label for current user', () => {
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={true}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('(You)')).toBeInTheDocument();
  });

  it('shows Turn badge for current turn', () => {
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={true}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('Turn')).toBeInTheDocument();
  });

  it('shows AI badge for AI players', () => {
    const aiPlayer = { ...mockPlayer, isAi: true };
    render(
      <PlayerCard
        player={aiPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('shows guided Jev objectives and fallback status', () => {
    const aiPlayer = { ...mockPlayer, isAi: true, aiType: 'jev' };
    render(
      <PlayerCard
        player={aiPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
        inspection={{
          player_id: 0,
          sequence: 3,
          basis_revision: 8,
          status: 'fallback',
          short_term_objective: 'Restore liquidity',
          long_term_objective: 'Return to development',
          cash_reserve_target: 300,
          latest_summary: 'Mortgaged Baltic Avenue',
          fallback_reason: 'timeout',
        }}
      />
    );
    expect(screen.getByText('Jev')).toBeInTheDocument();
    expect(screen.getByText('Now: Restore liquidity')).toBeInTheDocument();
    expect(screen.getByText('Fallback: timeout')).toBeInTheDocument();
  });

  it('shows Bankrupt badge for bankrupt players', () => {
    const bankruptPlayer = { ...mockPlayer, bankrupt: true };
    render(
      <PlayerCard
        player={bankruptPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('Bankrupt')).toBeInTheDocument();
  });

  it('shows In Jail badge when in jail', () => {
    const jailedPlayer = { ...mockPlayer, inJail: true };
    render(
      <PlayerCard
        player={jailedPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText('In Jail')).toBeInTheDocument();
  });

  it('displays player money', () => {
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    // Player has $1,500 - appears in Cash row
    expect(screen.getAllByText('$1,500').length).toBeGreaterThanOrEqual(1);
  });

  it('displays jail cards count when player has cards', () => {
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );
    expect(screen.getByText(/Get Out of Jail Free: 1/)).toBeInTheDocument();
  });

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn();
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
        onSelect={onSelect}
      />
    );

    fireEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it('supports keyboard selection', () => {
    const onSelect = vi.fn();
    render(
      <PlayerCard
        player={mockPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
        onSelect={onSelect}
      />
    );

    const card = screen.getByLabelText('Alice');
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it('has reduced opacity when bankrupt', () => {
    const bankruptPlayer = { ...mockPlayer, bankrupt: true };
    render(
      <PlayerCard
        player={bankruptPlayer}
        isCurrentTurn={false}
        isCurrentUser={false}
        properties={mockProperties}
      />
    );

    const card = screen.getByLabelText('Alice');
    expect(card.className).toContain('opacity-50');
  });
});
