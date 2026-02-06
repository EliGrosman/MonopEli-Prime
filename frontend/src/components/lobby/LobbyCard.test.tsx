import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LobbyCard } from '@/components/lobby/LobbyCard';
import type { LobbyListItem } from '@/types';

function createMockLobby(overrides: Partial<LobbyListItem> = {}): LobbyListItem {
  return {
    id: 'lobby-1',
    name: 'Test Lobby',
    host_name: 'TestHost',
    status: 'waiting',
    current_players: 2,
    max_players: 4,
    created_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('LobbyCard', () => {
  it('renders lobby information', () => {
    const lobby = createMockLobby();
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    expect(screen.getByText('Test Lobby')).toBeInTheDocument();
    expect(screen.getByText('Hosted by TestHost')).toBeInTheDocument();
    expect(screen.getByText('2/4')).toBeInTheDocument();
    // ID is sliced to first 8 chars
    expect(screen.getByText('lobby-1')).toBeInTheDocument();
  });

  it('shows Full badge when lobby is full', () => {
    const lobby = createMockLobby({ current_players: 4, max_players: 4 });
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    expect(screen.getByText('Full')).toBeInTheDocument();
  });

  it('calls onJoin when clicked', () => {
    const lobby = createMockLobby();
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    fireEvent.click(screen.getByRole('button'));
    expect(onJoin).toHaveBeenCalledWith('lobby-1');
  });

  it('does not call onJoin when lobby is full', () => {
    const lobby = createMockLobby({ current_players: 4, max_players: 4 });
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    // Full lobby should not trigger onJoin
    fireEvent.click(screen.getByRole('button'));
    expect(onJoin).not.toHaveBeenCalled();
  });

  it('does not call onJoin when disabled', () => {
    const lobby = createMockLobby();
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} disabled />);

    fireEvent.click(screen.getByRole('button'));
    expect(onJoin).not.toHaveBeenCalled();
  });

  it('supports keyboard navigation', () => {
    const lobby = createMockLobby();
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    const card = screen.getByRole('button');
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(onJoin).toHaveBeenCalledWith('lobby-1');
  });

  it('supports Space key for activation', () => {
    const lobby = createMockLobby();
    const onJoin = vi.fn();

    render(<LobbyCard lobby={lobby} onJoin={onJoin} />);

    const card = screen.getByRole('button');
    fireEvent.keyDown(card, { key: ' ' });
    expect(onJoin).toHaveBeenCalledWith('lobby-1');
  });
});
