import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LobbyList } from '@/components/lobby/LobbyList';
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

describe('LobbyList', () => {
  it('renders loading skeleton when loading', () => {
    const onJoin = vi.fn();

    render(<LobbyList lobbies={[]} isLoading={true} onJoin={onJoin} />);

    // Should show loading skeletons (3 by default)
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBe(3);
  });

  it('renders empty state when no lobbies', () => {
    const onJoin = vi.fn();

    render(<LobbyList lobbies={[]} isLoading={false} onJoin={onJoin} />);

    expect(screen.getByText('No public games available')).toBeInTheDocument();
    expect(screen.getByText('Create a new game or join with a code')).toBeInTheDocument();
  });

  it('renders lobby cards when lobbies exist', () => {
    const lobbies = [
      createMockLobby({ id: 'lobby-1', name: 'Game 1' }),
      createMockLobby({ id: 'lobby-2', name: 'Game 2' }),
    ];
    const onJoin = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} />);

    expect(screen.getByText('Game 1')).toBeInTheDocument();
    expect(screen.getByText('Game 2')).toBeInTheDocument();
  });

  it('shows correct game count', () => {
    const lobbies = [
      createMockLobby({ id: 'lobby-1', name: 'Game 1' }),
      createMockLobby({ id: 'lobby-2', name: 'Game 2' }),
    ];
    const onJoin = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} />);

    expect(screen.getByText('2 games available')).toBeInTheDocument();
  });

  it('shows singular game count for one lobby', () => {
    const lobbies = [createMockLobby({ id: 'lobby-1', name: 'Game 1' })];
    const onJoin = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} />);

    expect(screen.getByText('1 game available')).toBeInTheDocument();
  });

  it('calls onJoin when lobby card is clicked', () => {
    const lobbies = [createMockLobby({ id: 'lobby-1', name: 'Game 1' })];
    const onJoin = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} />);

    fireEvent.click(screen.getByText('Game 1'));
    expect(onJoin).toHaveBeenCalledWith('lobby-1');
  });

  it('shows refresh button when onRefresh is provided', () => {
    const lobbies = [createMockLobby()];
    const onJoin = vi.fn();
    const onRefresh = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} onRefresh={onRefresh} />);

    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('calls onRefresh when refresh button is clicked', () => {
    const lobbies = [createMockLobby()];
    const onJoin = vi.fn();
    const onRefresh = vi.fn();

    render(<LobbyList lobbies={lobbies} isLoading={false} onJoin={onJoin} onRefresh={onRefresh} />);

    fireEvent.click(screen.getByText('Refresh'));
    expect(onRefresh).toHaveBeenCalled();
  });

  it('shows refresh link in empty state when onRefresh provided', () => {
    const onJoin = vi.fn();
    const onRefresh = vi.fn();

    render(<LobbyList lobbies={[]} isLoading={false} onJoin={onJoin} onRefresh={onRefresh} />);

    expect(screen.getByText('Refresh list')).toBeInTheDocument();
  });
});
