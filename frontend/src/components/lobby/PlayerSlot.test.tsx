import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PlayerSlot, EmptySlot } from '@/components/lobby/PlayerSlot';
import type { LobbyPlayer } from '@/types';

function createMockPlayer(overrides: Partial<LobbyPlayer> = {}): LobbyPlayer {
  return {
    session_id: 'session-1',
    name: 'TestPlayer',
    is_host: false,
    is_ready: false,
    is_ai: false,
    slot_id: 0,
    joined_at: new Date().toISOString(),
    ...overrides,
  };
}

describe('PlayerSlot', () => {
  it('renders player name', () => {
    const player = createMockPlayer({ name: 'Alice' });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.getByText('Alice')).toBeInTheDocument();
  });

  it('shows Host badge for host player', () => {
    const player = createMockPlayer();

    render(<PlayerSlot player={player} isHost={true} isSelf={false} canManage={false} />);

    expect(screen.getByText('Host')).toBeInTheDocument();
  });

  it('shows You badge for self', () => {
    const player = createMockPlayer();

    render(<PlayerSlot player={player} isHost={false} isSelf={true} canManage={false} />);

    expect(screen.getByText('You')).toBeInTheDocument();
  });

  it('shows AI badge for AI players', () => {
    const player = createMockPlayer({ is_ai: true });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('shows AI type for AI players', () => {
    const player = createMockPlayer({ is_ai: true, ai_type: 'rule_based' });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.getByText('rule based')).toBeInTheDocument();
  });

  it('shows Ready status when player is ready', () => {
    const player = createMockPlayer({ is_ready: true });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('shows Waiting status when player is not ready', () => {
    const player = createMockPlayer({ is_ready: false });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.getByText('Waiting')).toBeInTheDocument();
  });

  it('does not show ready status for AI players', () => {
    const player = createMockPlayer({ is_ai: true, is_ready: true });

    render(<PlayerSlot player={player} isHost={false} isSelf={false} canManage={false} />);

    expect(screen.queryByText('Ready')).not.toBeInTheDocument();
    expect(screen.queryByText('Waiting')).not.toBeInTheDocument();
  });

  it('shows remove button when canManage is true', () => {
    const player = createMockPlayer();
    const onKick = vi.fn();

    render(
      <PlayerSlot player={player} isHost={false} isSelf={false} canManage={true} onKick={onKick} />
    );

    expect(screen.getByTitle('Kick player')).toBeInTheDocument();
  });

  it('does not show remove button for self', () => {
    const player = createMockPlayer();
    const onKick = vi.fn();

    render(
      <PlayerSlot player={player} isHost={false} isSelf={true} canManage={true} onKick={onKick} />
    );

    expect(screen.queryByTitle('Kick player')).not.toBeInTheDocument();
  });

  it('calls onKick when kick button is clicked', () => {
    const player = createMockPlayer();
    const onKick = vi.fn();

    render(
      <PlayerSlot player={player} isHost={false} isSelf={false} canManage={true} onKick={onKick} />
    );

    fireEvent.click(screen.getByTitle('Kick player'));
    expect(onKick).toHaveBeenCalled();
  });

  it('calls onRemoveAi for AI players', () => {
    const player = createMockPlayer({ is_ai: true });
    const onRemoveAi = vi.fn();

    render(
      <PlayerSlot
        player={player}
        isHost={false}
        isSelf={false}
        canManage={true}
        onRemoveAi={onRemoveAi}
      />
    );

    fireEvent.click(screen.getByTitle('Remove AI'));
    expect(onRemoveAi).toHaveBeenCalled();
  });
});

describe('EmptySlot', () => {
  it('renders empty slot text', () => {
    render(<EmptySlot slotNumber={0} canAddAi={false} />);

    expect(screen.getByText('Empty Slot')).toBeInTheDocument();
  });

  it('shows Add AI button when canAddAi is true', () => {
    const onAddAi = vi.fn();

    render(<EmptySlot slotNumber={0} canAddAi={true} onAddAi={onAddAi} />);

    expect(screen.getByText('+ Add AI')).toBeInTheDocument();
  });

  it('does not show Add AI button when canAddAi is false', () => {
    render(<EmptySlot slotNumber={0} canAddAi={false} />);

    expect(screen.queryByText('+ Add AI')).not.toBeInTheDocument();
  });

  it('calls onAddAi when button is clicked', () => {
    const onAddAi = vi.fn();

    render(<EmptySlot slotNumber={0} canAddAi={true} onAddAi={onAddAi} />);

    fireEvent.click(screen.getByText('+ Add AI'));
    expect(onAddAi).toHaveBeenCalled();
  });
});
