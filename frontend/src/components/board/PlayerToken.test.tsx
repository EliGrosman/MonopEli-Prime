import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { PlayerToken } from './PlayerToken';

describe('PlayerToken', () => {
  it('renders a player token', () => {
    render(<PlayerToken playerId={0} />);
    expect(screen.getByTestId('token-player-0')).toBeInTheDocument();
  });

  it('applies custom color when provided', () => {
    render(<PlayerToken playerId={0} color="#FF0000" />);
    const token = screen.getByTestId('token-player-0');
    expect(token).toHaveStyle({ backgroundColor: '#FF0000' });
  });

  it('uses default player color when no custom color provided', () => {
    render(<PlayerToken playerId={0} />);
    const token = screen.getByTestId('token-player-0');
    // Player 0 default color is #E53935
    expect(token).toHaveStyle({ backgroundColor: '#E53935' });
  });

  it('has correct aria-label for accessibility', () => {
    render(<PlayerToken playerId={2} />);
    const token = screen.getByTestId('token-player-2');
    expect(token).toHaveAttribute('aria-label', 'Player 3 token');
  });

  it('has correct title attribute', () => {
    render(<PlayerToken playerId={1} />);
    const token = screen.getByTestId('token-player-1');
    expect(token).toHaveAttribute('title', 'Player 2');
  });

  it('applies offset for stacking multiple tokens', () => {
    render(<PlayerToken playerId={0} offset={2} />);
    const token = screen.getByTestId('token-player-0');
    expect(token).toHaveStyle({ marginLeft: '-8px' });
  });
});
