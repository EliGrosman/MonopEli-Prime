import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LobbySettings } from '@/components/lobby/LobbySettings';
import type { LobbySettings as LobbySettingsType } from '@/types';

function createMockSettings(overrides: Partial<LobbySettingsType> = {}): LobbySettingsType {
  return {
    max_players: 4,
    min_players: 2,
    starting_money: 1500,
    go_salary: 200,
    allow_spectators: false,
    private: false,
    ...overrides,
  };
}

describe('LobbySettings', () => {
  describe('read-only mode (non-host)', () => {
    it('renders settings values', () => {
      const settings = createMockSettings({
        starting_money: 1500,
        max_players: 4,
        go_salary: 200,
        allow_spectators: true,
      });

      render(<LobbySettings settings={settings} isHost={false} />);

      expect(screen.getByText('$1500')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument();
      expect(screen.getByText('$200')).toBeInTheDocument();
      expect(screen.getByText('Allowed')).toBeInTheDocument();
    });

    it('shows Not Allowed for no spectators', () => {
      const settings = createMockSettings({ allow_spectators: false });

      render(<LobbySettings settings={settings} isHost={false} />);

      expect(screen.getByText('Not Allowed')).toBeInTheDocument();
    });

    it('does not show Edit button', () => {
      const settings = createMockSettings();

      render(<LobbySettings settings={settings} isHost={false} />);

      expect(screen.queryByText('Edit')).not.toBeInTheDocument();
    });
  });

  describe('host mode', () => {
    it('shows Edit button', () => {
      const settings = createMockSettings();

      render(<LobbySettings settings={settings} isHost={true} />);

      expect(screen.getByText('Edit')).toBeInTheDocument();
    });

    it('enters edit mode when Edit is clicked', () => {
      const settings = createMockSettings();

      render(<LobbySettings settings={settings} isHost={true} />);

      fireEvent.click(screen.getByText('Edit'));

      expect(screen.getByLabelText('Starting Money')).toBeInTheDocument();
      expect(screen.getByLabelText('Max Players')).toBeInTheDocument();
      expect(screen.getByLabelText('GO Salary')).toBeInTheDocument();
      expect(screen.getByLabelText('Allow Spectators')).toBeInTheDocument();
    });

    it('shows Save and Cancel buttons in edit mode', () => {
      const settings = createMockSettings();

      render(<LobbySettings settings={settings} isHost={true} />);
      fireEvent.click(screen.getByText('Edit'));

      expect(screen.getByText('Save')).toBeInTheDocument();
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    it('cancels editing and returns to read mode', () => {
      const settings = createMockSettings();

      render(<LobbySettings settings={settings} isHost={true} />);
      fireEvent.click(screen.getByText('Edit'));
      fireEvent.click(screen.getByText('Cancel'));

      expect(screen.getByText('Edit')).toBeInTheDocument();
      expect(screen.queryByText('Save')).not.toBeInTheDocument();
    });

    it('calls onUpdate with new settings when Save is clicked', () => {
      const settings = createMockSettings();
      const onUpdate = vi.fn();

      render(<LobbySettings settings={settings} isHost={true} onUpdate={onUpdate} />);
      fireEvent.click(screen.getByText('Edit'));

      // Change starting money
      fireEvent.change(screen.getByLabelText('Starting Money'), {
        target: { value: '2000' },
      });

      fireEvent.click(screen.getByText('Save'));

      expect(onUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          starting_money: 2000,
        })
      );
    });

    it('resets form values on cancel', () => {
      const settings = createMockSettings({ starting_money: 1500 });
      const onUpdate = vi.fn();

      render(<LobbySettings settings={settings} isHost={true} onUpdate={onUpdate} />);
      fireEvent.click(screen.getByText('Edit'));

      // Change value
      fireEvent.change(screen.getByLabelText('Starting Money'), {
        target: { value: '2000' },
      });

      // Cancel
      fireEvent.click(screen.getByText('Cancel'));

      // Re-enter edit mode
      fireEvent.click(screen.getByText('Edit'));

      // Value should be reset
      expect(screen.getByLabelText('Starting Money')).toHaveValue('1500');
    });
  });
});
