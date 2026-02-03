import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PropertyCard } from './PropertyCard';
import type { PropertyState } from '@/types';

describe('PropertyCard', () => {
  const mockProperty: PropertyState = {
    position: 1,
    owner: 0,
    houses: 0,
    mortgaged: false,
  };

  describe('compact mode', () => {
    it('renders property name', () => {
      render(<PropertyCard position={1} compact={true} />);
      expect(screen.getByText('Mediterranean Avenue')).toBeInTheDocument();
    });

    it('shows color indicator', () => {
      const { container } = render(<PropertyCard position={1} compact={true} />);
      // Should have brown color class
      expect(container.querySelector('.bg-property-brown')).toBeInTheDocument();
    });

    it('shows houses when present', () => {
      const property = { ...mockProperty, houses: 3 };
      render(<PropertyCard position={1} property={property} compact={true} />);
      // Should show 3 house indicators
      const houses = screen.getAllByTitle('House');
      expect(houses).toHaveLength(3);
    });

    it('shows hotel when 5 houses', () => {
      const property = { ...mockProperty, houses: 5 };
      render(<PropertyCard position={1} property={property} compact={true} />);
      expect(screen.getByTitle('Hotel')).toBeInTheDocument();
    });

    it('shows mortgage indicator', () => {
      const property = { ...mockProperty, mortgaged: true };
      render(<PropertyCard position={1} property={property} compact={true} />);
      expect(screen.getByText('M')).toBeInTheDocument();
    });

    it('calls onClick when clicked', () => {
      const onClick = vi.fn();
      render(<PropertyCard position={1} compact={true} onClick={onClick} />);
      fireEvent.click(screen.getByRole('button'));
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it('is keyboard accessible', () => {
      const onClick = vi.fn();
      render(<PropertyCard position={1} compact={true} onClick={onClick} />);
      const button = screen.getByRole('button');
      fireEvent.keyDown(button, { key: 'Enter' });
      // The component should be focusable
      expect(button).toHaveAttribute('tabIndex', '0');
    });
  });

  describe('full mode', () => {
    it('renders property name in header', () => {
      render(<PropertyCard position={1} compact={false} />);
      expect(screen.getByRole('heading', { name: 'Mediterranean Avenue' })).toBeInTheDocument();
    });

    it('shows rent schedule for properties', () => {
      render(<PropertyCard position={1} compact={false} />);
      expect(screen.getByText('Rent')).toBeInTheDocument();
      expect(screen.getByText('With 1 House')).toBeInTheDocument();
      expect(screen.getByText('With Hotel')).toBeInTheDocument();
    });

    it('shows house cost for properties', () => {
      render(<PropertyCard position={1} compact={false} />);
      expect(screen.getByText('House Cost')).toBeInTheDocument();
    });

    it('shows mortgage value', () => {
      render(<PropertyCard position={1} compact={false} />);
      expect(screen.getByText('Mortgage Value')).toBeInTheDocument();
    });

    it('shows price', () => {
      render(<PropertyCard position={1} compact={false} />);
      expect(screen.getByText('Price')).toBeInTheDocument();
    });

    it('shows MORTGAGED label when mortgaged', () => {
      const property = { ...mockProperty, mortgaged: true };
      render(<PropertyCard position={1} property={property} compact={false} />);
      expect(screen.getByText('MORTGAGED')).toBeInTheDocument();
    });

    it('highlights current rent level', () => {
      const property = { ...mockProperty, houses: 2 };
      const { container } = render(<PropertyCard position={1} property={property} compact={false} />);
      // The "With 2 Houses" row should be highlighted
      const highlightedRow = container.querySelector('.bg-yellow-100');
      expect(highlightedRow).toBeInTheDocument();
      expect(highlightedRow?.textContent).toContain('With 2 Houses');
    });
  });

  describe('railroad', () => {
    it('shows railroad rent schedule', () => {
      render(<PropertyCard position={5} compact={false} />);
      expect(screen.getByText('Rent (1 RR)')).toBeInTheDocument();
      expect(screen.getByText('Rent (2 RR)')).toBeInTheDocument();
      expect(screen.getByText('Rent (3 RR)')).toBeInTheDocument();
      expect(screen.getByText('Rent (4 RR)')).toBeInTheDocument();
    });
  });

  describe('utility', () => {
    it('shows utility rent description', () => {
      render(<PropertyCard position={12} compact={false} />);
      expect(screen.getByText(/4x dice roll/)).toBeInTheDocument();
      expect(screen.getByText(/10x dice roll/)).toBeInTheDocument();
    });
  });

  describe('actions', () => {
    it('shows build button when showActions and owned', () => {
      const onBuild = vi.fn();
      const property = { ...mockProperty, houses: 0 };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onBuild={onBuild}
        />
      );
      expect(screen.getByRole('button', { name: 'Build' })).toBeInTheDocument();
    });

    it('hides build button when mortgaged', () => {
      const property = { ...mockProperty, mortgaged: true };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onBuild={() => {}}
        />
      );
      expect(screen.queryByRole('button', { name: 'Build' })).not.toBeInTheDocument();
    });

    it('shows sell button when has houses', () => {
      const property = { ...mockProperty, houses: 2 };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onSell={() => {}}
        />
      );
      expect(screen.getByRole('button', { name: 'Sell' })).toBeInTheDocument();
    });

    it('shows mortgage button when no houses', () => {
      const property = { ...mockProperty, houses: 0 };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onMortgage={() => {}}
        />
      );
      expect(screen.getByRole('button', { name: 'Mortgage' })).toBeInTheDocument();
    });

    it('shows unmortgage button when mortgaged', () => {
      const property = { ...mockProperty, mortgaged: true };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onUnmortgage={() => {}}
        />
      );
      expect(screen.getByRole('button', { name: 'Unmortgage' })).toBeInTheDocument();
    });

    it('calls action callbacks', () => {
      const onBuild = vi.fn();
      const property = { ...mockProperty, houses: 0 };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onBuild={onBuild}
        />
      );
      fireEvent.click(screen.getByRole('button', { name: 'Build' }));
      expect(onBuild).toHaveBeenCalledTimes(1);
    });

    it('stops click propagation on action buttons', () => {
      const onClick = vi.fn();
      const onBuild = vi.fn();
      const property = { ...mockProperty, houses: 0 };
      render(
        <PropertyCard
          position={1}
          property={property}
          compact={false}
          showActions={true}
          onClick={onClick}
          onBuild={onBuild}
        />
      );
      fireEvent.click(screen.getByRole('button', { name: 'Build' }));
      expect(onBuild).toHaveBeenCalled();
      expect(onClick).not.toHaveBeenCalled();
    });
  });

  describe('null handling', () => {
    it('returns null for invalid position', () => {
      const { container } = render(<PropertyCard position={0} />);
      expect(container.firstChild).toBeNull();
    });
  });
});
