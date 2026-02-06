import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { HouseIndicator } from './HouseIndicator';

describe('HouseIndicator', () => {
  it('renders nothing when houses is 0', () => {
    const { container } = render(<HouseIndicator houses={0} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders 1 house', () => {
    render(<HouseIndicator houses={1} />);
    expect(screen.getAllByTestId('house')).toHaveLength(1);
  });

  it('renders 2 houses', () => {
    render(<HouseIndicator houses={2} />);
    expect(screen.getAllByTestId('house')).toHaveLength(2);
  });

  it('renders 3 houses', () => {
    render(<HouseIndicator houses={3} />);
    expect(screen.getAllByTestId('house')).toHaveLength(3);
  });

  it('renders 4 houses', () => {
    render(<HouseIndicator houses={4} />);
    expect(screen.getAllByTestId('house')).toHaveLength(4);
  });

  it('renders a hotel when houses is 5', () => {
    render(<HouseIndicator houses={5} />);
    expect(screen.getByTestId('hotel')).toBeInTheDocument();
    expect(screen.queryByTestId('house')).not.toBeInTheDocument();
  });

  it('houses have correct aria-label', () => {
    render(<HouseIndicator houses={2} />);
    const houses = screen.getAllByTestId('house');
    houses.forEach((house) => {
      expect(house).toHaveAttribute('aria-label', 'House');
    });
  });

  it('hotel has correct aria-label', () => {
    render(<HouseIndicator houses={5} />);
    const hotel = screen.getByTestId('hotel');
    expect(hotel).toHaveAttribute('aria-label', 'Hotel');
  });
});
