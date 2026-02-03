import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { CornerSpace } from './CornerSpace';

describe('CornerSpace', () => {
  it('renders GO space correctly', () => {
    render(<CornerSpace position={0} name="GO" />);
    expect(screen.getByText('GO')).toBeInTheDocument();
    expect(screen.getByText('COLLECT')).toBeInTheDocument();
    expect(screen.getByText('$200')).toBeInTheDocument();
  });

  it('renders Jail space correctly', () => {
    render(<CornerSpace position={10} name="Jail / Just Visiting" />);
    expect(screen.getByText('JAIL')).toBeInTheDocument();
    // Text is split with a br tag, so we use a regex matcher
    expect(screen.getByText(/JUST/)).toBeInTheDocument();
    expect(screen.getByText(/VISITING/)).toBeInTheDocument();
  });

  it('renders Free Parking space correctly', () => {
    render(<CornerSpace position={20} name="Free Parking" />);
    expect(screen.getByText('FREE')).toBeInTheDocument();
    expect(screen.getByText('PARKING')).toBeInTheDocument();
  });

  it('renders Go To Jail space correctly', () => {
    render(<CornerSpace position={30} name="Go To Jail" />);
    expect(screen.getByText('GO TO')).toBeInTheDocument();
    expect(screen.getByText('JAIL')).toBeInTheDocument();
  });

  it('renders default space for unknown position', () => {
    render(<CornerSpace position={99} name="Unknown" />);
    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });
});
