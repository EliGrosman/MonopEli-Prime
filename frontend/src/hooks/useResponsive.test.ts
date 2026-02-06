import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useResponsive, BREAKPOINTS } from './useResponsive';

describe('useResponsive', () => {
  const originalInnerWidth = window.innerWidth;
  const originalInnerHeight = window.innerHeight;

  function setWindowSize(width: number, height: number) {
    Object.defineProperty(window, 'innerWidth', { value: width, writable: true });
    Object.defineProperty(window, 'innerHeight', { value: height, writable: true });
  }

  beforeEach(() => {
    vi.useFakeTimers();
    setWindowSize(1024, 768);
  });

  afterEach(() => {
    vi.useRealTimers();
    setWindowSize(originalInnerWidth, originalInnerHeight);
  });

  describe('initial state', () => {
    it('returns correct dimensions', () => {
      setWindowSize(1200, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.width).toBe(1200);
      expect(result.current.height).toBe(800);
    });

    it('returns lg breakpoint for 1024px', () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('lg');
    });

    it('returns xs breakpoint for small screens', () => {
      setWindowSize(400, 600);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('xs');
    });
  });

  describe('breakpoint detection', () => {
    it('detects xs breakpoint', () => {
      setWindowSize(500, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('xs');
    });

    it('detects sm breakpoint', () => {
      setWindowSize(640, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('sm');
    });

    it('detects md breakpoint', () => {
      setWindowSize(768, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('md');
    });

    it('detects lg breakpoint', () => {
      setWindowSize(1024, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('lg');
    });

    it('detects xl breakpoint', () => {
      setWindowSize(1280, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('xl');
    });

    it('detects 2xl breakpoint', () => {
      setWindowSize(1536, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.breakpoint).toBe('2xl');
    });
  });

  describe('device type detection', () => {
    it('detects mobile device', () => {
      setWindowSize(400, 700);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.deviceType).toBe('mobile');
      expect(result.current.isMobile).toBe(true);
      expect(result.current.isTablet).toBe(false);
      expect(result.current.isDesktop).toBe(false);
    });

    it('detects tablet device', () => {
      setWindowSize(800, 600);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.deviceType).toBe('tablet');
      expect(result.current.isMobile).toBe(false);
      expect(result.current.isTablet).toBe(true);
      expect(result.current.isDesktop).toBe(false);
    });

    it('detects desktop device', () => {
      setWindowSize(1200, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.deviceType).toBe('desktop');
      expect(result.current.isMobile).toBe(false);
      expect(result.current.isTablet).toBe(false);
      expect(result.current.isDesktop).toBe(true);
    });
  });

  describe('orientation detection', () => {
    it('detects landscape orientation', () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.orientation).toBe('landscape');
      expect(result.current.isLandscape).toBe(true);
      expect(result.current.isPortrait).toBe(false);
    });

    it('detects portrait orientation', () => {
      setWindowSize(768, 1024);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.orientation).toBe('portrait');
      expect(result.current.isPortrait).toBe(true);
      expect(result.current.isLandscape).toBe(false);
    });

    it('treats equal width/height as landscape', () => {
      setWindowSize(800, 800);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.orientation).toBe('landscape');
    });
  });

  describe('isAbove utility', () => {
    it('returns true when above breakpoint', () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isAbove('sm')).toBe(true);
      expect(result.current.isAbove('md')).toBe(true);
      expect(result.current.isAbove('lg')).toBe(true);
    });

    it('returns false when below breakpoint', () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isAbove('xl')).toBe(false);
      expect(result.current.isAbove('2xl')).toBe(false);
    });
  });

  describe('isBelow utility', () => {
    it('returns true when below breakpoint', () => {
      setWindowSize(600, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBelow('sm')).toBe(true);
      expect(result.current.isBelow('md')).toBe(true);
    });

    it('returns false when at or above breakpoint', () => {
      setWindowSize(800, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBelow('sm')).toBe(false);
      expect(result.current.isBelow('md')).toBe(false);
    });
  });

  describe('isBetween utility', () => {
    it('returns true when between breakpoints', () => {
      setWindowSize(800, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBetween('sm', 'lg')).toBe(true);
    });

    it('returns false when outside range', () => {
      setWindowSize(1200, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBetween('sm', 'lg')).toBe(false);
    });

    it('includes min breakpoint', () => {
      setWindowSize(BREAKPOINTS.md, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBetween('md', 'lg')).toBe(true);
    });

    it('excludes max breakpoint', () => {
      setWindowSize(BREAKPOINTS.lg, 768);
      const { result } = renderHook(() => useResponsive());

      expect(result.current.isBetween('md', 'lg')).toBe(false);
    });
  });

  describe('resize handling', () => {
    it('updates on window resize with debounce', async () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive(100));

      expect(result.current.width).toBe(1024);

      // Simulate resize
      act(() => {
        setWindowSize(500, 700);
        window.dispatchEvent(new Event('resize'));
      });

      // Before debounce, still old value
      expect(result.current.width).toBe(1024);

      // After debounce
      act(() => {
        vi.advanceTimersByTime(100);
      });

      expect(result.current.width).toBe(500);
      expect(result.current.isMobile).toBe(true);
    });

    it('debounces multiple rapid resizes', async () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive(100));

      // Simulate multiple rapid resizes
      act(() => {
        setWindowSize(800, 600);
        window.dispatchEvent(new Event('resize'));
      });

      act(() => {
        vi.advanceTimersByTime(50);
      });

      act(() => {
        setWindowSize(600, 500);
        window.dispatchEvent(new Event('resize'));
      });

      act(() => {
        vi.advanceTimersByTime(50);
      });

      act(() => {
        setWindowSize(400, 300);
        window.dispatchEvent(new Event('resize'));
      });

      // Before final debounce completes
      expect(result.current.width).toBe(1024);

      // After debounce completes
      act(() => {
        vi.advanceTimersByTime(100);
      });

      // Should have the final resize value
      expect(result.current.width).toBe(400);
    });

    it('uses custom debounce delay', async () => {
      setWindowSize(1024, 768);
      const { result } = renderHook(() => useResponsive(200));

      act(() => {
        setWindowSize(500, 700);
        window.dispatchEvent(new Event('resize'));
      });

      // After 100ms, should not update yet
      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(result.current.width).toBe(1024);

      // After 200ms total, should update
      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(result.current.width).toBe(500);
    });

    it('cleans up event listener on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');
      const { unmount } = renderHook(() => useResponsive());

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));
      removeEventListenerSpy.mockRestore();
    });
  });

  describe('BREAKPOINTS constant', () => {
    it('exports correct breakpoint values', () => {
      expect(BREAKPOINTS.sm).toBe(640);
      expect(BREAKPOINTS.md).toBe(768);
      expect(BREAKPOINTS.lg).toBe(1024);
      expect(BREAKPOINTS.xl).toBe(1280);
      expect(BREAKPOINTS['2xl']).toBe(1536);
    });
  });
});
