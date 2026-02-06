import { useState, useEffect, useCallback, useMemo } from 'react';

/**
 * Tailwind CSS v4 breakpoints
 */
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

export type Breakpoint = keyof typeof BREAKPOINTS;

export type DeviceType = 'mobile' | 'tablet' | 'desktop';

export type Orientation = 'portrait' | 'landscape';

export interface ResponsiveState {
  /** Current window width */
  width: number;
  /** Current window height */
  height: number;
  /** Current breakpoint name */
  breakpoint: Breakpoint | 'xs';
  /** Device type based on screen size */
  deviceType: DeviceType;
  /** Screen orientation */
  orientation: Orientation;
  /** True if screen is smaller than sm breakpoint */
  isMobile: boolean;
  /** True if screen is sm or md breakpoint */
  isTablet: boolean;
  /** True if screen is lg or larger */
  isDesktop: boolean;
  /** True if screen width < height */
  isPortrait: boolean;
  /** True if screen width >= height */
  isLandscape: boolean;
  /** Check if current width is at or above a breakpoint */
  isAbove: (breakpoint: Breakpoint) => boolean;
  /** Check if current width is below a breakpoint */
  isBelow: (breakpoint: Breakpoint) => boolean;
  /** Check if current width is between two breakpoints */
  isBetween: (min: Breakpoint, max: Breakpoint) => boolean;
}

function getBreakpoint(width: number): Breakpoint | 'xs' {
  if (width >= BREAKPOINTS['2xl']) return '2xl';
  if (width >= BREAKPOINTS.xl) return 'xl';
  if (width >= BREAKPOINTS.lg) return 'lg';
  if (width >= BREAKPOINTS.md) return 'md';
  if (width >= BREAKPOINTS.sm) return 'sm';
  return 'xs';
}

function getDeviceType(width: number): DeviceType {
  if (width < BREAKPOINTS.sm) return 'mobile';
  if (width < BREAKPOINTS.lg) return 'tablet';
  return 'desktop';
}

function getOrientation(width: number, height: number): Orientation {
  return width >= height ? 'landscape' : 'portrait';
}

/**
 * Hook for responsive design utilities.
 * Tracks window dimensions and provides breakpoint/device type information.
 *
 * @param debounceMs - Debounce delay for resize events (default: 100ms)
 * @returns Responsive state with dimensions, breakpoints, and utility functions
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { isMobile, isAbove, breakpoint } = useResponsive();
 *
 *   return (
 *     <div>
 *       {isMobile ? <MobileLayout /> : <DesktopLayout />}
 *       {isAbove('lg') && <SidePanel />}
 *       <p>Current breakpoint: {breakpoint}</p>
 *     </div>
 *   );
 * }
 * ```
 */
export function useResponsive(debounceMs = 100): ResponsiveState {
  const [dimensions, setDimensions] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
  }));

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const handleResize = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      timeoutId = setTimeout(() => {
        setDimensions({
          width: window.innerWidth,
          height: window.innerHeight,
        });
      }, debounceMs);
    };

    window.addEventListener('resize', handleResize);

    // Trigger initial check via timeout to avoid synchronous setState in effect
    timeoutId = setTimeout(() => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }, 0);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [debounceMs]);

  const { width, height } = dimensions;

  const breakpoint = useMemo(() => getBreakpoint(width), [width]);
  const deviceType = useMemo(() => getDeviceType(width), [width]);
  const orientation = useMemo(() => getOrientation(width, height), [width, height]);

  const isMobile = width < BREAKPOINTS.sm;
  const isTablet = width >= BREAKPOINTS.sm && width < BREAKPOINTS.lg;
  const isDesktop = width >= BREAKPOINTS.lg;
  const isPortrait = orientation === 'portrait';
  const isLandscape = orientation === 'landscape';

  const isAbove = useCallback(
    (bp: Breakpoint) => width >= BREAKPOINTS[bp],
    [width]
  );

  const isBelow = useCallback(
    (bp: Breakpoint) => width < BREAKPOINTS[bp],
    [width]
  );

  const isBetween = useCallback(
    (min: Breakpoint, max: Breakpoint) =>
      width >= BREAKPOINTS[min] && width < BREAKPOINTS[max],
    [width]
  );

  return {
    width,
    height,
    breakpoint,
    deviceType,
    orientation,
    isMobile,
    isTablet,
    isDesktop,
    isPortrait,
    isLandscape,
    isAbove,
    isBelow,
    isBetween,
  };
}

export default useResponsive;
