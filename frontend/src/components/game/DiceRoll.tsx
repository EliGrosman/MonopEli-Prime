import { useState, useEffect, useRef } from 'react';
import type { DiceRoll as DiceRollType } from '@/types';

interface DiceRollProps {
  roll: DiceRollType | null;
  onRoll?: () => void;
  canRoll?: boolean;
  isRolling?: boolean;
}

/**
 * Dice display component with roll animation.
 */
export function DiceRoll({ roll, onRoll, canRoll = false, isRolling = false }: DiceRollProps) {
  const [animating, setAnimating] = useState(false);
  const [displayDice, setDisplayDice] = useState<[number, number]>([1, 1]);
  // Track the last roll values to prevent re-animating on state updates
  const lastRollRef = useRef<string | null>(null);

  // Animate dice when a NEW roll comes in (not just state refresh)
  useEffect(() => {
    if (roll && !isRolling) {
      // Create a key from the roll values to compare
      const rollKey = `${roll.die1}-${roll.die2}`;

      // Only animate if this is actually a new roll
      if (lastRollRef.current === rollKey) {
        // Same roll values, defer update to avoid cascading renders
        const t = setTimeout(() => setDisplayDice([roll.die1, roll.die2]), 0);
        return () => clearTimeout(t);
      }

      // New roll - update ref and animate
      lastRollRef.current = rollKey;

      // Defer state update to avoid cascading renders
      const startTimeout = setTimeout(() => {
        setAnimating(true);
      }, 0);

      // Random dice animation
      const interval = setInterval(() => {
        setDisplayDice([
          Math.floor(Math.random() * 6) + 1,
          Math.floor(Math.random() * 6) + 1,
        ]);
      }, 50);

      // Stop after animation
      const timeout = setTimeout(() => {
        clearInterval(interval);
        setDisplayDice([roll.die1, roll.die2]);
        setAnimating(false);
      }, 500);

      return () => {
        clearTimeout(startTimeout);
        clearInterval(interval);
        clearTimeout(timeout);
      };
    } else if (!roll) {
      // Reset when roll is cleared (new turn)
      lastRollRef.current = null;
    }
  }, [roll, isRolling]);

  const total = roll ? roll.die1 + roll.die2 : 0;

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Dice display */}
      <div className="flex items-center gap-4">
        <Die
          value={roll ? displayDice[0] : null}
          animating={animating || isRolling}
        />
        <Die
          value={roll ? displayDice[1] : null}
          animating={animating || isRolling}
        />
      </div>

      {/* Roll result */}
      {roll && !animating && (
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">
            {total}
          </div>
          {roll.isDoubles && (
            <div className="mt-1 px-3 py-1 bg-yellow-400 text-yellow-900 rounded-full text-sm font-semibold animate-bounce">
              Doubles!
            </div>
          )}
        </div>
      )}

      {/* Roll button */}
      {canRoll && onRoll && (
        <button
          onClick={onRoll}
          disabled={isRolling || animating}
          className={`
            px-6 py-3 rounded-lg font-bold text-lg transition-all
            ${isRolling || animating
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600 active:scale-95 shadow-lg hover:shadow-xl'
            }
          `}
        >
          {isRolling ? 'Rolling...' : 'Roll Dice'}
        </button>
      )}
    </div>
  );
}

interface DieProps {
  value: number | null;
  animating?: boolean;
}

/**
 * Single die component.
 */
function Die({ value, animating = false }: DieProps) {
  return (
    <div
      className={`
        w-16 h-16 bg-white rounded-xl shadow-lg border-2 border-gray-200
        flex items-center justify-center
        ${animating ? 'animate-spin' : ''}
      `}
    >
      {value !== null ? (
        <DiceFace value={value} />
      ) : (
        <span className="text-gray-300 text-2xl">?</span>
      )}
    </div>
  );
}

interface DiceFaceProps {
  value: number;
}

/**
 * Dice face with dots.
 */
function DiceFace({ value }: DiceFaceProps) {
  const dotPositions: Record<number, string[]> = {
    1: ['center'],
    2: ['top-right', 'bottom-left'],
    3: ['top-right', 'center', 'bottom-left'],
    4: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
    5: ['top-left', 'top-right', 'center', 'bottom-left', 'bottom-right'],
    6: ['top-left', 'top-right', 'middle-left', 'middle-right', 'bottom-left', 'bottom-right'],
  };

  const positions = dotPositions[value] || [];

  const getPosition = (pos: string): string => {
    switch (pos) {
      case 'top-left':
        return 'top-1 left-1';
      case 'top-right':
        return 'top-1 right-1';
      case 'middle-left':
        return 'top-1/2 -translate-y-1/2 left-1';
      case 'middle-right':
        return 'top-1/2 -translate-y-1/2 right-1';
      case 'bottom-left':
        return 'bottom-1 left-1';
      case 'bottom-right':
        return 'bottom-1 right-1';
      case 'center':
        return 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2';
      default:
        return '';
    }
  };

  return (
    <div className="relative w-12 h-12">
      {positions.map((pos, i) => (
        <div
          key={i}
          className={`absolute w-2.5 h-2.5 bg-gray-800 rounded-full ${getPosition(pos)}`}
        />
      ))}
    </div>
  );
}

/**
 * Compact dice display for sidebar/panel use.
 */
export function DiceDisplay({ roll }: { roll: DiceRollType | null }) {
  if (!roll) return null;

  const total = roll.die1 + roll.die2;

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1">
        <span className="w-6 h-6 bg-white rounded border border-gray-300 flex items-center justify-center text-sm font-bold text-gray-900">
          {roll.die1}
        </span>
        <span className="text-gray-500">+</span>
        <span className="w-6 h-6 bg-white rounded border border-gray-300 flex items-center justify-center text-sm font-bold text-gray-900">
          {roll.die2}
        </span>
        <span className="text-gray-500">=</span>
        <span className="font-bold text-lg text-gray-900">{total}</span>
      </div>
      {roll.isDoubles && (
        <span className="px-1.5 py-0.5 text-xs bg-yellow-400 text-yellow-900 rounded">
          Doubles
        </span>
      )}
    </div>
  );
}
