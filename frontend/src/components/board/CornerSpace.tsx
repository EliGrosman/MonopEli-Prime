interface CornerSpaceProps {
  position: number;
  name: string;
}

/**
 * Corner space component for GO, Jail, Free Parking, and Go To Jail.
 */
export function CornerSpace({ position, name }: CornerSpaceProps) {
  switch (position) {
    case 0: // GO
      return (
        <div className="flex flex-col items-center justify-center h-full p-2">
          <div className="text-xs font-bold text-red-600 mb-1">COLLECT</div>
          <div className="text-2xl font-bold text-red-600">GO</div>
          <div className="text-xs font-bold text-red-600 mt-1">$200</div>
          <div className="text-xl mt-1">➡️</div>
        </div>
      );

    case 10: // Jail / Just Visiting
      return (
        <div className="flex flex-col items-center justify-center h-full p-1 relative">
          {/* Just Visiting section */}
          <div className="absolute top-1 left-1 text-[7px] font-semibold">
            JUST
            <br />
            VISITING
          </div>

          {/* Jail section */}
          <div className="border-2 border-gray-800 bg-gray-200 p-1 rounded">
            <div className="text-xs font-bold">JAIL</div>
            <div className="text-lg">🔒</div>
          </div>
        </div>
      );

    case 20: // Free Parking
      return (
        <div className="flex flex-col items-center justify-center h-full p-2">
          <div className="text-xs font-bold text-gray-700">FREE</div>
          <div className="text-2xl my-1">🅿️</div>
          <div className="text-xs font-bold text-gray-700">PARKING</div>
        </div>
      );

    case 30: // Go To Jail
      return (
        <div className="flex flex-col items-center justify-center h-full p-2">
          <div className="text-xs font-bold text-gray-700">GO TO</div>
          <div className="text-2xl my-1">👮</div>
          <div className="text-xs font-bold text-gray-700">JAIL</div>
        </div>
      );

    default:
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-xs font-semibold">{name}</div>
        </div>
      );
  }
}
