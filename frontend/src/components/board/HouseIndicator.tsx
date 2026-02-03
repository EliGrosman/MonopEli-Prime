interface HouseIndicatorProps {
  houses: number; // 0-4 for houses, 5 for hotel
}

/**
 * House/hotel indicator component.
 * Displays houses (green) or hotel (red) on property color band.
 */
export function HouseIndicator({ houses }: HouseIndicatorProps) {
  if (houses === 0) return null;

  // Hotel (5 houses = 1 hotel)
  if (houses === 5) {
    return (
      <div className="flex justify-center items-center h-full gap-0.5">
        <div
          className="hotel w-3 h-2.5 bg-red-600 rounded-sm border border-red-800"
          data-testid="hotel"
          title="Hotel"
          aria-label="Hotel"
        />
      </div>
    );
  }

  // Houses (1-4)
  return (
    <div className="flex justify-center items-center h-full gap-0.5">
      {Array.from({ length: houses }).map((_, index) => (
        <div
          key={index}
          className="house w-2 h-2 bg-green-600 rounded-sm border border-green-800"
          data-testid="house"
          title={`House ${index + 1}`}
          aria-label="House"
        />
      ))}
    </div>
  );
}
