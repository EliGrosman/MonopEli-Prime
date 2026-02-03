import { Board } from '@/components/board/Board';

/**
 * Main game page - displays the game board and controls.
 */
export function GamePage() {
  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Board section */}
        <div className="flex-1 flex justify-center items-start">
          <Board />
        </div>

        {/* Sidebar */}
        <div className="lg:w-80 space-y-4">
          {/* Player Panel placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Players</h2>
            <p className="text-gray-500">Players will appear here...</p>
          </div>

          {/* Action Panel placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Actions</h2>
            <p className="text-gray-500">Actions will appear here...</p>
          </div>

          {/* Event Log placeholder */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-semibold mb-4">Event Log</h2>
            <p className="text-gray-500">Game events will appear here...</p>
          </div>
        </div>
      </div>
    </div>
  );
}
