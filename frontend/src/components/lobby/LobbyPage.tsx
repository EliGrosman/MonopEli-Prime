import { Link } from 'react-router-dom';

/**
 * Lobby page - displays list of available games and options.
 */
export function LobbyPage() {
  return (
    <div className="container mx-auto p-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Game Lobby</h1>
        <Link
          to="/lobby/create"
          className="px-6 py-2 bg-board-border text-white rounded-lg font-semibold hover:bg-green-800 transition-colors"
        >
          Create New Game
        </Link>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Join a Game</h2>

        <div className="mb-6">
          <label htmlFor="gameCode" className="block text-sm font-medium text-gray-700 mb-2">
            Enter Game Code
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              id="gameCode"
              placeholder="Enter 6-digit code"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-board-border focus:border-transparent"
            />
            <button className="px-6 py-2 bg-board-border text-white rounded-lg font-semibold hover:bg-green-800 transition-colors">
              Join
            </button>
          </div>
        </div>

        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Available Games</h3>
          <p className="text-gray-500 text-center py-8">No public games available right now.</p>
        </div>
      </div>
    </div>
  );
}
