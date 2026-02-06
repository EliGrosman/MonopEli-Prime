import { Link } from 'react-router-dom';

/**
 * Home page - landing page for the application.
 * Provides options to create or join a game.
 */
export function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-120px)] p-8">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-board-border mb-4">MonopEli</h1>
        <p className="text-xl text-gray-800">The Classic Property Trading Game</p>
      </div>

      <div className="flex flex-col sm:flex-row gap-6">
        <Link
          to="/lobby/create"
          className="px-8 py-4 bg-board-border text-white rounded-lg text-xl font-semibold hover:bg-green-800 transition-colors shadow-lg"
        >
          Create Game
        </Link>
        <Link
          to="/lobby/join"
          className="px-8 py-4 bg-white text-board-border border-2 border-board-border rounded-lg text-xl font-semibold hover:bg-gray-100 transition-colors shadow-lg"
        >
          Join Game
        </Link>
      </div>

      <div className="mt-16 max-w-2xl text-center">
        <h2 className="text-2xl font-semibold mb-4 text-gray-900">How to Play</h2>
        <ul className="text-left space-y-2 text-gray-800">
          <li>1. Create a new game or join an existing lobby</li>
          <li>2. Wait for other players or add AI opponents</li>
          <li>3. Roll the dice and move around the board</li>
          <li>4. Buy properties, build houses, and collect rent</li>
          <li>5. Be the last player standing to win!</li>
        </ul>
      </div>
    </div>
  );
}
