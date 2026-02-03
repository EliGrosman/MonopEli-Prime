# MonopEli Frontend

React/TypeScript web frontend for MonopEli - a browser-based multiplayer Monopoly game.

## Tech Stack

- **React 19** - UI framework
- **TypeScript 5.9** - Type safety
- **Vite 7** - Build tool
- **Tailwind CSS v4** - Styling
- **React Router v7** - Routing
- **Zustand** - State management (coming in Week 2)
- **Vitest** - Testing framework

## Getting Started

### Prerequisites

- Node.js 20+
- npm 10+

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at http://localhost:5173

### Development Commands

```bash
# Start dev server
npm run dev

# Type check
npm run type-check

# Lint
npm run lint

# Lint with auto-fix
npm run lint:fix

# Format code
npm run format

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/         # UI Components
│   │   ├── board/         # Board-related components
│   │   │   ├── Board.tsx  # Main board layout
│   │   │   ├── BoardSpace.tsx
│   │   │   ├── PropertySpace.tsx
│   │   │   ├── CornerSpace.tsx
│   │   │   ├── PlayerToken.tsx
│   │   │   └── HouseIndicator.tsx
│   │   ├── game/          # Game page components
│   │   ├── lobby/         # Lobby components
│   │   ├── home/          # Home page components
│   │   ├── common/        # Shared components
│   │   └── layout/        # Layout components
│   ├── hooks/             # Custom React hooks
│   ├── store/             # Zustand stores
│   ├── types/             # TypeScript types
│   ├── api/               # API client
│   ├── utils/             # Utility functions
│   │   ├── board.ts       # Board layout helpers
│   │   ├── colors.ts      # Player color helpers
│   │   └── format.ts      # Formatting utilities
│   ├── styles/            # Global styles
│   ├── test/              # Test setup
│   ├── App.tsx            # Root component
│   └── main.tsx           # Entry point
├── public/                # Static assets
├── tailwind.config.js     # Tailwind configuration
├── vite.config.ts         # Vite configuration
└── package.json
```

## Implementation Status

### Week 1 ✅ Complete
- [x] Vite + React + TypeScript project setup
- [x] Tailwind CSS v4 with custom Monopoly theme
- [x] ESLint + Prettier configuration
- [x] React Router with Home, Lobby, Game pages
- [x] Board component with 40 spaces
- [x] Space components (property, corner, card, tax)
- [x] Player token visualization
- [x] House/hotel indicators
- [x] 29 component tests passing

### Week 2 (Upcoming)
- [ ] Zustand stores (game, session, UI)
- [ ] WebSocket connection hook
- [ ] Live game state display
- [ ] Session management

### Week 3-6
See `/docs/08_PHASE4_DETAILED_PLAN.md` for full implementation plan.

## Testing

Tests use Vitest with React Testing Library.

```bash
# Run all tests
npm run test

# Run tests in watch mode
npm run test -- --watch

# Run with coverage
npm run test:coverage
```

## Connecting to Backend

The frontend connects to the FastAPI backend (Phase 3). Configure the backend URL:

```bash
# .env.local
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Contributing

1. Follow TypeScript strict mode
2. Write tests for new components
3. Use Tailwind CSS for styling
4. Follow existing code patterns
