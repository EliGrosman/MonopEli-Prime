# MonopEli Frontend

A modern React/TypeScript web frontend for MonopEli - a browser-based multiplayer Monopoly game.

## Tech Stack

- **React 19** - UI framework
- **TypeScript 5.9** - Type safety
- **Vite 7** - Build tool and dev server
- **Tailwind CSS 4** - Styling
- **Zustand 5** - State management
- **React Router 7** - Routing
- **Vitest** - Unit testing
- **Playwright** - E2E testing

## Getting Started

### Prerequisites

- Node.js 20+
- npm or pnpm

### Installation

```bash
npm install
```

### Development

```bash
# Start dev server
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
npm run format:check
```

### Testing

```bash
# Unit tests (watch mode)
npm test

# Unit tests with coverage
npm run test:coverage

# Unit tests UI
npm run test:ui

# E2E tests
npm run test:e2e

# E2E tests with UI
npm run test:e2e:ui

# E2E tests (headed browser)
npm run test:e2e:headed

# View E2E test report
npm run test:e2e:report
```

### Building

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── e2e/                    # Playwright E2E tests
│   ├── navigation.e2e.ts   # Navigation tests
│   ├── responsive.e2e.ts   # Responsive design tests
│   └── accessibility.e2e.ts # Accessibility tests
├── src/
│   ├── api/                # API client
│   │   └── client.ts       # Axios HTTP client
│   ├── components/
│   │   ├── board/          # Board components
│   │   │   ├── Board.tsx           # Main board grid
│   │   │   ├── BoardSpace.tsx      # Individual space
│   │   │   ├── PropertySpace.tsx   # Property-specific rendering
│   │   │   ├── CornerSpace.tsx     # Corner spaces
│   │   │   ├── PlayerToken.tsx     # Player tokens
│   │   │   └── HouseIndicator.tsx  # Houses/hotels
│   │   ├── common/         # Shared UI components
│   │   │   ├── Modal.tsx           # Modal dialog
│   │   │   ├── Toast.tsx           # Toast notifications
│   │   │   ├── Loading.tsx         # Loading states
│   │   │   └── ConnectionStatus.tsx # Connection indicator
│   │   ├── game/           # Game-specific components
│   │   │   ├── GamePage.tsx        # Game page
│   │   │   ├── PlayerPanel.tsx     # Player sidebar
│   │   │   ├── PlayerCard.tsx      # Player info card
│   │   │   ├── ActionPanel.tsx     # Action buttons
│   │   │   ├── DiceRoll.tsx        # Dice display
│   │   │   ├── BuildingControls.tsx # Build/mortgage controls
│   │   │   ├── EventLog.tsx        # Game event log
│   │   │   └── MobileControls.tsx  # Mobile navigation
│   │   ├── lobby/          # Lobby components
│   │   │   ├── LobbyPage.tsx       # Lobby page
│   │   │   ├── LobbyRoom.tsx       # Lobby room view
│   │   │   ├── LobbyList.tsx       # Lobby list
│   │   │   ├── LobbyCard.tsx       # Lobby preview card
│   │   │   ├── LobbySettings.tsx   # Settings form
│   │   │   └── PlayerSlot.tsx      # Player slots
│   │   └── property/       # Property components
│   │       ├── PropertyCard.tsx    # Property card
│   │       ├── PropertyModal.tsx   # Property details modal
│   │       └── PropertyList.tsx    # Properties grouped by color
│   ├── hooks/              # Custom React hooks
│   │   ├── useWebSocket.ts         # WebSocket connection
│   │   ├── useSession.ts           # Session management
│   │   ├── useGameState.ts         # Game state hook
│   │   ├── useActions.ts           # Game actions
│   │   ├── useLobbyState.ts        # Lobby state
│   │   ├── useResponsive.ts        # Responsive utilities
│   │   └── useKeyboardNavigation.ts # Keyboard shortcuts
│   ├── pages/              # Route pages
│   │   ├── HomePage.tsx
│   │   ├── LobbyPage.tsx
│   │   └── GamePage.tsx
│   ├── store/              # Zustand stores
│   │   ├── gameStore.ts            # Game state
│   │   ├── sessionStore.ts         # Session state (persisted)
│   │   ├── uiStore.ts              # UI state
│   │   └── lobbyStore.ts           # Lobby state
│   ├── types/              # TypeScript types
│   │   ├── game.ts                 # Game types
│   │   ├── player.ts               # Player types
│   │   ├── property.ts             # Property types
│   │   ├── lobby.ts                # Lobby types
│   │   └── websocket.ts            # WebSocket message types
│   └── utils/              # Utility functions
│       ├── board.ts                # Board layout utilities
│       ├── colors.ts               # Color utilities
│       └── format.ts               # Formatting utilities
├── playwright.config.ts    # Playwright config
├── vite.config.ts          # Vite config
├── tsconfig.json           # TypeScript config
└── tailwind.config.js      # Tailwind config
```

## Features

### Responsive Design

The frontend is fully responsive with breakpoints:
- **Mobile** (< 640px): Compact layout, mobile controls
- **Tablet** (640px - 1023px): Medium layout
- **Desktop** (1024px+): Full layout with sidebars

Use the `useResponsive` hook for responsive logic:

```tsx
import { useResponsive } from '@/hooks';

function MyComponent() {
  const { isMobile, isTablet, isDesktop, isAbove } = useResponsive();

  return isMobile ? <MobileLayout /> : <DesktopLayout />;
}
```

### Keyboard Navigation

Keyboard shortcuts for common actions:

| Key | Action |
|-----|--------|
| `r` | Roll dice |
| `e` | End turn |
| `b` | Buy property |
| `h` | Build house |
| `m` | Mortgage |
| `t` | Trade |
| `Escape` | Close modal |
| `Enter` | Confirm |
| `Shift + ?` | Show help |

Use the `useKeyboardNavigation` hook for custom shortcuts:

```tsx
import { useKeyboardNavigation, createGameShortcuts } from '@/hooks';

function GameControls({ onRollDice, onEndTurn }) {
  useKeyboardNavigation({
    shortcuts: createGameShortcuts({
      onRollDice,
      onEndTurn,
      canRoll: true,
      canEndTurn: false,
    }),
  });
}
```

### Accessibility

The frontend follows WCAG 2.1 guidelines:
- Semantic HTML elements
- ARIA labels and roles
- Keyboard navigation support
- Focus management in modals
- Screen reader announcements for game events

### WebSocket Integration

Real-time game updates via WebSocket:

```tsx
import { useWebSocket } from '@/hooks';

function Game() {
  const { send, connectionState } = useWebSocket({
    url: 'ws://localhost:8000/ws',
    onMessage: (message) => {
      // Handle server messages
    },
  });

  return <ActionPanel send={send} />;
}
```

### State Management

Zustand stores for different concerns:

- **gameStore**: Game state (players, properties, turn info)
- **sessionStore**: User session (persisted to localStorage)
- **uiStore**: UI state (modals, toasts, selected items)
- **lobbyStore**: Lobby state (current lobby, player list)

## Implementation Status

### Week 1-4 Complete
- [x] Vite + React + TypeScript project setup
- [x] Tailwind CSS v4 with custom Monopoly theme
- [x] ESLint + Prettier configuration
- [x] React Router with Home, Lobby, Game pages
- [x] Board component with 40 spaces
- [x] Space components (property, corner, card, tax)
- [x] Player token visualization
- [x] House/hotel indicators
- [x] Zustand stores (game, session, UI, lobby)
- [x] WebSocket connection hook
- [x] Live game state display
- [x] Session management
- [x] Player panel and action system
- [x] Property modals and cards
- [x] Building controls

### Week 5 Complete
- [x] Lobby system components
- [x] Join by code functionality
- [x] AI player slots
- [x] Ready/unready states
- [x] Game event log

### Week 6 Complete
- [x] Responsive design (useResponsive hook)
- [x] Mobile controls component
- [x] Keyboard navigation
- [x] ARIA labels and roles
- [x] Playwright E2E tests
- [x] Documentation

**Total: 348+ unit tests passing**

## Testing

### Unit Tests (Vitest)

348 tests covering:
- Component rendering
- Store actions
- Hook behavior
- Utility functions

```bash
npm test
```

### E2E Tests (Playwright)

Scenarios covering:
- Navigation flows
- Responsive layouts
- Accessibility compliance
- Keyboard navigation
- Modal interactions

```bash
npm run test:e2e
```

## Environment Variables

Create a `.env` file for configuration:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

## Connecting to Backend

The frontend connects to the FastAPI backend (Phase 3). Make sure the backend is running at http://localhost:8000.

## Contributing

1. Follow TypeScript strict mode
2. Write tests for new components
3. Use Tailwind CSS for styling
4. Follow existing code patterns
5. Ensure all tests pass before submitting

## License

MIT
