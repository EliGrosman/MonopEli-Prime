import type { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

/**
 * Main layout wrapper component.
 * Provides consistent page structure with header and content area.
 */
export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-board-border text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold">MonopEli</h1>
          <nav className="flex gap-4">
            <a href="/" className="hover:underline">
              Home
            </a>
            <a href="/lobby" className="hover:underline">
              Lobby
            </a>
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="bg-board-border text-white p-2 text-center text-sm">
        <p>MonopEli - Monopoly Game</p>
      </footer>
    </div>
  );
}
