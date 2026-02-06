import { apiClient } from './client';
import type { GameState } from '@/types';

/**
 * Game list item.
 */
export interface GameListItem {
  game_id: string;
  player_count: number;
  current_turn: number;
  created_at: string;
  is_finished: boolean;
}

/**
 * List all games.
 */
export async function listGames(): Promise<GameListItem[]> {
  const response = await apiClient.get<GameListItem[]>('/games');
  return response.data;
}

/**
 * Get game by ID.
 */
export async function getGame(gameId: string): Promise<GameState> {
  const response = await apiClient.get<GameState>(`/games/${gameId}`);
  return response.data;
}

/**
 * Delete a game.
 */
export async function deleteGame(gameId: string): Promise<void> {
  await apiClient.delete(`/games/${gameId}`);
}

/**
 * Get game state for a player.
 */
export async function getGameState(gameId: string, playerId: number): Promise<GameState> {
  const response = await apiClient.get<GameState>(`/games/${gameId}/state`, {
    params: { player_id: playerId },
  });
  return response.data;
}
