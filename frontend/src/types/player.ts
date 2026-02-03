export interface PlayerState {
  id: number;
  name: string;
  money: number;
  position: number;
  inJail: boolean;
  jailTurns: number;
  jailCards: number;
  bankrupt: boolean;
  isAi: boolean;
  color: string;
}

export interface PlayerSession {
  sessionId: string;
  playerId: number;
  displayName: string;
  gameId: string | null;
  lobbyId: string | null;
}
