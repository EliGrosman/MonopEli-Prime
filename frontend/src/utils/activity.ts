import type { GameEventType, GameState } from '@/types';

export function actionEventType(action: string): GameEventType {
  if (action.includes('Trade')) return 'trade';
  if (action.includes('Mortgage') || action.includes('mortgage')) return 'mortgage';
  if (action.includes('Build') || action.includes('Sell')) return 'build';
  if (action.includes('Jail')) return 'jail';
  if (action === 'DeclareBankruptcy') return 'bankrupt';
  if (action === 'RollDice') return 'roll';
  if (action === 'BuyProperty' || action === 'PassBuy') return 'buy';
  return 'move';
}

export function decisionDescription(state: GameState): string {
  const names: Record<string, string> = {
    pre_roll: 'Ready to roll',
    purchase_decision: 'Deciding whether to buy',
    asset_management: 'Managing properties',
    debt_resolution: 'Raising cash to pay a debt',
    trade_response: 'Responding to a trade',
    jail_decision: 'Choosing how to leave jail',
    post_roll: 'Finishing their turn',
    in_jail: 'In jail',
    waiting: 'Waiting for players',
    terminal: 'Game finished',
    game_over: 'Game finished',
    bankrupt: 'Bankrupt',
  };
  return state.gameOver ? 'Game finished' : (names[state.gamePhase] ?? 'Taking their turn');
}
