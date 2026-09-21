import { test, expect } from '@playwright/test';

for (const scenario of [
  { name: 'jail-fine', button: /Pay 50 dollar fine/ },
  { name: 'jail-card', button: /Use Get Out of Jail Free Card/ },
  { name: 'jail-roll', button: /roll/i },
  { name: 'debt', button: /Mortgage Property · property 39/ },
  { name: 'purchase', button: /Pass on buying/ },
]) {
  test(`live engine/API/browser: ${scenario.name}`, async ({ page, request }) => {
    const response = await request.post(`http://127.0.0.1:18000/__scenario/${scenario.name}`);
    expect(response.ok()).toBeTruthy();
    const { game_id: id, session_id: session } = await response.json();
    await page.addInitScript((sessionId) => {
      localStorage.setItem(
        'monopoly-session',
        JSON.stringify({ state: { sessionId, displayName: 'Player 1' }, version: 0 })
      );
    }, session);
    await page.goto(`/game/${id}`);
    await expect(page.getByRole('button', { name: scenario.button })).toBeEnabled();
    await page.reload();
    await expect(page.getByRole('button', { name: scenario.button })).toBeEnabled();
    await page.getByRole('button', { name: scenario.button }).click();
    await expect
      .poll(
        async () =>
          (await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json()).revision
      )
      .toBe(1);
    const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
    if (scenario.name === 'debt') {
      expect(state.debt).toBeNull();
      expect(state.players[0].money).toBe(100);
      expect(state.properties['39'].mortgaged).toBe(true);
      expect(state.decision_player).toBe(1);
      await expect(page.getByRole('region', { name: 'Debt resolution' })).toHaveCount(0);
    } else if (scenario.name === 'jail-fine') {
      expect(state.players[0].money).toBe(1450);
      expect(state.players[0].in_jail).toBe(false);
    } else if (scenario.name === 'jail-card') {
      expect(state.players[0].jail_cards).toBe(0);
      expect(state.players[0].in_jail).toBe(false);
    } else if (scenario.name === 'jail-roll') {
      expect(state.players[0].in_jail).toBe(true);
      expect(state.players[0].jail_turns).toBe(1);
      expect(state.legal_actions.some((a: { type: string }) => a.type === 'EndTurn')).toBe(true);
    } else {
      expect(state.current_player).toBe(0);
      expect(state.properties['6'].owner).toBeNull();
      await expect(page.getByRole('button', { name: /roll/i })).toBeEnabled();
      await expect(page.getByRole('button', { name: /end.*turn/i })).toHaveCount(0);
    }
  });
}
