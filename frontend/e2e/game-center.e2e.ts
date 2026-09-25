import { expect, test, type Locator, type Page, type WebSocketRoute } from '@playwright/test';
import states from './fixtures/foundation.json' with { type: 'json' };

function activity(revision: number) {
  return {
    id: `center-game:${revision}`,
    actor: 1,
    revision,
    turn_number: revision,
    action_type: 'BuyProperty',
    summary: `Opponent bought a property · action ${revision}`,
    details: ['Opponent paid $200 to the bank', 'Opponent moved to Reading Railroad'],
    occurred_at: '2026-09-25T12:00:00Z',
  };
}

function stateFor(count: number) {
  return {
    ...states.paid,
    revision: 100,
    players: Array.from({ length: count }, (_, id) => ({
      ...states.paid.players[0],
      id,
      name: id === 0 ? 'Player 1' : `Opponent ${id} with a very long display name`,
      position: id === 0 ? 0 : 5,
      is_ai: id > 0,
      ai_type: 'jev',
      in_jail: false,
    })),
    properties: Object.fromEntries(
      Object.entries(states.paid.properties).map(([key, prop]) => [key, { ...prop, owner: 0 }])
    ),
    current_player: 0,
    decision_player: 0,
    game_phase: 'pre_roll',
    last_roll: null,
    legal_actions: [
      { type: 'RollDice', player_id: 0 },
      { type: 'MortgageProperty', player_id: 0, property_id: 39 },
    ],
    game_activity: Array.from({ length: 100 }, (_, i) => activity(i + 1)),
  };
}

async function connect(page: Page, initial: Record<string, unknown>) {
  let state = initial;
  let socket: WebSocketRoute;
  await page.addInitScript(() =>
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId: 'center-test', displayName: 'Player 1' }, version: 0 })
    )
  );
  await page.routeWebSocket('**/ws/games/**', (ws) => {
    socket = ws;
    ws.send(JSON.stringify({ type: 'identity', data: { player_id: 0 } }));
    ws.send(JSON.stringify({ type: 'state_update', data: state }));
  });
  await page.goto('/game/center-game');
  await expect(page.locator('.center-layout')).toBeVisible();
  return (next: Record<string, unknown>) => {
    state = next;
    socket.send(JSON.stringify({ type: 'state_update', data: state }));
  };
}

async function insideCenter(page: Page, locator: Locator) {
  await expect(locator).toBeVisible();
  const bounds = await locator.boundingBox();
  const center = await page.locator('.board-center').boundingBox();
  expect(bounds).not.toBeNull();
  expect(center).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(center!.x);
  expect(bounds!.y).toBeGreaterThanOrEqual(center!.y);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(center!.x + center!.width + 1);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(center!.y + center!.height + 1);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(page.viewportSize()!.height);
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
}

for (const [width, height] of [
  [320, 568],
  [390, 844],
  [768, 1024],
  [1366, 768],
  [1920, 1080],
]) {
  for (const count of [2, 4, 8]) {
    for (const colorScheme of ['light', 'dark'] as const) {
      test(`center fits ${width}×${height}, ${count} players, ${colorScheme}`, async ({ page }) => {
        await page.setViewportSize({ width, height });
        await page.emulateMedia({ colorScheme });
        const base = stateFor(count);
        const publish = await connect(page, base);
        await insideCenter(page, page.getByRole('button', { name: 'Roll Dice', exact: true }));
        await insideCenter(
          page,
          page.getByRole('button', { name: 'Manage properties', exact: true })
        );
        const compact = (await page.locator('.board-center').boundingBox())!.height < 420;
        const recent = compact
          ? page.getByRole('button', { name: 'Open activity history' })
          : page.getByRole('log');
        await insideCenter(page, recent);
        await expect(recent).toContainText('action 100');
        if (!compact)
          expect(
            (await page.locator('.center-history').boundingBox())!.height
          ).toBeGreaterThanOrEqual(140);
        expect(
          await page
            .locator('.center-layout')
            .evaluate((el) => getComputedStyle(el).backgroundColor)
        ).toBe('rgb(255, 255, 255)');
        expect(
          await page.locator('.center-layout').evaluate((el) => getComputedStyle(el).color)
        ).toBe('rgb(23, 43, 36)');
        expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
          width
        );

        publish({
          ...base,
          revision: 101,
          game_phase: 'asset_management',
          legal_actions: [
            { type: 'EndTurn', player_id: 0 },
            { type: 'MortgageProperty', player_id: 0, property_id: 39 },
          ],
        });
        await insideCenter(page, page.getByRole('button', { name: 'End your turn' }));
        await page.getByRole('button', { name: 'Manage properties', exact: true }).click();
        await expect(page.getByRole('dialog', { name: 'Manage properties' })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Mortgage', exact: true })).toBeVisible();
        await page.keyboard.press('Escape');

        publish({
          ...base,
          revision: 102,
          game_phase: 'purchase_decision',
          players: base.players.map((p) => ({ ...p, position: p.id === 0 ? 39 : p.position })),
          legal_actions: [
            { type: 'BuyProperty', player_id: 0, property_id: 39 },
            { type: 'PassBuy', player_id: 0 },
          ],
        });
        await insideCenter(page, page.getByRole('button', { name: /Buy Boardwalk/ }));
        await insideCenter(page, page.getByRole('button', { name: /Pass on buying/ }));
        await expect(page.getByRole('group', { name: 'Dice controls' })).toHaveCount(0);

        publish({
          ...base,
          revision: 103,
          game_phase: 'jail_decision',
          players: base.players.map((p) => ({ ...p, in_jail: p.id === 0, jail_cards: 1 })),
          legal_actions: [
            { type: 'RollDice', player_id: 0 },
            { type: 'PayJailFine', player_id: 0 },
            { type: 'UseJailCard', player_id: 0 },
          ],
        });
        await insideCenter(page, page.getByRole('button', { name: /Pay 50 dollar fine/ }));
        await insideCenter(page, page.getByRole('button', { name: /Roll dice to try/ }));
        await insideCenter(page, page.getByRole('button', { name: /Use Get Out/ }));

        publish({
          ...base,
          revision: 104,
          game_phase: 'debt_resolution',
          debt: { debtor: 0, creditor: 1, amount: 200 },
          legal_actions: [{ type: 'MortgageProperty', player_id: 0, property_id: 39 }],
        });
        await insideCenter(
          page,
          page.getByRole('button', { name: /Mortgage Property · Boardwalk/ })
        );
        await insideCenter(page, recent);
        if (count === 8 && colorScheme === 'dark' && [320, 1366].includes(width))
          await page.screenshot({ path: `/tmp/monopeli-center-${width}.png` });
      });
    }
  }
}

test('reading position survives rapid turns, duplicate broadcasts and reconnect', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  const base = stateFor(8);
  const publish = await connect(page, base);
  const log = page.getByRole('log');
  await log.evaluate((el) => {
    el.scrollTop = 160;
    el.dispatchEvent(new Event('scroll'));
  });
  const original = await log.locator('article').first().boundingBox();
  for (let revision = 101; revision <= 130; revision++) {
    const update = { ...base, revision, game_activity: [activity(revision)] };
    publish(update);
    publish(update);
  }
  await expect(page.getByRole('button', { name: 'New activity' })).toBeVisible();
  expect(await log.evaluate((el) => el.scrollTop)).toBe(160);
  expect(await log.locator('article').first().boundingBox()).toEqual(original);
  await expect(log).not.toContainText('action 130');
  await page.getByRole('button', { name: 'New activity' }).click();
  await expect(log.locator('article').first()).toContainText('action 130');
  expect(await log.evaluate((el) => el.scrollTop)).toBe(0);
  await expect(log.locator('[data-event-id="center-game:130"]')).toHaveCount(1);
  await page.reload();
  await expect(page.getByRole('log').locator('[data-event-id="center-game:130"]')).toHaveCount(1);
});
