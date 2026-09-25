import { test, expect } from '@playwright/test';

for (const scenario of [
  { name: 'jail-fine', button: /Pay 50 dollar fine/ },
  { name: 'jail-card', button: /Use Get Out of Jail Free Card/ },
  { name: 'jail-roll', button: /roll/i },
  { name: 'debt', button: /Mortgage Property · Boardwalk/ },
  { name: 'trade-debt', button: /Mortgage Property · Boardwalk/ },
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
    if (scenario.name === 'debt' || scenario.name === 'trade-debt') {
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

test('live engine/API/browser: human trade proposal and response', async ({ browser, request }) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/trade');
  expect(response.ok()).toBeTruthy();
  const {
    game_id: id,
    session_id: proposerSession,
    recipient_session_id: recipientSession,
  } = await response.json();

  const proposer = await browser.newPage();
  await proposer.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 1' }, version: 0 })
    );
  }, proposerSession);
  await proposer.goto(`/game/${id}`);
  await proposer.getByRole('button', { name: 'Open trading' }).click();
  await expect(proposer.getByRole('region', { name: 'Compose trade' })).toBeVisible();
  await proposer.getByLabel('Recipient').selectOption('1');
  await proposer.getByRole('checkbox', { name: /Mediterranean Avenue/ }).check();
  await proposer.getByRole('checkbox', { name: /Baltic Avenue/ }).check();
  await proposer.getByRole('button', { name: 'Review and send offer' }).click();
  await expect(proposer.getByRole('region', { name: 'Review trade' })).toBeVisible();
  await proposer.getByRole('button', { name: 'Send offer' }).click();

  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return state.game_phase;
    })
    .toBe('trade_response');

  const recipient = await browser.newPage();
  await recipient.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 2' }, version: 0 })
    );
  }, recipientSession);
  await recipient.goto(`/game/${id}`);
  await expect(recipient.getByText('Your response')).toBeVisible();
  await recipient.reload();
  await expect(recipient.getByRole('button', { name: 'Accept' })).toBeEnabled();
  await recipient.getByRole('button', { name: 'Accept' }).click();

  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return [state.game_phase, state.properties['1'].owner, state.properties['3'].owner];
    })
    .toEqual(['asset_management', 1, 0]);
  await expect(proposer.getByRole('dialog')).toHaveCount(0);
  await expect(proposer.getByRole('button', { name: 'Open trading' })).toBeEnabled();
});

test('live engine/API/browser: human offer receives a bot response', async ({ page, request }) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/trade-human-bot');
  const { game_id: id, session_id: session } = await response.json();
  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 1' }, version: 0 })
    );
  }, session);
  await page.goto(`/game/${id}`);
  await page.getByRole('button', { name: 'Open trading' }).click();
  await page.getByLabel('Recipient').selectOption('1');
  await page.getByRole('checkbox', { name: /Oriental Avenue/ }).check();
  await page.getByRole('checkbox', { name: /Baltic Avenue/ }).check();
  await page.getByRole('button', { name: 'Review and send offer' }).click();
  await page.getByRole('button', { name: 'Send offer' }).click();
  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return [state.game_phase, state.properties['6'].owner, state.properties['3'].owner];
    })
    .toEqual(['asset_management', 1, 0]);
});

test('live engine/API/browser: human rejects a bot offer', async ({ page, request }) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/trade-bot-human');
  const { game_id: id, recipient_session_id: session } = await response.json();
  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 2' }, version: 0 })
    );
  }, session);
  await page.goto(`/game/${id}`);
  await expect(page.getByText('Your response')).toBeVisible();
  await page.getByRole('button', { name: 'Reject' }).click();
  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return [state.properties['6'].owner, state.properties['3'].owner];
    })
    .toEqual([0, 1]);
});

test('live engine/API/browser: human offer receives a guided Jev response', async ({
  page,
  request,
}) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/jev-human-bot');
  const { game_id: id, session_id: session } = await response.json();
  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 1' }, version: 0 })
    );
  }, session);
  await page.goto(`/game/${id}`);
  await expect(page.getByText('Jev')).toBeVisible();
  await page.getByRole('button', { name: 'Open trading' }).click();
  await page.getByLabel('Recipient').selectOption('1');
  await page.getByRole('checkbox', { name: /Mediterranean Avenue/ }).check();
  await page.getByRole('checkbox', { name: /Baltic Avenue/ }).check();
  await page.getByRole('button', { name: 'Review and send offer' }).click();
  await page.getByRole('button', { name: 'Send offer' }).click();
  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return [state.game_phase, state.properties['1'].owner, state.properties['3'].owner];
    })
    .toEqual(['asset_management', 1, 0]);
  await expect(page.getByText(/accepted .* trade/i).first()).toBeVisible();
});

test('live engine/API/browser: human rejects a guided Jev offer after reconnect', async ({
  page,
  request,
}) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/jev-bot-human');
  const { game_id: id, recipient_session_id: session } = await response.json();
  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 2' }, version: 0 })
    );
  }, session);
  await page.goto(`/game/${id}`);
  await expect(page.getByText('Your response')).toBeVisible();
  await expect(page.getByText('Jev')).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Reject' })).toBeEnabled();
  await page.getByRole('button', { name: 'Reject' }).click();
  await expect
    .poll(async () => {
      const state = await (await request.get(`http://127.0.0.1:18000/api/games/${id}`)).json();
      return state.game_phase;
    })
    .toBe('asset_management');
});

test('live engine/API/browser: timeout fallback remains visible after reconnect', async ({
  page,
  request,
}) => {
  const response = await request.post('http://127.0.0.1:18000/__scenario/jev-timeout');
  const { game_id: id, recipient_session_id: session } = await response.json();
  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      'monopoly-session',
      JSON.stringify({ state: { sessionId, displayName: 'Player 2' }, version: 0 })
    );
  }, session);
  await page.goto(`/game/${id}`);
  await page.getByRole('button', { name: /View Player 1 details/ }).click();
  await expect(page.getByText('Fallback: timeout')).toBeVisible();
  await page.reload();
  await page.getByRole('button', { name: /View Player 1 details/ }).click();
  await expect(page.getByText('Fallback: timeout')).toBeVisible();
});
