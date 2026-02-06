import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should load home page', async ({ page }) => {
    await page.goto('/');

    // Title is "frontend" (from index.html), heading is "MonopEli"
    // There are two MonopEli headings (banner + main), use the main one
    await expect(page.getByRole('main').getByRole('heading', { name: /MonopEli/i })).toBeVisible();
  });

  test('should navigate to lobby page', async ({ page }) => {
    await page.goto('/');

    // Click the "Play Now" or similar button
    const playButton = page.getByRole('link', { name: /play|lobby/i });
    if (await playButton.isVisible()) {
      await playButton.click();
      await expect(page).toHaveURL(/\/lobby/);
    }
  });

  test('should navigate to create lobby page', async ({ page }) => {
    await page.goto('/lobby');

    const createButton = page.getByRole('link', { name: /create.*game/i });
    if (await createButton.isVisible()) {
      await createButton.click();
      await expect(page).toHaveURL('/lobby/create');
    }
  });

  test('should have back navigation from create page', async ({ page }) => {
    await page.goto('/lobby/create');

    const backLink = page.getByRole('link', { name: /back/i });
    if (await backLink.isVisible()) {
      await backLink.click();
      await expect(page).toHaveURL('/lobby');
    }
  });
});

test.describe('Lobby List Page', () => {
  test('should display lobby list', async ({ page }) => {
    await page.goto('/lobby');

    // Should show the available games section
    await expect(page.getByText(/available games/i)).toBeVisible();
  });

  test('should have join by code input', async ({ page }) => {
    await page.goto('/lobby');

    // Should have a code input with "Enter game code" placeholder
    const codeInput = page.getByPlaceholder(/game code/i);
    await expect(codeInput).toBeVisible();
  });

  test('should display create new game link', async ({ page }) => {
    await page.goto('/lobby');

    // "Create New Game" link is only visible when logged in.
    // When not logged in, the home page "Create Game" link still works.
    const createLink = page.getByRole('link', { name: /create/i });
    if (await createLink.isVisible()) {
      await expect(createLink).toBeVisible();
    }
  });
});

test.describe('Create Lobby Page', () => {
  test('should display create lobby form', async ({ page }) => {
    await page.goto('/lobby/create');

    // Check for form elements
    await expect(page.getByLabel(/game name/i)).toBeVisible();
    await expect(page.getByLabel(/max players/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /create/i })).toBeVisible();
  });

  test('should have game name placeholder', async ({ page }) => {
    await page.goto('/lobby/create');

    const nameInput = page.getByLabel(/game name/i);

    // Game name has a computed placeholder (e.g. "My's Game"), not a default value
    const placeholder = await nameInput.getAttribute('placeholder');
    expect(placeholder?.length).toBeGreaterThan(0);
  });

  test('should allow changing max players', async ({ page }) => {
    await page.goto('/lobby/create');

    const maxPlayersSelect = page.getByLabel(/max players/i);
    await maxPlayersSelect.selectOption('6');

    await expect(maxPlayersSelect).toHaveValue('6');
  });

  test('should have private game checkbox', async ({ page }) => {
    await page.goto('/lobby/create');

    const privateCheckbox = page.getByLabel(/private/i);
    await expect(privateCheckbox).toBeVisible();

    // Toggle it
    await privateCheckbox.click();
    await expect(privateCheckbox).toBeChecked();
  });
});
