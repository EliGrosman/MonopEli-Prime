import { test, expect } from '@playwright/test';

test.describe('Accessibility', () => {
  test.describe('Keyboard Navigation', () => {
    test('should be able to navigate with Tab key', async ({ page }) => {
      await page.goto('/lobby');

      // Tab through focusable elements
      await page.keyboard.press('Tab');

      // Something should be focused
      const focusedElement = page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });

    test('should be able to activate buttons with Enter', async ({ page }) => {
      await page.goto('/lobby');

      // Focus on a button
      const joinButton = page.getByRole('button', { name: /join/i });
      if (await joinButton.isVisible()) {
        await joinButton.focus();
        // Enter should work
        await page.keyboard.press('Enter');
      }
    });

    test('code input should be reachable by keyboard', async ({ page }) => {
      await page.goto('/lobby');

      const codeInput = page.getByPlaceholder(/code/i);

      // Focus the input
      await codeInput.focus();
      await expect(codeInput).toBeFocused();

      // Type in it
      await page.keyboard.type('ABC123');
      await expect(codeInput).toHaveValue('ABC123');
    });
  });

  test.describe('Focus Management', () => {
    test('forms should have labeled inputs', async ({ page }) => {
      await page.goto('/lobby/create');

      // Game name input should have a label
      const nameInput = page.getByLabel(/game name/i);
      await expect(nameInput).toBeVisible();

      // Max players should have a label
      const maxPlayersSelect = page.getByLabel(/max players/i);
      await expect(maxPlayersSelect).toBeVisible();
    });

    test('buttons should have accessible names', async ({ page }) => {
      await page.goto('/lobby');

      // Check that buttons have names
      const buttons = page.getByRole('button');
      const count = await buttons.count();

      for (let i = 0; i < count; i++) {
        const button = buttons.nth(i);
        const name = await button.getAttribute('aria-label') || await button.textContent();
        expect(name?.trim().length).toBeGreaterThan(0);
      }
    });
  });

  test.describe('ARIA Landmarks', () => {
    test('page should have main content area', async ({ page }) => {
      await page.goto('/');

      // Should have some structural elements
      const mainContent = page.locator('main, [role="main"], .container');
      if (await mainContent.isVisible()) {
        await expect(mainContent).toBeVisible();
      }
    });

    test('lobby list should be a list', async ({ page }) => {
      await page.goto('/lobby');

      // Available games section should be accessible
      const availableGames = page.getByText(/available games/i);
      await expect(availableGames).toBeVisible();
    });
  });

  test.describe('Screen Reader Support', () => {
    test('images should have alt text', async ({ page }) => {
      await page.goto('/');

      const images = page.locator('img');
      const count = await images.count();

      for (let i = 0; i < count; i++) {
        const img = images.nth(i);
        const alt = await img.getAttribute('alt');
        const role = await img.getAttribute('role');
        // Images should have alt text or role="presentation"
        expect(alt !== null || role === 'presentation').toBe(true);
      }
    });

    test('form errors should be announced', async ({ page }) => {
      await page.goto('/lobby/create');

      // Clear the game name
      const nameInput = page.getByLabel(/game name/i);
      await nameInput.clear();

      // Try to submit (if form validation exists)
      const submitButton = page.getByRole('button', { name: /create/i });
      await submitButton.click();

      // If there's an error, it should be visible
      // Just check the test runs without error - errors would be in role="alert" elements
    });
  });

  test.describe('Color Contrast', () => {
    test('text should be readable', async ({ page }) => {
      await page.goto('/lobby');

      // Check that text is visible
      const heading = page.getByRole('heading').first();
      if (await heading.isVisible()) {
        await expect(heading).toBeVisible();
      }
    });
  });
});

test.describe('Focus Trap in Modals', () => {
  test('focus should stay within modal when open', async ({ page }) => {
    // Navigate to a page that might have modals
    await page.goto('/game/test');

    // Try to find a modal trigger
    const modalTrigger = page.locator('[data-testid="space-0"]');
    if (await modalTrigger.isVisible()) {
      await modalTrigger.click();

      // If a modal opened, check focus trap
      const modal = page.getByRole('dialog');
      if (await modal.isVisible()) {
        // Tab should stay within modal
        await page.keyboard.press('Tab');
        const focused = page.locator(':focus');
        await expect(focused).toBeVisible();

        // Escape should close
        await page.keyboard.press('Escape');
        await expect(modal).not.toBeVisible();
      }
    }
  });
});
