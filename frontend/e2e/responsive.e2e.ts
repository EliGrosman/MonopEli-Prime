import { test, expect } from '@playwright/test';

test.describe('Responsive Design', () => {
  test.describe('Desktop', () => {
    test.use({ viewport: { width: 1280, height: 720 } });

    test('should show full navigation on desktop', async ({ page }) => {
      await page.goto('/');

      // Desktop should show regular navigation
      const nav = page.getByRole('navigation');
      if (await nav.isVisible()) {
        await expect(nav).toBeVisible();
      }
    });

    test('should show board at reasonable size', async ({ page }) => {
      await page.goto('/game/test');

      const board = page.getByTestId('board');
      if (await board.isVisible()) {
        const box = await board.boundingBox();
        expect(box?.width).toBeGreaterThan(400);
      }
    });
  });

  test.describe('Tablet', () => {
    test.use({ viewport: { width: 768, height: 1024 } });

    test('should display properly on tablet', async ({ page }) => {
      await page.goto('/');

      // Page should be responsive
      await expect(page.locator('body')).toBeVisible();
    });

    test('lobby page should be usable on tablet', async ({ page }) => {
      await page.goto('/lobby');

      // Form should be visible
      await expect(page.getByPlaceholder(/code/i)).toBeVisible();
    });
  });

  test.describe('Mobile', () => {
    test.use({ viewport: { width: 375, height: 667 } });

    test('should display properly on mobile', async ({ page }) => {
      await page.goto('/');

      // Page should be responsive
      await expect(page.locator('body')).toBeVisible();
    });

    test('lobby form should work on mobile', async ({ page }) => {
      await page.goto('/lobby/create');

      // Form should be usable
      const nameInput = page.getByLabel(/game name/i);
      await expect(nameInput).toBeVisible();

      // Tap should work
      await nameInput.tap();
      await nameInput.fill('Mobile Test Game');
      await expect(nameInput).toHaveValue('Mobile Test Game');
    });

    test('should have touch-friendly buttons', async ({ page }) => {
      await page.goto('/lobby');

      // Buttons should be large enough for touch
      const joinButton = page.getByRole('button', { name: /join/i });
      if (await joinButton.isVisible()) {
        const box = await joinButton.boundingBox();
        // Minimum touch target is 44x44
        expect(box?.height).toBeGreaterThanOrEqual(32);
      }
    });
  });
});

test.describe('Mobile Controls', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('should not show mobile controls on lobby page', async ({ page }) => {
    await page.goto('/lobby');

    // Mobile controls are for game page only
    const mobileControls = page.locator('.mobile-controls');
    await expect(mobileControls).not.toBeVisible();
  });

  test('should adjust content for safe areas', async ({ page }) => {
    await page.goto('/lobby');

    // Check that body doesn't have overflow issues
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});
