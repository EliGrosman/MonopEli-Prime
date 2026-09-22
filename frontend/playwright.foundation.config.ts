import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e',
  testMatch: 'foundation-live.ts',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:5300', browserName: 'chromium' },
  webServer: [
    {
      command:
        '../.venv/bin/uvicorn tests.foundation.browser_app:app --app-dir .. --host 127.0.0.1 --port 18000',
      url: 'http://127.0.0.1:18000/health',
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5300',
      url: 'http://127.0.0.1:5300',
      env: { VITE_WS_URL: 'ws://127.0.0.1:18000' },
    },
  ],
});
