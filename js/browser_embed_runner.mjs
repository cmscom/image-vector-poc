/**
 * Automate Chromium browser to run SigLIP embeddings via Playwright.
 *
 * Usage:
 *   node browser_embed_runner.mjs --image-list <json> --output <json> [--dtype fp32]
 *
 * Requires: npx playwright install chromium
 */

import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';
import { resolve, dirname, extname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const { values: args } = parseArgs({
  options: {
    'image-list': { type: 'string' },
    'output': { type: 'string' },
    'dtype': { type: 'string', default: 'fp32' },
  },
});

if (!args['image-list'] || !args['output']) {
  console.error('Usage: node browser_embed_runner.mjs --image-list <json> --output <json> [--dtype fp32]');
  process.exit(1);
}

async function main() {
  const imageList = JSON.parse(readFileSync(args['image-list'], 'utf-8'));
  const dtype = args['dtype'];
  const htmlPath = resolve(__dirname, 'browser_embed.html');

  console.error(`Images: ${imageList.length}, dtype: ${dtype}`);
  console.error(`HTML: ${htmlPath}`);

  // Launch Chromium
  console.error('Launching Chromium...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto(`file://${htmlPath}`);
  await page.waitForFunction(() => typeof window.runEmbedding === 'function', { timeout: 60000 });
  console.error('Page loaded, runEmbedding function available');

  // Use Playwright route to serve local files via intercepted HTTP requests
  // (file:// fetch is blocked in Chromium, so we route through a fake host)
  const FAKE_HOST = 'http://local-images';
  await page.route(`${FAKE_HOST}/**`, async (route) => {
    const url = route.request().url();
    const filePath = decodeURIComponent(url.replace(`${FAKE_HOST}/`, ''));
    try {
      const body = readFileSync(filePath);
      const ext = extname(filePath).toLowerCase();
      const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp' };
      await route.fulfill({ body, contentType: mimeTypes[ext] || 'image/jpeg' });
    } catch (err) {
      await route.abort('failed');
    }
  });

  const imageUrls = imageList.map(item => `${FAKE_HOST}/${item.file_path}`);

  // Run embedding in browser
  console.error(`Starting embedding for ${imageUrls.length} images...`);
  const startTime = performance.now();

  try {
    const result = await page.evaluate(async ({ urls, dtype }) => {
      return await window.runEmbedding(urls, dtype);
    }, { urls: imageUrls, dtype });

    // Map back to original IDs
    const output = {
      model: result.model,
      dtype: result.dtype,
      count: result.count,
      environment: 'chromium-browser',
      model_load_seconds: result.model_load_seconds,
      processing_time_seconds: result.processing_time_seconds,
      embeddings: result.embeddings.map((emb, i) => ({
        id: imageList[i].id,
        embedding: emb.embedding,
        time_ms: emb.time_ms,
        error: emb.error || null,
      })),
    };

    writeFileSync(args['output'], JSON.stringify(output));
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    console.error(`Done. ${output.count} embeddings saved to ${args['output']} (${elapsed}s)`);
  } catch (err) {
    console.error(`Browser embedding failed: ${err.message}`);
    const browserError = await page.evaluate(() => window.__ERROR__);
    if (browserError) console.error(`Browser error: ${browserError}`);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
