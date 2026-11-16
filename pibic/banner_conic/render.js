// Uso: node render.js  -> gera banner_conic_pibiti.png (1080x1920, RGB, <5 MB)
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await p.goto('file://' + __dirname + '/banner.html');
  await p.waitForTimeout(300);
  await p.screenshot({ path: __dirname + '/banner_conic_pibiti.png' });
  await b.close();
})();
