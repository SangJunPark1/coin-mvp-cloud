const { chromium } = require("playwright");

(async () => {
  const errors = [];
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  });
  const page = await browser.newPage({ viewport: { width: 1365, height: 768 } });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("file:///C:/Users/tkdwn/Documents/Codex/2026-04-19-3d-r/reports/reset_report.html");
  const desktop = await page.evaluate(() => ({
    title: document.title,
    korean: document.body.innerText.includes("코인 오토 트레이딩 시스템"),
    buttons: [...document.querySelectorAll("[data-chart-window]")].map((button) => button.textContent.trim()),
    overflow: document.documentElement.scrollWidth > window.innerWidth + 2,
  }));

  await page.click('[data-chart-window="60"]');
  const active = await page.textContent(".chart-tabs button.active");

  await page.setViewportSize({ width: 390, height: 844 });
  const mobile = await page.evaluate(() => ({
    width: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    overflow: document.documentElement.scrollWidth > window.innerWidth + 2,
    offscreenSections: [...document.querySelectorAll("section")].filter(
      (section) => section.getBoundingClientRect().right > window.innerWidth + 2
    ).length,
  }));

  await browser.close();
  console.log(JSON.stringify({ desktop, active, mobile, errors }, null, 2));
  if (!desktop.korean || active !== "최근 60" || errors.length > 0 || mobile.offscreenSections > 0) {
    process.exit(1);
  }
})();
