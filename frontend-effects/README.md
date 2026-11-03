
# FinGuard browser effects

This directory contains a framework-free browser effects module that can be
imported by the separate React/Vite frontend.

## Add to the React frontend

Copy `frontend-effects/finguard-effects.js` into the frontend source tree
(for example, `src/finguard-effects.js`) and import it once from
`src/main.jsx` or `src/main.tsx`:

```js
import "./finguard-effects.js";
```

The module installs:
- A large animated gold orb with slow drift and glow (respects reduced-motion).
- Neon click sparkles (respects reduced-motion).
- A notification stack with a single beep for MEDIUM and three beeps for HIGH.

The gold orb is a fixed visual overlay. It has pointer-events disabled; the
application content stays interactive.

## Connect scan alerts

The backend endpoint remains unchanged:
`POST /api/v1/transactions/scan`.

After receiving the JSON response, pass it to the helper:

```js
import { handleTransactionScanResponse } from "./finguard-effects.js";

const response = await fetch(`${API_BASE}/api/v1/transactions/scan`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ transactions }),
});
if (!response.ok) throw new Error("Transaction scan failed");
const result = await response.json();
handleTransactionScanResponse(result);
```

Or notify an individual alert:

```js
import { notifyFraudRisk } from "./finguard-effects.js";
notifyFraudRisk(alert, { level: "HIGH", currency: "INR" });
```

Only MEDIUM and HIGH risks notify; LOW stays silent. A transaction ID is
deduplicated for the current page session so repeated scans do not repeatedly
sound the same alert. The helper uses the existing alert fields
(`id`/`txn_id`, `amount`, `description`, `receiver`, `reasons`, and
`fraud_risk_level`). If `fraud_risk_level` is absent, the scan-response helper
defaults that alert to MEDIUM because the backend already returned it in
`alerts`.

## Browser sound behaviour

Browsers block autoplay until the user interacts with the page. The first
click/tap on the page unlocks Web Audio in supported browsers; alerts that
arrive before that may show visually without sound. The tone is synthesized
locally and no external audio asset is required.

## Important integration note

This repository is the FastAPI backend and does not contain a React app.
The module is committed here so it can be copied into the actual frontend.
The frontend must import the file and call `handleTransactionScanResponse`
after each scan response. This module does not create a server-side live event
stream; continuous live alerts require the frontend's existing scan/polling
loop or a separately implemented WebSocket/SSE transport.
