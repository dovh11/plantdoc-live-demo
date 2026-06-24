/**
 * PlantDoc — Runtime Configuration
 *
 * API_BASE_URL controls where fetch() calls are sent.
 *
 * ─── Deployment scenarios ───────────────────────────────────────────────
 *  Vercel (frontend) + Render (backend, proxied):
 *    API_BASE_URL = '/api'
 *    → Vercel edge rewrites /api/* → https://plantdoc-live-demo.onrender.com/*
 *    → Browser only sees same-origin requests (no ad-blocker interference)
 *
 *  Local Docker (unified — backend also serves the frontend):
 *    API_BASE_URL = ''
 *    → fetch('/health') and fetch('/predict') hit the same FastAPI process
 * ────────────────────────────────────────────────────────────────────────
 */
window.PLANTDOC_CONFIG = {
  API_BASE_URL: '/api',
};
