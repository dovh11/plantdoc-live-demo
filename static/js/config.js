/**
 * PlantDoc — Runtime Configuration
 *
 * This file is loaded BEFORE app.js. It sets the API backend URL so the
 * static frontend (served from Vercel) can reach the FastAPI backend
 * (hosted separately on Render.com).
 *
 * When running locally (Docker / uvicorn) with the frontend served by the
 * same FastAPI process, set API_BASE_URL to an empty string so all fetch
 * calls use a relative path (same origin).
 *
 * ─── Deployment scenarios ───────────────────────────────────────────────
 *  Local Docker (unified):   API_BASE_URL = ''
 *  Vercel + Render split:    API_BASE_URL = 'https://your-backend.onrender.com'
 * ────────────────────────────────────────────────────────────────────────
 */
window.PLANTDOC_CONFIG = {
  // Replace with your Render.com backend URL when deploying to Vercel.
  // Leave as '' when frontend and backend are served from the same origin.
  API_BASE_URL: 'https://plantdoc-live-demo.onrender.com',
};
