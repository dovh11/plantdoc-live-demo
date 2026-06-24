"""
PlantDoc Live Demo — FastAPI Inference Engine
=============================================
Serves the static frontend and exposes a /predict endpoint that runs
YOLOv8-Medium ONNX inference on uploaded plant images.

Architecture
------------
  POST /predict  →  letterbox resize  →  ONNX session  →  NMS  →  JSON
  GET  /          →  static/index.html  (served by StaticFiles mount)
  GET  /health    →  {"status": "ok"}   (used by Docker HEALTHCHECK & cloud LBs)

YOLOv8 ONNX output shape: [1, N, 8400]
  where N = 4 (cx, cy, w, h) + num_classes
  The model was exported with its native head; this code auto-detects N
  and maps class indices to the 28 labels in assets/labels.json.
"""

from __future__ import annotations

import json
import logging
import warnings

# Suppress Pydantic v2 "protected_namespaces" warning that fires from
# FastAPI's internal schema generation (fields prefixed with "model_").
# This is a cosmetic warning only — it does not affect runtime behaviour.
warnings.filterwarnings("ignore", message=".*protected_namespaces.*")

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "assets" / "best.onnx"
LABELS_PATH= BASE_DIR / "assets" / "labels.json"
STATIC_DIR = BASE_DIR / "static"

INPUT_SIZE  = 640          # YOLOv8 default input resolution
MIN_THRESH  = 0.01         # absolute minimum confidence allowed
MAX_THRESH  = 0.95
DEFAULT_THRESH = 0.25
NMS_IOU_THRESHOLD = 0.45   # IoU threshold for NMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plantdoc")


# ─────────────────────────────────────────────────────────────────────────────
# App State (populated on startup)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    session:    ort.InferenceSession | None = None
    labels:     list[str]                   = []
    input_name: str                         = ""
    num_classes: int                        = 0   # detected from model output shape


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load model once at startup, release at shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Loading labels from %s", LABELS_PATH)
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    with open(LABELS_PATH) as f:
        state.labels = json.load(f)
    logger.info("Loaded %d class labels", len(state.labels))

    logger.info("Loading ONNX model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Use CPU provider for maximum compatibility (no CUDA required)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2   # balanced for demo server

    state.session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Introspect the model to get input name and output shape
    inp = state.session.get_inputs()[0]
    out = state.session.get_outputs()[0]
    state.input_name = inp.name

    # Output shape is [1, N, 8400]; N = 4 + num_classes
    # Some ONNX exports have dynamic dims (None) — handle gracefully
    raw_n = out.shape[1]
    if raw_n is None or (isinstance(raw_n, str)):
        # Dynamic dim: infer from label count
        state.num_classes = len(state.labels)
    else:
        state.num_classes = int(raw_n) - 4

    logger.info(
        "Model ready | input=%s shape=%s | output=%s shape=%s | classes=%d",
        inp.name, inp.shape, out.name, out.shape, state.num_classes,
    )
    logger.info("🌱 PlantDoc inference engine is live!")

    yield  # ── application runs here ──────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing ONNX session.")
    state.session = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlantDoc Live Demo",
    description=(
        "Real-time plant disease detection powered by a custom "
        "YOLOv8-Medium ONNX model. Upload a leaf photo and receive "
        "bounding boxes with disease classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open to all origins for this public demo API.
# This allows the Vercel frontend (any subdomain Vercel assigns) to call
# the Render backend without being blocked by the browser's CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing — Letterbox
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target_size: int = INPUT_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize *image* to *target_size*×*target_size* while preserving aspect
    ratio by padding with *color*.

    Returns
    -------
    padded  : resized-and-padded image  (target_size × target_size × 3)
    scale   : scale factor applied to both dimensions
    pad_wh  : (pad_w, pad_h) — pixels added to reach target size
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )
    return padded, scale, (left, top)


def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Convert a BGR OpenCV image to a normalised [1, 3, 640, 640] float32
    tensor ready for ONNX inference.

    Returns the tensor, the letterbox scale, and the (pad_w, pad_h) offsets
    so bounding boxes can be mapped back to original pixel coordinates.
    """
    padded, scale, (pad_w, pad_h) = letterbox(image_bgr)
    rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normed = rgb.astype(np.float32) / 255.0          # [0, 1]
    chw    = np.transpose(normed, (2, 0, 1))          # HWC → CHW
    tensor = np.expand_dims(chw, axis=0)              # CHW → 1CHW
    return tensor, scale, (pad_w, pad_h)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing — YOLOv8 decoding + NMS
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(
    raw_output: np.ndarray,
    orig_h: int,
    orig_w: int,
    scale: float,
    pad_wh: tuple[int, int],
    conf_threshold: float,
    num_classes: int,
    labels: list[str],
) -> list[dict[str, Any]]:
    """
    Decode YOLOv8 ONNX output → list of detection dicts.

    Parameters
    ----------
    raw_output     : ONNX output, shape [1, 4+num_classes, 8400]
    orig_h, orig_w : dimensions of the *original* (pre-letterbox) image
    scale          : letterbox scale factor
    pad_wh         : (pad_w, pad_h) letterbox offsets
    conf_threshold : minimum class confidence
    num_classes    : number of classes encoded in the model head
    labels         : class name list — may be shorter than num_classes
    """
    pad_w, pad_h = pad_wh

    # Shape: [8400, 4 + num_classes]  (transpose from [1, N, 8400])
    predictions = raw_output[0].T          # (8400, N)

    boxes_xywh   : list[list[float]] = []
    scores       : list[float]       = []
    class_ids    : list[int]         = []

    for pred in predictions:
        cx, cy, bw, bh = pred[:4]
        class_scores   = pred[4:4 + num_classes]

        # Only consider classes that exist in our label list
        label_count = min(num_classes, len(labels))
        class_scores = class_scores[:label_count]

        class_id  = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_threshold:
            continue

        # Convert centre-xywh (letterbox space) → xywh for cv2.dnn.NMSBoxes
        # 1. Remove letterbox padding offset
        x_pad = (cx - pad_w) / scale
        y_pad = (cy - pad_h) / scale
        w_orig = bw / scale
        h_orig = bh / scale

        # 2. Top-left corner (for NMS input)
        xmin = x_pad - w_orig / 2
        ymin = y_pad - h_orig / 2

        # Clip to image bounds
        xmin = float(np.clip(xmin, 0, orig_w))
        ymin = float(np.clip(ymin, 0, orig_h))
        w_orig = float(np.clip(w_orig, 0, orig_w - xmin))
        h_orig = float(np.clip(h_orig, 0, orig_h - ymin))

        boxes_xywh.append([xmin, ymin, w_orig, h_orig])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes_xywh:
        return []

    # Non-Maximum Suppression via OpenCV
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=NMS_IOU_THRESHOLD,
    )

    results: list[dict[str, Any]] = []
    for idx in (indices.flatten() if len(indices) > 0 else []):
        x, y, w, h = boxes_xywh[idx]
        xmin = int(round(x))
        ymin = int(round(y))
        xmax = int(round(x + w))
        ymax = int(round(y + h))

        cid   = class_ids[idx]
        label = labels[cid] if cid < len(labels) else f"class_{cid}"

        results.append({
            "class":      label,
            "class_id":   cid,
            "confidence": round(scores[idx], 4),
            "box":        [xmin, ymin, xmax, ymax],   # [x1, y1, x2, y2] in original pixels
        })

    # Sort by confidence descending
    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Lightweight liveness probe used by Docker HEALTHCHECK and cloud LBs."""
    model_ready = state.session is not None
    return JSONResponse(
        content={
            "status":      "ok" if model_ready else "initialising",
            "model_ready": model_ready,
            "num_classes": state.num_classes,
            "labels_count": len(state.labels),
        },
        status_code=200 if model_ready else 503,
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    file:      UploadFile = File(...,  description="Plant leaf image (JPEG/PNG/WebP)"),
    threshold: float      = Form(DEFAULT_THRESH, description="Confidence threshold (0.01–0.95)"),
):
    """
    Run YOLOv8 inference on an uploaded plant image.

    Returns a JSON array of detections:
    ```json
    [
      {"class": "Apple Scab Leaf", "class_id": 0, "confidence": 0.87,
       "box": [xmin, ymin, xmax, ymax]},
      ...
    ]
    ```
    """
    # ── Guards ────────────────────────────────────────────────────────────────
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not ready yet — please retry.")

    threshold = float(np.clip(threshold, MIN_THRESH, MAX_THRESH))

    if file.content_type not in ("image/jpeg", "image/png", "image/webp",
                                  "image/jpg", "image/bmp", "image/tiff"):
        # Be lenient — try to decode whatever arrives; only reject on decode fail
        logger.warning("Unusual content-type received: %s", file.content_type)

    # ── Read & decode image ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    np_arr    = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid JPEG/PNG/WebP.")

    orig_h, orig_w = image_bgr.shape[:2]
    logger.info(
        "Received image | size=%dx%d | content_type=%s | threshold=%.2f",
        orig_w, orig_h, file.content_type, threshold,
    )

    # ── Preprocess ────────────────────────────────────────────────────────────
    tensor, scale, pad_wh = preprocess(image_bgr)

    # ── Inference ─────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        raw_output = state.session.run(
            None,
            {state.input_name: tensor},
        )
    except Exception as exc:
        logger.exception("ONNX inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    t2 = time.perf_counter()

    # raw_output is a list; first element is the detection tensor [1, N, 8400]
    output_tensor = raw_output[0]

    # ── Post-process ─────────────────────────────────────────────────────────
    detections = postprocess(
        raw_output     = output_tensor,
        orig_h         = orig_h,
        orig_w         = orig_w,
        scale          = scale,
        pad_wh         = pad_wh,
        conf_threshold = threshold,
        num_classes    = state.num_classes,
        labels         = state.labels,
    )
    t3 = time.perf_counter()

    logger.info(
        "Inference done | detections=%d | preprocess=%.1fms | inference=%.1fms | postprocess=%.1fms",
        len(detections),
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
        (t3 - t2) * 1000,
    )

    return JSONResponse(content=detections)


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving  (must be mounted AFTER all API routes)
# ─────────────────────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning("static/ directory not found — frontend will not be served.")


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,         # auto-reload on source changes during development
        log_level="info",
    )
