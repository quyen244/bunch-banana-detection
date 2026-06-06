import base64
import io
import math
import os
import logging
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.modules import Conv, DWConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BananaDetection")


# ---------------------------------------------------------------------------
# Custom modules required to load ultimate_model.pt
# ---------------------------------------------------------------------------

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) / gamma) + (b / gamma)))
        k = t if t % 2 == 1 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


class DCPBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=1.0):
        super().__init__()
        self.dw7 = DWConv(c1, c1, k=7, s=1)
        self.dw3 = DWConv(c1, c1, k=3, s=1)
        self.coratt = CoordAtt(c1, c1)
        self.pw = nn.Conv2d(c1, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_multi_scale = self.dw7(x) + self.dw3(x)
        y = self.pw(self.coratt(x_multi_scale))
        return x + y if self.add else y


class NVQ(nn.Module):
    def __init__(self, c1, c2, n=1, residual=True, e=0.5, shortcut=True):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv((1 + n) * self.c_, c2, 1)
        self.res_flag = residual and (c1 == c2)
        self.m = nn.ModuleList(DCPBlock(self.c_, self.c_, shortcut) for _ in range(n))
        self.eca = ECA(channels=c1)

    def forward(self, x):
        y_list = [self.cv1(x)]
        for m in self.m:
            y_list.append(m(y_list[-1]))
        out = self.cv2(torch.cat(y_list, 1))
        out = self.eca(out)
        return x + out


import sys
import __main__

_CUSTOM_CLASSES = [h_sigmoid, h_swish, CoordAtt, ECA, DCPBlock, NVQ]

for _cls in _CUSTOM_CLASSES:
    # Inject vào __main__ — đây là chỗ pickle tìm khi load .pt
    setattr(__main__, _cls.__name__, _cls)
    sys.modules['__main__'].__dict__[_cls.__name__] = _cls
    # Inject vào ultralytics.nn.tasks để YOLO nhận diện
    setattr(tasks, _cls.__name__, _cls)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/ultimate_model.pt")

logger.info("Loading YOLO model from %s", MODEL_PATH)
try:
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as exc:
    logger.error("Failed to load model: %s", exc)
    raise

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Banana Bunch Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    bunch_count: int
    confidences: List[float]
    annotated_image: str  # base64-encoded JPEG


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (image/*)")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)

    # Cap at 1280px — YOLO internally uses 640px, so anything larger wastes time
    MAX_DIM = 1280
    h, w = img_np.shape[:2]
    scale = min(MAX_DIM / max(h, w), 1.0)
    if scale < 1.0:
        img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    logger.info("Running inference on image size=%s", img_np.shape[:2])
    results = model.predict(img_np, conf=0.4, iou=0.5, max_det=100)
    result = results[0]

    boxes = result.boxes
    confidences: List[float] = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
    bunch_count = len(confidences)

    annotated_bgr = result.plot(labels=False)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode()

    logger.info("Detected %d bunches", bunch_count)
    return PredictResponse(
        bunch_count=bunch_count,
        confidences=confidences,
        annotated_image=encoded,
    )
