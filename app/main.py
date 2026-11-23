from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from contextlib import asynccontextmanager, suppress
from secrets import compare_digest
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .brains import DIRS, JEV_QUESTION, JevBrain, LLMBrain, reference_axis, state_text

BASE = Path(__file__).resolve().parent
jev = JevBrain()
llm = LLMBrain()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prewarm = asyncio.create_task(jev.warm_cache()) if os.getenv("JEV_PREWARM") == "1" else None
    try:
        yield
    finally:
        if prewarm:
            prewarm.cancel()
            with suppress(asyncio.CancelledError):
                await prewarm


app = FastAPI(title="Traffic AI Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


# Colab's public tunnel uses a separate password so visitors cannot spend the API key's quota.
if tunnel_password := os.getenv("JEV_TUNNEL_PASSWORD"):
    tunnel_credentials = base64.b64encode(f"colab:{tunnel_password}".encode())

    @app.middleware("http")
    async def protect_tunnel(request, call_next):
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "basic" or not compare_digest(token.encode(), tunnel_credentials):
            return JSONResponse(
                {"error": "Use the session credentials shown in Colab."},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Traffic Solver Colab"'},
            )
        return await call_next(request)


class Sensors(BaseModel):
    """What the browser's road sensors measured at one intersection."""
    id: str
    queues: Dict[str, float] = Field(default_factory=dict)       # stopped cars per approach
    approaching: Dict[str, float] = Field(default_factory=dict)  # moving cars on the approach
    wait: Dict[str, float] = Field(default_factory=dict)         # longest wait per approach (s)
    downstream: Dict[str, float] = Field(default_factory=dict)   # exit-road occupancy 0..1
    emergency_dir: Optional[str] = None
    blocked_dir: Optional[str] = None
    green_axis: Optional[Literal["ns", "ew"]] = None


class DecideRequest(BaseModel):
    brain: Literal["jev", "laya", "llm"] = "jev"
    intersections: List[Sensors]


class LLMSettings(BaseModel):
    provider: str = "none"
    base_url: str = ""
    model: str = ""
    api_key: str = ""


@app.get("/")
def root():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/jev")
@app.get("/laya", include_in_schema=False)
def jev_page():
    return FileResponse(BASE / "static" / "laya.html")


class PlaygroundRequest(BaseModel):
    state: Any
    questions: Dict[str, Any]


@app.post("/api/jev/predict")
@app.post("/api/laya/predict", include_in_schema=False)
async def jev_predict(req: PlaygroundRequest):
    """General-purpose Jev call for the playground page: any text/JSON state, any typed questions."""
    if not req.questions or len(req.questions) > 16:
        return JSONResponse({"error": "بین ۱ تا ۱۶ سؤال بفرست."}, status_code=400)
    if len(json.dumps(req.state, ensure_ascii=False)) > 6000:
        return JSONResponse({"error": "متن ورودی بیش از ۶۰۰۰ نویسه است."}, status_code=400)
    t = time.perf_counter()
    try:
        result = await jev.predict_raw(req.state, req.questions)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    return {"latency_ms": round((time.perf_counter() - t) * 1000, 1), "model": jev.model_id,
            "device": jev.device, **result}


@app.get("/api/status")
def status():
    model = {"status": jev.status, "model": jev.model_id, "device": jev.device, "error": jev.error,
             "question": JEV_QUESTION["green"], "prewarm": jev.prewarm,
             "model_ms_per_sentence": jev.model_ms_per_sentence}
    return {"jev": model, "laya": model,
            "llm": {"ready": llm.ready, "provider": llm.settings.get("provider"), "model": llm.settings.get("model"),
                    "error": llm.last_error}}


@app.post("/api/decide")
async def decide(req: DecideRequest):
    states = [s.model_dump() for s in req.intersections]
    brain = llm if req.brain == "llm" else jev
    decisions, ms, source = await brain.decide(states)
    body: Dict[str, Any] = {"source": source, "latency_ms": round(ms, 1), "decisions": decisions}
    if brain is jev:
        if source != "reference-fallback":
            body["stats"] = jev.last_stats
        if source != "jev":
            body["error"] = jev.error
    if source == "llm-error":
        body["error"] = llm.last_error
    return body


@app.post("/api/llm/settings")
def llm_settings(s: LLMSettings):
    llm.configure(s.model_dump())
    llm.last_error = None
    return {"ok": True, "ready": llm.ready}


@app.post("/api/llm/test")
async def llm_test():
    if not llm.ready:
        return JSONResponse({"ok": False, "error": "LLM تنظیم نشده است."}, status_code=400)
    s = {"id": "T1", "queues": {"N": 12, "S": 8, "E": 2, "W": 1}}
    d, ms, src = await llm.decide([s])
    return {"ok": src == "llm" and "T1" in d, "latency_ms": round(ms, 1), "source": src,
            "decision": d.get("T1"), "error": llm.last_error if src == "llm-error" else None}


def benchmark_cases(n: int = 24, seed: int = 2026) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cases = []
    for k in range(n):
        s = {"id": f"X{k}",
             "queues": {d: rng.randint(0, 12) for d in DIRS},
             "approaching": {d: rng.randint(0, 5) for d in DIRS},
             "wait": {d: rng.uniform(0, 60) for d in DIRS},
             "downstream": {d: rng.uniform(0.1, 0.95) for d in DIRS},
             "green_axis": rng.choice(["ns", "ew"])}
        if k % 6 == 3:
            s["emergency_dir"] = rng.choice(DIRS)
        cases.append(s)
    return cases


@app.post("/api/benchmark")
async def benchmark():
    cases = benchmark_cases()
    ref = {s["id"]: reference_axis(s) for s in cases}
    out: Dict[str, Any] = {"cases": len(cases),
                           "note": "Reference = قاعده‌ی شفاف فشار صف + اولویت امدادی؛ دقت ایمنی واقعی نیست."}
    picks: Dict[str, Dict[str, Any]] = {}
    for name, brain in (("jev", jev), ("llm", llm)):
        if name == "llm" and not llm.ready:
            out[name] = None
            continue
        d, ms, src = await brain.decide(cases)
        picks[name] = d
        ok = sum(1 for s in cases if d.get(s["id"], {}).get("axis") == ref[s["id"]])
        em = [s for s in cases if s.get("emergency_dir")]
        em_ok = sum(1 for s in em if d.get(s["id"], {}).get("axis") == ref[s["id"]])
        out[name] = {"source": src, "latency_ms": round(ms, 1), "per_case_ms": (jev.model_ms_per_sentence if brain is jev and jev.model_ms_per_sentence
                                         else round(ms / len(cases), 1)),
                     "alignment": round(100 * ok / len(cases), 1), "emergency": f"{em_ok}/{len(em)}"}
    out["samples"] = [{"id": s["id"], "text": state_text(s), "reference": ref[s["id"]],
                       **{k: v.get(s["id"], {}).get("axis") for k, v in picks.items()}} for s in cases[:8]]
    out["laya"] = out["jev"]
    for sample in out["samples"]:
        sample["laya"] = sample.get("jev")
    return out
