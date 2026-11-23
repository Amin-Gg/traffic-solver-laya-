"""Decision brains for the traffic lights.

Pipeline for every intersection (the same for every brain):

    browser sensors  ->  sensor fusion (pressure per axis)  ->  short English sentence
                     ->  brain answers "ns" or "ew"          ->  browser switches the lights

The sensor layer turns raw counts into qualitative bands ("light", "heavy", ...).
Jev chooses a direction from each intersection's short description and the answers
are cached by description.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

DIRS = ("N", "S", "E", "W")
AXIS_DIRS = {"ns": ("N", "S"), "ew": ("E", "W")}
AXIS_EN = {"ns": "north-south", "ew": "east-west"}

JEV_QUESTION = {
    "green": {
        "type": "choice",
        "instructions": "Give the green light to the busier road; an ambulance always gets green first.",
        "criteria": {"ns": "north-south road", "ew": "east-west road"},
    }
}


# --------------------------------------------------------------------------- sensor fusion

def _get(d: Optional[Dict[str, float]], k: str, default: float = 0.0) -> float:
    try:
        return float((d or {}).get(k, default) or 0.0)
    except (TypeError, ValueError):
        return default


def exit_factor(occupancy: float) -> float:
    """How much of an approach's queue can actually leave: 1 when the road ahead is clear."""
    if occupancy <= 0.5:
        return 1.0
    return max(0.1, (1.0 - occupancy) / 0.5)


def approach_pressure(s: Dict[str, Any], d: str) -> float:
    q = _get(s.get("queues"), d) + 0.4 * _get(s.get("approaching"), d) + _get(s.get("wait"), d) / 20.0
    q *= exit_factor(_get(s.get("downstream"), d))
    if s.get("blocked_dir") == d:
        q *= 0.15
    return q


def axis_pressure(s: Dict[str, Any]) -> Dict[str, float]:
    return {ax: sum(approach_pressure(s, d) for d in ds) for ax, ds in AXIS_DIRS.items()}


def band(p: float) -> str:
    if p < 0.8:
        return "empty"
    if p < 4:
        return "light"
    if p < 9:
        return "moderate"
    if p < 16:
        return "heavy"
    return "severely jammed"


def emergency_axis(s: Dict[str, Any]) -> Optional[str]:
    d = s.get("emergency_dir")
    if d in ("N", "S"):
        return "ns"
    if d in ("E", "W"):
        return "ew"
    return None


BANDS = ("empty", "light", "moderate", "heavy", "severely jammed")


def compose_text(band_ns: str, band_ew: str, busier: Optional[str], ambulance: Optional[str]) -> str:
    text = f"North-south traffic is {band_ns}. East-west traffic is {band_ew}."
    # State the comparison explicitly, including when an ambulance can override it.
    text += f" More cars are waiting on the {AXIS_EN[busier]} road." if busier else " Both roads are equally busy."
    if ambulance:
        text += f" An ambulance is coming on the {AXIS_EN[ambulance]} road."
    return text


def all_texts() -> List[str]:
    """Every sentence the sensor layer can produce (105): small enough to pre-compute once."""
    out = []
    for i, bn in enumerate(BANDS):
        for j, be in enumerate(BANDS):
            options = ["ns" if i > j else "ew"] if i != j else ["ns", "ew", None]
            for busier in options:
                for amb in (None, "ns", "ew"):
                    out.append(compose_text(bn, be, busier, amb))
    return out


def state_text(s: Dict[str, Any]) -> str:
    """The exact sentence Jev reads for one intersection."""
    p = axis_pressure(s)
    busier = None if abs(p["ns"] - p["ew"]) <= 0.3 else ("ns" if p["ns"] > p["ew"] else "ew")
    return compose_text(band(p["ns"]), band(p["ew"]), busier, emergency_axis(s))


BAND_FA = {"empty": "خالی", "light": "خلوت", "moderate": "متوسط", "heavy": "سنگین", "severely jammed": "خیلی سنگین"}
AXIS_FA = {"ns": "شمال–جنوب", "ew": "شرق–غرب"}


def state_fa(s: Dict[str, Any]) -> str:
    """Persian translation of exactly what Jev reads (for the UI)."""
    p = axis_pressure(s)
    parts = [f"خیابان شمال–جنوب {BAND_FA[band(p['ns'])]}", f"خیابان شرق–غرب {BAND_FA[band(p['ew'])]}"]
    if abs(p["ns"] - p["ew"]) > 0.3:
        parts.append(f"صف {AXIS_FA['ns' if p['ns'] > p['ew'] else 'ew']} بیشتر است")
    else:
        parts.append("هر دو یکسان‌اند")
    em = emergency_axis(s)
    if em:
        parts.append(f"🚑 آمبولانس در خیابان {AXIS_FA[em]}")
    return "، ".join(parts) + "."


def reference_axis(s: Dict[str, Any]) -> str:
    """Transparent rule used as a reference line (and as fallback when Jev is unavailable)."""
    em = emergency_axis(s)
    if em:
        return em
    p = axis_pressure(s)
    if abs(p["ns"] - p["ew"]) <= 0.3:
        return s.get("green_axis") or "ns"
    return "ns" if p["ns"] > p["ew"] else "ew"


def reason_fa(s: Dict[str, Any], axis: str) -> str:
    p = axis_pressure(s)
    em = emergency_axis(s)
    if em and em == axis:
        return f"سبز برای {AXIS_FA[axis]}، چون آمبولانس در این خیابان است."
    if abs(p["ns"] - p["ew"]) <= 0.3:
        return f"دو خیابان هم‌اندازه شلوغ‌اند؛ {AXIS_FA[axis]} را انتخاب کرد."
    busier = "ns" if p["ns"] > p["ew"] else "ew"
    if busier == axis:
        return f"سبز برای {AXIS_FA[axis]}، چون صف آن بیشتر است."
    return f"سبز برای {AXIS_FA[axis]}، برخلاف صف بیشتر در {AXIS_FA[busier]}."


def _decision(s: Dict[str, Any], axis: str, probs: Dict[str, float], text: str) -> Dict[str, Any]:
    em = emergency_axis(s)
    return {
        "axis": axis,
        "phase": "EMERGENCY_PRIORITY" if em and em == axis else ("NS_GREEN" if axis == "ns" else "EW_GREEN"),
        "probabilities": probs,
        "confidence": float(probs.get(axis, 0.0)) if probs else 0.0,
        "state_text": text,
        "state_fa": state_fa(s),
        "reference": reference_axis(s),
        "pressure": {k: round(v, 2) for k, v in axis_pressure(s).items()},
        "reason": reason_fa(s, axis),
    }


# --------------------------------------------------------------------------- brains

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_BATCH_SIZE = 4


class JevBrain:
    name = "jev"

    def __init__(self):
        self.model_id = os.getenv("JEV_MODEL", "").strip() or "jev-latest"
        self.api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self.device = "TypeSafe API"
        self.error = None if self.api_key else "Set TYPESAFE_API_KEY to enable Jev."
        self.retry_at = 0.0
        self._lock = asyncio.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.last_stats = {"fresh": 0, "cached": 0, "model_ms": 0.0}
        self.prewarm = {"done": 0, "total": 0}
        self.model_ms_samples: List[float] = []

    @property
    def status(self) -> str:
        return "error" if self.error else "ready"

    def _check_available(self) -> None:
        if not self.api_key or time.monotonic() < self.retry_at:
            raise RuntimeError(self.error or "Jev is temporarily unavailable.")

    def _unavailable(self, message: str, retry_after: float = 10.0) -> RuntimeError:
        self.error = message
        self.retry_at = max(self.retry_at, time.monotonic() + retry_after)
        return RuntimeError(message)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(8.0, connect=3.0),
        )

    async def _predict(self, client: httpx.AsyncClient, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await client.post(JEV_ENDPOINT, json={
                "model": self.model_id, "state": state, "questions": questions,
            })
        except httpx.RequestError as exc:
            raise self._unavailable("Could not reach the TypeSafe API.") from exc

        code = response.status_code
        if code in (400, 413, 422):
            raise ValueError(f"Jev rejected the state or questions (HTTP {code}). Check the question format and limits.")
        if code in (401, 403):
            raise self._unavailable(f"TypeSafe rejected access (HTTP {code}). Check the API key and model access.", 60)
        if code == 402:
            raise self._unavailable("TypeSafe reports insufficient credit (HTTP 402).", 60)
        if code == 429:
            retry_after = response.headers.get("Retry-After", "")
            delay = max(1, int(retry_after)) if retry_after.isdecimal() and len(retry_after) < 8 else 10
            raise self._unavailable("TypeSafe API rate limit reached (HTTP 429).", delay)
        if not response.is_success:
            raise self._unavailable(f"TypeSafe API is unavailable (HTTP {code}).")

        try:
            result = response.json()
            answers = result["answers"]
            if set(answers) != set(questions) or any(
                not isinstance(answer, dict) or answer.get("type") != questions[qid]["type"]
                for qid, answer in answers.items()
            ):
                raise ValueError("Invalid answers")
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise self._unavailable("TypeSafe API returned an invalid answer.") from exc
        if time.monotonic() >= self.retry_at:
            self.error = None
        return result

    async def _green_answer(self, client: httpx.AsyncClient, text: str) -> Dict[str, Any]:
        answer = (await self._predict(client, text, JEV_QUESTION))["answers"]["green"]
        axis = answer.get("choice")
        probabilities = answer.get("probabilities")
        if axis not in ("ns", "ew") or not isinstance(probabilities, dict) or set(probabilities) != set(AXIS_DIRS) or any(
            not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities.values()
        ):
            raise self._unavailable("Jev returned an invalid traffic decision.")
        return answer

    @property
    def model_ms_per_sentence(self) -> Optional[float]:
        m = self.model_ms_samples
        return round(sum(m) / len(m), 1) if m else None

    async def _fill_cache(self, texts: List[str]) -> Tuple[int, float]:
        # Each distinct state gets its own Jev request. A shared lock prevents
        # overlapping city views and prewarm from paying twice for the same state.
        async with self._lock:
            fresh = [t for t in dict.fromkeys(texts) if t not in self._cache]
            if not fresh or not self.api_key or time.monotonic() < self.retry_at:
                return 0, 0.0
            start = time.perf_counter()
            count = 0
            async with self._client() as client:
                for k in range(0, len(fresh), JEV_BATCH_SIZE):
                    chunk = fresh[k:k + JEV_BATCH_SIZE]
                    batch_started = time.perf_counter()
                    answers = await asyncio.gather(
                        *(self._green_answer(client, text) for text in chunk), return_exceptions=True,
                    )
                    for text, answer in zip(chunk, answers):
                        if isinstance(answer, dict):
                            self._cache[text] = answer
                            count += 1
                        elif isinstance(answer, ValueError):
                            self._unavailable(str(answer))
                        elif not isinstance(answer, RuntimeError):
                            raise answer
                    if self.error:
                        break
                    self.model_ms_samples.append((time.perf_counter() - batch_started) * 1000 / len(chunk))
                    del self.model_ms_samples[:-50]
            return count, (time.perf_counter() - start) * 1000

    async def warm_cache(self) -> None:
        if not self.api_key:
            return
        texts = all_texts()
        self.prewarm = {"done": len(self._cache), "total": len(texts)}
        for k in range(0, len(texts), JEV_BATCH_SIZE):
            await self._fill_cache(texts[k:k + JEV_BATCH_SIZE])
            self.prewarm["done"] = len(self._cache)
            if self.error:
                break

    async def predict_raw(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Send typed playground questions to Jev."""
        if not isinstance(state, (str, dict, list)):
            raise ValueError("State must be text, a JSON object, or an array.")
        normalized = {}
        for key, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
                raise ValueError(f"Invalid question {key!r}: choose choice, score, or noul.")
            criteria = question.get("criteria")
            if question["type"] == "choice" and isinstance(criteria, list):
                question = {**question, "criteria": {label: label for label in criteria if isinstance(label, str)}}
            normalized[key] = question
        async with self._lock:
            self._check_available()
            async with self._client() as client:
                return await self._predict(client, state, normalized)

    async def decide(self, states: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict], float, str]:
        start = time.perf_counter()
        texts = [state_text(s) for s in states]
        fresh, model_ms = await self._fill_cache(texts)
        out = {}
        cached = 0
        for s, text in zip(states, texts):
            answer = self._cache.get(text)
            if answer is None:
                axis = reference_axis(s)
                out[s["id"]] = {**_decision(s, axis, {axis: 1.0}, text), "source": "reference-fallback"}
            else:
                out[s["id"]] = _decision(s, answer["choice"], answer["probabilities"], text)
                cached += 1
        self.last_stats = {"fresh": fresh, "cached": cached - fresh, "model_ms": round(model_ms, 1)}
        source = "jev" if cached == len(texts) and (texts or self.api_key) else (
            "jev-partial-fallback" if cached else "reference-fallback"
        )
        return out, (time.perf_counter() - start) * 1000, source


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a >= 0 and b > a:
            return json.loads(raw[a:b + 1])
        raise


class LLMBrain:
    name = "llm"

    def __init__(self):
        self.settings = {"provider": "none", "base_url": "", "model": "", "api_key": ""}
        self.last_error: Optional[str] = None

    def configure(self, settings: Dict[str, str]):
        self.settings.update(settings)

    @property
    def ready(self) -> bool:
        return self.settings.get("provider") in {"ollama", "openai"} and bool(self.settings.get("model"))

    def prompt(self, states: List[Dict[str, Any]]) -> str:
        rows = {s["id"]: state_text(s) for s in states}
        return (
            "You control traffic lights. For each intersection choose which road gets the green light: "
            "\"ns\" (north-south) or \"ew\" (east-west). Give green to the busier road; an ambulance always "
            "gets green first. Return ONLY JSON like {\"decisions\": {\"A1\": \"ns\"}}.\n"
            f"Intersections: {json.dumps(rows)}"
        )

    async def _ollama(self, prompt: str) -> str:
        base = (self.settings.get("base_url") or "http://localhost:11434").rstrip("/")
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(base + "/api/chat", json={
                "model": self.settings["model"], "stream": False, "format": "json",
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def _openai(self, prompt: str) -> str:
        base = (self.settings.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.settings.get("api_key"):
            headers["Authorization"] = "Bearer " + self.settings["api_key"]
        payload = {"model": self.settings["model"], "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}],
                   "response_format": {"type": "json_object"}}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(base + "/chat/completions", headers=headers, json=payload)
            if r.status_code in (400, 404, 422):
                payload.pop("response_format", None)
                r = await c.post(base + "/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def decide(self, states: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict], float, str]:
        if not self.ready:
            return {}, 0.0, "llm-disabled"
        t = time.perf_counter()
        try:
            p = self.prompt(states)
            raw = await (self._ollama(p) if self.settings["provider"] == "ollama" else self._openai(p))
            data = parse_json_object(raw)
            d = data.get("decisions", data)
            out = {}
            for s in states:
                v = d.get(s["id"])
                if isinstance(v, dict):
                    v = v.get("axis") or v.get("green") or v.get("phase")
                v = str(v or "").lower()
                ax = "ns" if v.startswith("ns") or "north" in v else "ew" if v.startswith("ew") or "east" in v else None
                if ax is None:
                    continue  # unusable answer: the light keeps its current state
                out[s["id"]] = _decision(s, ax, {}, state_text(s))
            return out, (time.perf_counter() - t) * 1000, "llm"
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return {}, (time.perf_counter() - t) * 1000, "llm-error"
