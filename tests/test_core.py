import os

os.environ["JEV_PREWARM"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.brains import (JEV_QUESTION, JevBrain, axis_pressure, band, parse_json_object,  # noqa: E402
                        reference_axis, state_text)


def sensors(**kw):
    s = {"id": "A1", "queues": {"N": 0, "S": 0, "E": 0, "W": 0}, "approaching": {}, "wait": {}, "downstream": {},
         "emergency_dir": None, "blocked_dir": None, "green_axis": "ns"}
    s.update(kw)
    return s


def no_model(monkeypatch):
    """Use the reference fallback without contacting TypeSafe."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(main, "jev", JevBrain())


def test_state_text_is_short_and_qualitative():
    s = sensors(queues={"N": 9, "S": 7, "E": 1, "W": 0})
    t = state_text(s)
    assert t == ("North-south traffic is severely jammed. East-west traffic is light. "
                 "More cars are waiting on the north-south road.")
    assert len(t.split()) < 40


def test_comparison_sentence():
    t = state_text(sensors(queues={"N": 3, "S": 3, "E": 2, "W": 3}))
    assert "More cars are waiting on the north-south road." in t
    assert state_text(sensors()).endswith("Both roads are equally busy.")


def test_emergency_sentence_and_reference_priority():
    s = sensors(queues={"N": 0, "S": 0, "E": 10, "W": 10}, emergency_dir="S")
    assert "An ambulance is coming on the north-south road." in state_text(s)
    assert reference_axis(s) == "ns"


def test_full_exit_road_reduces_pressure():
    free = axis_pressure(sensors(queues={"N": 8, "S": 0, "E": 0, "W": 0}))["ns"]
    full = axis_pressure(sensors(queues={"N": 8, "S": 0, "E": 0, "W": 0}, downstream={"N": 0.97}))["ns"]
    assert full < free * 0.2


def test_blocked_approach_reduces_pressure():
    s = sensors(queues={"N": 10, "S": 0, "E": 4, "W": 0}, blocked_dir="N")
    assert reference_axis(s) == "ew"


def test_band_edges():
    assert [band(x) for x in (0, 1, 5, 10, 20)] == ["empty", "light", "moderate", "heavy", "severely jammed"]


def test_question_is_valid_jev_choice():
    q = JEV_QUESTION["green"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"ns", "ew"}


def test_parse_json_object_handles_fences():
    assert parse_json_object('```json\n{"decisions": {"A1": "ns"}}\n```')["decisions"]["A1"] == "ns"
    assert parse_json_object('sure: {"a": 1} ok')["a"] == 1


def test_decide_endpoint_fallback_is_labelled(monkeypatch):
    no_model(monkeypatch)
    c = TestClient(main.app)
    r = c.post("/api/decide", json={"brain": "laya", "intersections": [sensors(queues={"N": 1, "S": 0, "E": 9, "W": 8})]})
    body = r.json()
    assert r.status_code == 200
    assert body["source"] == "reference-fallback"
    d = body["decisions"]["A1"]
    assert d["axis"] == "ew" and d["phase"] == "EW_GREEN"
    assert d["state_text"].startswith("North-south traffic is")


def test_llm_disabled_returns_no_decisions():
    c = TestClient(main.app)
    r = c.post("/api/decide", json={"brain": "llm", "intersections": [sensors()]})
    assert r.json()["source"] == "llm-disabled"
    assert r.json()["decisions"] == {}


def test_status_and_benchmark(monkeypatch):
    no_model(monkeypatch)
    c = TestClient(main.app)
    st = c.get("/api/status").json()
    assert st["jev"]["question"]["type"] == "choice"
    assert st["laya"] == st["jev"]
    b = c.post("/api/benchmark").json()
    assert b["cases"] == 24
    assert b["jev"]["alignment"] == 100.0  # fallback == reference by construction
    assert b["laya"] == b["jev"]
    assert b["llm"] is None


def test_every_live_sentence_is_precomputable():
    import random
    from app.brains import all_texts
    known = set(all_texts())
    rng = random.Random(3)
    for _ in range(3000):
        s = sensors(queues={d: rng.randint(0, 15) for d in "NSEW"}, approaching={d: rng.randint(0, 6) for d in "NSEW"},
                    wait={d: rng.uniform(0, 90) for d in "NSEW"}, downstream={d: rng.uniform(0, 1) for d in "NSEW"},
                    emergency_dir=rng.choice([None, None, "N", "S", "E", "W"]), blocked_dir=rng.choice([None, "N", "E"]))
        assert state_text(s) in known


def test_playground_endpoint_validation_and_no_model(monkeypatch):
    no_model(monkeypatch)
    c = TestClient(main.app)
    assert c.get("/jev").status_code == 200
    assert c.get("/laya").content == c.get("/jev").content
    assert c.post("/api/laya/predict", json={"state": "hi", "questions": {}}).status_code == 400
    r = c.post("/api/laya/predict", json={"state": "hi", "questions": {"q": {"type": "noul", "instructions": "Is it a greeting?"}}})
    assert r.status_code == 503  # Missing credentials are reported by the playground.


def fake_jev(monkeypatch, respond):
    import httpx

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    brain = JevBrain()
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(brain, "_client", lambda: httpx.AsyncClient(
        transport=transport, headers={"Authorization": f"Bearer {brain.api_key}"},
    ))
    monkeypatch.setattr(main, "jev", brain)
    return TestClient(main.app)


def test_jev_decide_uses_provider_answer_and_deduplicates_state(monkeypatch):
    import json
    import httpx

    calls = []

    def respond(req):
        assert str(req.url) == "https://jevtypesafe.org/api/v1/decide"
        assert req.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(req.content)
        assert payload["model"] == "jev"
        assert payload["questions"] == JEV_QUESTION
        calls.append(payload["state"])
        return httpx.Response(200, json={"answers": {
            "green": {"type": "choice", "choice": "ew", "probabilities": {"ns": 0.2, "ew": 0.8}},
        }})

    c = fake_jev(monkeypatch, respond)
    sample = [sensors(queues={"N": 9}), sensors(id="B2", queues={"N": 9})]
    first = c.post("/api/decide", json={"intersections": sample}).json()
    again = c.post("/api/decide", json={"brain": "laya", "intersections": sample}).json()
    assert calls == [state_text(sample[0])]
    assert first["source"] == again["source"] == "jev"
    assert first["stats"]["fresh"] == 1
    assert again["stats"]["cached"] == 2
    assert first["decisions"]["A1"]["axis"] == "ew"
    assert first["decisions"]["A1"]["reference"] == "ns"


def test_playground_forwards_all_three_question_types(monkeypatch):
    import json
    import httpx

    def respond(req):
        body = json.loads(req.content)
        assert body["state"] == {"text": "Refund the double charge today."}
        assert body["questions"]["team"]["criteria"] == {"finance": "finance", "other": "other"}
        assert set(body["questions"]) == {"team", "urgency", "refund"}
        return httpx.Response(200, json={
            "model": "test-jev",
            "answers": {
                "team": {"type": "choice", "choice": "finance", "probabilities": {"finance": 0.9, "other": 0.1}, "confidence": 0.8},
                "urgency": {"type": "score", "score": 0.7, "probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.6},
                "refund": {"type": "noul", "noul": 0.92},
            },
        })

    c = fake_jev(monkeypatch, respond)
    result = c.post("/api/jev/predict", json={
        "state": {"text": "Refund the double charge today."},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": ["finance", "other"]},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
            "refund": {"type": "noul", "instructions": "Is it a refund request?"},
        },
    })
    assert result.status_code == 200
    assert result.json()["model"] == "test-jev"
    assert result.json()["answers"]["refund"]["noul"] == 0.92


def test_rate_limit_uses_reference_and_pauses_api_calls(monkeypatch):
    import httpx

    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(429, headers={"Retry-After": "20"})

    c = fake_jev(monkeypatch, respond)
    for _ in range(2):
        r = c.post("/api/decide", json={"intersections": [sensors(emergency_dir="E")]}).json()
        assert r["source"] == "reference-fallback"
        assert r["decisions"]["A1"]["phase"] == "EMERGENCY_PRIORITY"
        assert "429" in r["error"]
    assert len(calls) == 1
