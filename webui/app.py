"""SubFury v2 web UI — FastAPI backend.

    python webui/app.py            # http://127.0.0.1:8000

Streams the prediction pipeline over Server-Sent Events so the browser
shows beam-search candidates and DNS hits as they happen.

Only run resolution against domains you are authorized to test.
"""

import asyncio
import json
import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "subfury_v2"))

from predict import LABEL_RE, load_model, predict_labels, resolve_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("SUBFURY_MODEL",
                           os.path.join(HERE, "..", "results", "subfury_v2"))

app = FastAPI(title="SubFury v2")
_state = {}


def get_model():
    if "model" not in _state:
        _state["model"], _state["tok"], _state["device"] = load_model(MODEL_DIR)
    return _state["model"], _state["tok"], _state["device"]


class PredictRequest(BaseModel):
    domain: str
    known: list[str] = []
    topn: int = 100
    num_beams: int = 64
    resolve: bool = True
    max_recursion: int = 3


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/status")
def status():
    exists = os.path.exists(os.path.join(MODEL_DIR, "best.pt"))
    info = {"model_loaded": "model" in _state, "model_available": exists}
    if exists and "model" in _state:
        info["device"] = _state["device"]
        info["params_m"] = round(_state["model"].num_params() / 1e6, 1)
    return info


async def run_pipeline(req: PredictRequest):
    loop = asyncio.get_event_loop()
    try:
        yield sse("log", {"msg": "loading model…"})
        model, tok, device = await loop.run_in_executor(None, get_model)
        yield sse("log", {"msg": f"model ready ({model.num_params()/1e6:.1f}M params, {device})"})

        known = set()
        for k in req.known:
            k = k.strip().lower()
            if k.endswith("." + req.domain):
                k = k[: -len(req.domain) - 1]
            if k and LABEL_RE.match(k):
                known.add(k)
        if not known:
            yield sse("error", {"msg": "no valid known subdomains provided"})
            return
        yield sse("log", {"msg": f"conditioning on {len(known)} known labels"})

        total_hits = {}
        rounds = req.max_recursion if req.resolve else 1
        for rnd in range(1, rounds + 1):
            yield sse("round", {"round": rnd, "known": len(known)})
            preds = await loop.run_in_executor(
                None,
                lambda: predict_labels(model, tok, device, sorted(known),
                                       topn=req.topn, num_beams=req.num_beams),
            )
            labels = [p for p, _ in preds]
            yield sse("candidates", {
                "round": rnd,
                "items": [{"label": l, "score": round(s, 3)} for l, s in preds],
            })

            if not req.resolve:
                yield sse("done", {"hits": [], "candidates": len(labels)})
                return

            yield sse("log", {"msg": f"resolving {len(labels)} candidates…"})
            hits = await loop.run_in_executor(None, lambda: resolve_all(labels, req.domain))
            new = {k: v for k, v in hits.items() if k not in total_hits}
            for label, ip in sorted(new.items()):
                yield sse("hit", {"fqdn": f"{label}.{req.domain}", "ip": ip, "round": rnd})
            yield sse("round_done", {
                "round": rnd, "new": len(new), "tested": len(labels),
                "rate": round(len(hits) / max(len(labels), 1), 4),
            })
            total_hits.update(new)
            if not new:
                yield sse("log", {"msg": "no new resolutions — stopping recursion"})
                break
            known |= set(new)

        yield sse("done", {
            "hits": [{"fqdn": f"{k}.{req.domain}", "ip": v} for k, v in sorted(total_hits.items())],
            "total": len(total_hits),
        })
    except Exception as exc:  # surface failures to the UI instead of hanging
        yield sse("error", {"msg": f"{type(exc).__name__}: {exc}"})


@app.post("/api/predict")
async def predict(req: PredictRequest):
    return StreamingResponse(run_pipeline(req), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
