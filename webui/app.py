"""SubFury web UI — FastAPI backend.

    python webui/app.py            # http://127.0.0.1:8000

Streams the prediction pipeline over Server-Sent Events so the browser
shows beam-search candidates and DNS hits as they happen.

Only run resolution against domains you are authorized to test.
"""

import asyncio
import datetime
import json
import os
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "subfury"))

from predict import LABEL_RE, load_model, predict_labels, resolve_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("SUBFURY_MODEL",
                           os.path.join(HERE, "..", "results", "subfury"))

app = FastAPI(title="SubFury")
_state = {}


def get_model():
    if "model" not in _state:
        _state["model"], _state["tok"], _state["device"] = load_model(MODEL_DIR)
        import torch
        meta = torch.load(os.path.join(MODEL_DIR, "best.pt"), map_location="cpu")
        _state["val_loss"] = round(meta.get("val_loss", 0), 4) or None
        _state["train_step"] = meta.get("step")
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


# the wordmark and any other page assets, served from disk so the UI works offline
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/api/status")
def status():
    exists = os.path.exists(os.path.join(MODEL_DIR, "best.pt"))
    info = {"model_loaded": "model" in _state, "model_available": exists}
    if exists and "model" in _state:
        info["device"] = _state["device"]
        info["params_m"] = round(_state["model"].num_params() / 1e6, 1)
    return info


def _load_json(path):
    """Read an artifact off disk, or None if it was never produced. Every caller
    is expected to say "not measured" rather than invent a number."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _eval_results():
    """Measured recall, written by `evaluate.py --out`. Absent until it is run —
    the UI says so rather than showing numbers nobody measured."""
    return _load_json(os.path.join(MODEL_DIR, "eval.json"))


# ── research artifacts ──────────────────────────────────────────────────────
# eval.json only ever compared the model to a wordlist, which is the weakest
# baseline on the shelf. research/run_baselines.py puts every method on one
# shared split with bootstrap CIs, and that is what the model page reports.
# Each artifact is loaded independently and each may be absent: a missing file
# turns into a named gap the UI can render, never into a fabricated number.

REPO = os.path.abspath(os.path.join(HERE, ".."))
RESEARCH_DIR = os.path.join(REPO, "results", "research")
BASELINES_JSON = os.path.join(RESEARCH_DIR, "baselines.json")
SWAP_JSON = os.path.join(RESEARCH_DIR, "swap.json")
TEST_JSONL = os.path.join(REPO, "data", "groups_test.jsonl")
CT_JSONL = os.path.join(REPO, "research", "data", "ct_observations.jsonl")

# the one method every other method is scored against
REFERENCE = "frequency-prior"
MODEL_METHOD = "subfury-beam"
_SHORT_LABEL_CHARS = 2      # "one or two characters", the corpus-divergence cut


def _rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def _baselines():
    """Every ranker on the shared split, trimmed to what the page plots: the
    recall curve with its CI, and the paired difference against the frequency
    prior. Order is preserved so the model can be found by key, not by index."""
    doc = _load_json(BASELINES_JSON)
    if not doc or not doc.get("results"):
        return None
    paired = doc.get("paired_vs_frequency_prior") or {}
    ns = [str(n) for n in doc.get("ns") or []]

    def curve(rec):
        out = {}
        for n in ns:
            point = (rec or {}).get(n)
            if point:
                out[n] = {"mean": point.get("mean"), "ci95": point.get("ci95")}
        return out

    methods = []
    for res in doc["results"]:
        name = res.get("name", "")
        diffs = {}
        for n, d in (paired.get(name) or {}).items():
            diffs[n] = {"diff": d.get("mean_diff"), "ci95": d.get("ci95"),
                        "p": d.get("boot_p_two_sided"),
                        "significant": d.get("excludes_zero")}
        methods.append({
            "key": name,
            "is_model": name == MODEL_METHOD,
            "is_reference": name == REFERENCE,
            "map": res.get("map"),
            "recall": curve(res.get("recall")),
            "paired": diffs or None,
        })
    return {
        "source": _rel(BASELINES_JSON),
        "ns": doc.get("ns") or [],
        "apexes": doc.get("apexes"),
        "held_out_labels": doc.get("held_out_labels"),
        "protocol": doc.get("protocol"),
        "bootstrap_rounds": doc.get("bootstrap_rounds"),
        "ci": doc.get("ci"),
        "reference": REFERENCE,
        "model": MODEL_METHOD,
        "methods": methods,
    }


def _swap():
    """The ablation that shows the conditioning is doing the work: rank the same
    apex against another organisation's labels and see what is left."""
    doc = _load_json(SWAP_JSON)
    summary = (doc or {}).get("summary")
    if not summary or not summary.get("per_variant"):
        return None
    return {
        "source": _rel(SWAP_JSON),
        "apexes": summary.get("apexes"),
        "budgets": summary.get("budgets") or [],
        "variants": summary.get("per_variant"),
        "own_minus_swapped": summary.get("own_minus_swapped") or {},
    }


def _shape(source, apexes, pools):
    """Short-label share and the labels shared by the most organisations, over a
    list of per-apex label sets. Deduplicated per apex then pooled, so one
    heavily-certificated domain cannot set the shape of the corpus."""
    import collections
    counts = collections.Counter()
    total = short = 0
    for labels in pools:
        for label in labels:
            counts[label] += 1
            total += 1
            if len(label) <= _SHORT_LABEL_CHARS:
                short += 1
    if not total:
        return None
    return {"source": source, "apexes": apexes, "labels": total,
            "short": short, "short_share": short / total,
            "top": [w for w, _c in counts.most_common(8)]}


def _harness_holdout():
    """The exact labels the recall numbers are scored against: apexes with at
    least `min_labels`, split ~50/50 by an RNG seeded from sha256("<seed>:<apex>")
    — `research/harness.py:make_cases`, mirrored here so the corpus figure and
    the recall figure describe the same population. The holdout is a set, as it
    is there, so a label repeated within one apex is counted once."""
    import hashlib
    import numpy as np
    doc = _load_json(BASELINES_JSON) or {}
    seed = doc.get("seed", 1337)
    min_labels = doc.get("min_labels", 6)
    pools, apexes = [], 0
    try:
        with open(TEST_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    labels = sorted({str(x) for x in rec["labels"] if x})
                except (ValueError, KeyError, TypeError):
                    continue
                if len(labels) < min_labels:
                    continue
                h = hashlib.sha256(f"{seed}:{rec['apex']}".encode()).digest()[:8]
                rng = np.random.default_rng(int.from_bytes(h, "big"))
                shuffled = [labels[i] for i in rng.permutation(len(labels))]
                pools.append(set(shuffled[len(shuffled) // 2:]))
                apexes += 1
    except (OSError, ImportError):
        return None
    out = _shape(_rel(TEST_JSONL), apexes, pools)
    if not out:
        return None
    # the split is reproduced, not trusted: if it does not land on the apex and
    # label counts the artifact recorded, the page says so instead of showing it
    out["expected"] = {"apexes": doc.get("apexes"),
                       "labels": doc.get("held_out_labels")}
    out["matches_artifact"] = (out["apexes"] == doc.get("apexes")
                               and out["labels"] == doc.get("held_out_labels"))
    out["min_labels"] = min_labels
    return out


def _ct_shape():
    """The same count over hostnames real organisations publish to Certificate
    Transparency. First-level labels only — `a.b.example.com` is one host under
    `b`, and counting `a.b` as its own label would compare a different unit."""
    pools, apexes = [], 0
    try:
        with open(CT_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    labels = {str(o["label"]) for o in rec["observations"]
                              if o.get("label") and "." not in str(o["label"])}
                except (ValueError, KeyError, TypeError):
                    continue
                if not labels:
                    continue
                pools.append(labels)
                apexes += 1
    except OSError:
        return None
    return _shape(_rel(CT_JSONL), apexes, pools)


def _corpus():
    """Why benchmark recall and real-target yield diverge: the labels the
    benchmark scores against and the hostnames real targets actually expose are
    not the same distribution, and the short-label share is where it shows."""
    train, real = _harness_holdout(), _ct_shape()
    if not train and not real:
        return None
    return {"cut": _SHORT_LABEL_CHARS, "train": train, "real": real}


def _research():
    """The three research artifacts the model page reports from, each absent-safe.
    `missing` names the files that were not on disk so the UI can say which
    evidence is unavailable instead of quietly dropping a claim."""
    out = {"baselines": _baselines(), "swap": _swap(), "corpus": _corpus()}
    missing = []
    if out["baselines"] is None:
        missing.append(_rel(BASELINES_JSON))
    if out["swap"] is None:
        missing.append(_rel(SWAP_JSON))
    if out["corpus"] is None:
        missing.append(_rel(TEST_JSONL) + " / " + _rel(CT_JSONL))
    out["missing"] = missing
    return out


def _research_cached():
    """The corpus counts walk ~650 KB of JSONL; the artifacts do not change while
    the server is up, so read them once. A run that produced no artifact at all
    is not cached, so restarting the research and reloading the page is enough."""
    if "research" not in _state:
        doc = _research()
        if doc["baselines"] or doc["swap"] or doc["corpus"]:
            _state["research"] = doc
        return doc
    return _state["research"]


def _count_lines(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _runtime():
    """Where the model is actually executing — CUDA and driver versions included,
    so "cuda" in the header is checkable rather than a claim."""
    import platform
    import torch
    info = {"torch": torch.__version__.split("+")[0],
            "python": platform.python_version(),
            "backend": "cpu"}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(backend="cuda", cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    capability=f"{p.major}.{p.minor}",
                    vram_gb=round(p.total_memory / 1e9, 1))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["backend"] = "mps"
    return info


ARCHITECTURE = "nanoGPT-style causal decoder (pre-LN, SDPA, tied head)"
# a known set can be thousands of labels; the trace keeps the head of it so a
# stored journey stays a readable record rather than a copy of the input
KNOWN_IN_TRACE = 200

PROJECT = {
    "name": "SubFury",
    "version": "2.0",
    "author": "4riful",
    "repo": "https://github.com/4riful/subfury",
    "license": "MIT",
}


@app.get("/api/model")
def model_card():
    """Real provenance read from the checkpoint — never hardcoded."""
    ckpt_path = os.path.join(MODEL_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        return {"available": False}
    model, tok, device = get_model()
    cfg = model.cfg
    data_dir = os.path.join(HERE, "..", "data")
    return {
        "available": True,
        "architecture": ARCHITECTURE,
        "params": model.num_params(),
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "block_size": cfg.block_size,
        "vocab_size": tok.get_vocab_size(),
        "tokenizer": "BPE trained on subdomain labels",
        "device": device,
        "val_loss": _state.get("val_loss"),
        "train_step": _state.get("train_step"),
        "train_apexes": _count_lines(os.path.join(data_dir, "groups_train.jsonl")),
        "test_apexes": _count_lines(os.path.join(data_dir, "groups_test.jsonl")),
        "objective": "known_1 [SEP] … [DELIM] target [END], loss on target only",
        "decoding": "deterministic beam search",
        "eval": _eval_results(),
        "research": _research_cached(),
        "runtime": _runtime(),
        "project": PROJECT,
    }


class TokenizeRequest(BaseModel):
    known: list[str] = []


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    """Expose the conditioned prefix the model actually receives."""
    _, tok, _ = get_model()
    sep, delim = tok.token_to_id("[SEP]"), tok.token_to_id("[DELIM]")
    labels, ids = [], []
    for i, raw in enumerate(req.known):
        lab = raw.strip().lower()
        if not lab or not LABEL_RE.match(lab):
            continue
        if i:
            ids.append(sep)
        enc = tok.encode(lab)
        labels.append({"label": lab, "tokens": enc.tokens, "ids": enc.ids})
        ids.extend(enc.ids)
    ids.append(delim)
    seq = []
    for t in ids:
        seq.append("[SEP]" if t == sep else "[DELIM]" if t == delim
                   else tok.id_to_token(t))
    return {"labels": labels, "sequence": seq, "length": len(ids)}


# ── passive seeding ─────────────────────────────────────────────────────────
# Certificate-transparency and passive-DNS sources, queried concurrently. Any
# single one of these fails often — crt.sh in particular is slow and regularly
# 502s — so a seed only fails when every source does, and the UI is told which
# ones answered.

SEED_TIMEOUT = 20          # overall budget, seconds
SEED_PER_SOURCE = 18       # per-request timeout, seconds


def _http(url, headers=None, timeout=SEED_PER_SOURCE, retries=1):
    """These archives 502 and 503 as a matter of course; one quick retry turns a
    large share of those into answers."""
    import urllib.error
    import urllib.request
    hdrs = {"User-Agent": "subfury/2.0", "Accept": "application/json, text/plain, */*"}
    hdrs.update(headers or {})
    last = None
    for attempt in range(retries + 1):
        try:
            rq = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                raise                      # 404 or a hard rejection: do not retry
        except Exception as exc:
            last = exc
        if attempt < retries:
            time.sleep(0.6)
    raise last


def _loads(body):
    """crt.sh has historically returned both a JSON array and a bare stream of
    concatenated objects. Accept either."""
    try:
        return json.loads(body)
    except ValueError:
        return json.loads("[" + body.strip().replace("}\n{", "},{").replace("}{", "},{") + "]")


def _src_crtsh(domain):
    import urllib.parse
    q = urllib.parse.quote("%." + domain)
    # The full history is worth having, but on a big apex it is megabytes and
    # crt.sh regularly stalls on it. Fall back to unexpired certs only, which is
    # ~10x smaller and answers in about a second.
    attempts = [(f"https://crt.sh/?q={q}&output=json", 10),
                (f"https://crt.sh/?q={q}&output=json&exclude=expired", 8)]
    last = None
    for url, timeout in attempts:
        try:
            rows = _loads(_http(url, timeout=timeout))
            break
        except Exception as exc:
            last = exc
    else:
        raise last
    names = []
    for row in rows:
        # name_value holds the SANs (newline separated); common_name is separate
        # and is sometimes the only place a host appears.
        names += str(row.get("name_value", "")).split("\n")
        names.append(str(row.get("common_name", "")))
    return names


def _src_certspotter(domain):
    rows = _loads(_http("https://api.certspotter.com/v1/issuances?domain=" + domain
                        + "&include_subdomains=true&expand=dns_names"))
    names = []
    for row in rows:
        names += list(row.get("dns_names") or [])
    return names


class QuotaExceeded(RuntimeError):
    """The source said it is done with us for today, not that it broke."""


def _src_hackertarget(domain):
    # HackerTarget's free tier is ~50 calls a day per IP. They sell a key that
    # lifts it; set HACKERTARGET_API_KEY and it is used automatically.
    key = os.environ.get("HACKERTARGET_API_KEY", "")
    url = "https://api.hackertarget.com/hostsearch/?q=" + domain
    if key:
        url += "&apikey=" + key
    body = _http(url)
    if "API count exceeded" in body or "API limit" in body:
        raise QuotaExceeded("daily quota spent" + ("" if key else " — set HACKERTARGET_API_KEY to lift it"))
    if "error" in body[:40].lower():
        raise RuntimeError(body.strip()[:80])
    return [line.split(",")[0] for line in body.splitlines() if line.strip()]


def _src_wayback(domain):
    body = _http("https://web.archive.org/cdx/search/cdx?url=*." + domain
                 + "&output=text&fl=original&collapse=urlkey&limit=20000")
    names = []
    for line in body.splitlines():
        host = line.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        if host:
            names.append(host)
    return names


def _src_subdomaincenter(domain):
    """Aggregator that answers in under a second — the one source that has not
    let us down while crt.sh 502s and hackertarget burns its daily quota."""
    return _loads(_http("https://api.subdomain.center/?domain=" + domain, timeout=12))


def _src_rapiddns(domain):
    """HTML, not an API: the hostnames sit in the first cell of each table row."""
    import re as _re
    body = _http("https://rapiddns.io/subdomain/" + domain + "?full=1", timeout=12)
    return _re.findall(r"<td>\s*([A-Za-z0-9_.*-]+\.%s)\s*</td>" % _re.escape(domain), body)


def _src_urlscan(domain):
    import urllib.parse
    doc = _loads(_http("https://urlscan.io/api/v1/search/?q="
                       + urllib.parse.quote("domain:" + domain) + "&size=200", timeout=16))
    out = []
    for r in doc.get("results", []):
        for k in ("page", "task"):
            d = (r.get(k) or {}).get("domain")
            if d:
                out.append(d)
    return out


def _src_otx(domain):
    key = os.environ.get("OTX_API_KEY", "")
    doc = _loads(_http("https://otx.alienvault.com/api/v1/indicators/domain/"
                       + domain + "/passive_dns", headers={"X-OTX-API-KEY": key}))
    return [r.get("hostname", "") for r in doc.get("passive_dns", [])]


# tier 1 runs on every seed; tier 2 is metered, so it is only spent when tier 1
# came back thin. 25 labels is the line: below it a seed is not worth running.
SEED_SOURCES = [
    ("crt.sh", _src_crtsh, 1),
    ("certspotter", _src_certspotter, 1),
    ("subdomain.center", _src_subdomaincenter, 1),
    ("rapiddns", _src_rapiddns, 1),
    ("urlscan", _src_urlscan, 1),
    ("wayback", _src_wayback, 1),
    ("hackertarget", _src_hackertarget, 2),
]
SEED_TIER2_BELOW = 25

# a source that says "quota spent" is believed until the next UTC day
_SOURCE_COOLDOWN = {}


def _cooling(name):
    until = _SOURCE_COOLDOWN.get(name)
    return until if until and until > time.time() else None


def _cool_down(name):
    tomorrow = (datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                      microsecond=0)
    _SOURCE_COOLDOWN[name] = tomorrow.timestamp()
    return tomorrow
# Keyless by default; OTX only joins the rotation when a key is present, so an
# unconfigured source never shows up as a failure.
if os.environ.get("OTX_API_KEY"):
    SEED_SOURCES.append(("otx", _src_otx, 1))


def _labels_from(names, domain):
    """Hostnames → the labels below `domain`. Drops wildcards, the apex itself,
    and anything the model's label grammar rejects."""
    out = set()
    suffix = "." + domain
    for name in names:
        name = str(name).strip().lower().rstrip(".")
        while name.startswith("*."):
            name = name[2:]
        if not name.endswith(suffix):
            continue
        label = name[: -len(suffix)]
        if label and LABEL_RE.match(label):
            out.add(label)
    return out


def _seed_valid(domain):
    domain = domain.strip().lower().strip(".")
    return domain if domain and LABEL_RE.match(domain) and "." in domain else None


def _seed_module(name):
    """The function that actually queried this source, named the way the trace
    names every other step: file:function."""
    for src, fn, _tier in SEED_SOURCES:
        if src == name:
            return "webui/app.py:" + fn.__name__
    return "webui/app.py:_seed_one"


async def _seed_one(name, fn, domain):
    """One source, bounded and never allowed to take the others down with it.
    Every outcome carries its module and how long it took, so a seed is as
    traceable as a run."""
    t0 = time.monotonic()
    stamp = lambda d: dict(d, module=_seed_module(name),
                           ms=round((time.monotonic() - t0) * 1000, 1))
    until = _cooling(name)
    if until:
        mins = int((until - time.time()) / 60)
        return stamp({"name": name, "ok": False, "skipped": True,
                      "error": f"quota spent · resets in {mins // 60}h {mins % 60}m"}), set()
    loop = asyncio.get_event_loop()
    try:
        names = await asyncio.wait_for(
            loop.run_in_executor(None, fn, domain), SEED_TIMEOUT)
        found = _labels_from(names, domain)
        return stamp({"name": name, "ok": True, "count": len(found)}), found
    except asyncio.TimeoutError:
        return stamp({"name": name, "ok": False, "error": "timed out"}), set()
    except QuotaExceeded as exc:
        # remember it, so the next seed does not spend a second finding out again
        when = _cool_down(name)
        return stamp({"name": name, "ok": False, "skipped": True,
                      "error": f"{exc} · retrying after {when:%H:%M} UTC"}), set()
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        return stamp({"name": name, "ok": False, "error": msg[:120]}), set()


_SEED_CACHE = {}          # domain -> (timestamp, sources, labels)
SEED_CACHE_TTL = 180


@app.get("/api/seed/stream")
async def seed_stream(domain: str):
    """Same sources, reported as each one lands — so the UI can show the work
    instead of a spinner that hides four separate queries."""
    d = _seed_valid(domain)

    async def gen():
        import time
        if not d:
            yield sse("error", {"msg": "not a valid apex domain"})
            return
        yield sse("start", {"domain": d,
                            "sources": [n for n, _f, _t in SEED_SOURCES]})

        hit = _SEED_CACHE.get(d)
        if hit and time.time() - hit[0] < SEED_CACHE_TTL:
            for st in hit[1]:
                yield sse("source", dict(st, cached=True))
            yield sse("done", {"labels": hit[2], "sources": hit[1], "cached": True,
                               "ok": bool(hit[2]), "error": None})
            return

        labels, sources = set(), []

        async def drain(group):
            nonlocal labels
            pending = [asyncio.create_task(_seed_one(n, f, d)) for n, f, _ in group]
            for coro in asyncio.as_completed(pending):
                status, found = await coro
                fresh = found - labels
                labels |= found
                status["new"] = len(fresh)
                sources.append(status)
                yield status

        async for st in drain([x for x in SEED_SOURCES if x[2] == 1]):
            yield sse("source", st)

        tier2 = [x for x in SEED_SOURCES if x[2] == 2]
        if tier2:
            if len(labels) >= SEED_TIER2_BELOW:
                for n, _f, _t in tier2:
                    st = {"name": n, "ok": False, "skipped": True, "new": 0,
                          "module": _seed_module(n), "ms": 0.0,
                          "error": f"held back — {len(labels)} labels already found"}
                    sources.append(st)
                    yield sse("source", st)
            else:
                async for st in drain(tier2):
                    yield sse("source", st)
        live = [x for x in sources if x["ok"]]
        out = sorted(labels)
        _SEED_CACHE[d] = (time.time(), sources, out)
        yield sse("done", {"labels": out, "sources": sources,
                           "ok": bool(labels),
                           "error": None if labels else (
                               "no source returned a subdomain for this apex" if live
                               else "every source failed")})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/seed")
async def seed(domain: str):
    """Seed known subdomains from passive sources — no traffic to the target."""
    domain = _seed_valid(domain)
    if not domain:
        return {"ok": False, "error": "not a valid apex domain", "labels": [],
                "sources": []}

    tier1 = [x for x in SEED_SOURCES if x[2] == 1]
    results = list(await asyncio.gather(*(_seed_one(n, f, domain) for n, f, _t in tier1)))
    found = set().union(*[r[1] for r in results]) if results else set()
    for n, fn, _t in [x for x in SEED_SOURCES if x[2] == 2]:
        if len(found) >= SEED_TIER2_BELOW:
            results.append(({"name": n, "ok": False, "skipped": True,
                             "module": _seed_module(n), "ms": 0.0,
                             "error": f"held back — {len(found)} labels already found"}, set()))
        else:
            r = await _seed_one(n, fn, domain)
            found |= r[1]
            results.append(r)

    labels = set()
    sources = []
    for status, found in results:
        sources.append(status)
        labels |= found

    if labels:
        return {"ok": True, "labels": sorted(labels), "sources": sources}
    live = [s for s in sources if s["ok"]]
    error = ("no source returned a subdomain for this apex" if live else
             "every source failed: "
             + "; ".join(f"{s['name']} ({s.get('error')})" for s in sources))
    return {"ok": False, "error": error, "labels": [], "sources": sources}


# ── per-run provenance ────────────────────────────────────────────────────
# Every event says which module produced it and how far into the run it landed,
# so the journey the UI stores is a record of what happened rather than a
# reconstruction from the order things appeared on screen.

MOD_PIPELINE = "webui/app.py:run_pipeline"
MOD_PREDICT = "subfury/predict.py:predict_labels"
MOD_DECODE = "subfury/model.py:beam_search"
MOD_RESOLVE = "subfury/predict.py:resolve_all"


async def run_pipeline(req: PredictRequest):
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()

    def out(event, data, module=MOD_PIPELINE):
        """One SSE frame, stamped with its source module and its offset into the
        run. Both are facts the browser cannot work out for itself."""
        return sse(event, dict(data, module=module,
                               t_ms=round((time.monotonic() - t0) * 1000, 1)))

    try:
        yield out("log", {"msg": "loading model…"})
        model, tok, device = await loop.run_in_executor(None, get_model)
        yield out("log", {"msg": f"model ready ({model.num_params()/1e6:.1f}M params, {device})"})

        known = set()
        for k in req.known:
            k = k.strip().lower()
            if k.endswith("." + req.domain):
                k = k[: -len(req.domain) - 1]
            if k and LABEL_RE.match(k):
                known.add(k)
        if not known:
            yield out("error", {"msg": "no valid known subdomains provided"})
            return

        # the run's own header: what is executing, on what, with which settings
        cfg = model.cfg
        yield out("meta", {
            "domain": req.domain,
            "started": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
            "model": {
                "params": model.num_params(),
                "architecture": ARCHITECTURE,
                "device": device,
                "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
                "block_size": cfg.block_size, "vocab_size": tok.get_vocab_size(),
                "checkpoint": _rel(os.path.join(MODEL_DIR, "best.pt")),
                "tokenizer": _rel(os.path.join(MODEL_DIR, "tokenizer.json")),
                "val_loss": _state.get("val_loss"),
                "train_step": _state.get("train_step"),
            },
            "config": {"topn": req.topn, "num_beams": req.num_beams,
                       "resolve": req.resolve, "max_recursion": req.max_recursion,
                       "rounds_planned": req.max_recursion if req.resolve else 1},
            "known": {"submitted": len(req.known), "accepted": len(known),
                      "labels": sorted(known)[:KNOWN_IN_TRACE],
                      "dropped_from_trace": max(0, len(known) - KNOWN_IN_TRACE)},
            "modules": {"decode": MOD_DECODE, "rank": MOD_PREDICT,
                        "resolve": MOD_RESOLVE},
        })
        yield out("log", {"msg": f"conditioning on {len(known)} known labels"})

        total_hits = {}
        rounds = req.max_recursion if req.resolve else 1
        for rnd in range(1, rounds + 1):
            yield out("round", {"round": rnd, "known": len(known)})
            t_beam = time.monotonic()
            preds = await loop.run_in_executor(
                None,
                lambda: predict_labels(model, tok, device, sorted(known),
                                       topn=req.topn, num_beams=req.num_beams),
            )
            labels = [p for p, _ in preds]
            yield out("candidates", {
                "round": rnd,
                "count": len(preds),
                "decoder": MOD_DECODE,
                "num_beams": req.num_beams,
                "elapsed_ms": round((time.monotonic() - t_beam) * 1000, 1),
                "items": [{"label": l, "score": round(s, 3)} for l, s in preds],
            }, MOD_PREDICT)

            if not req.resolve:
                yield out("done", {"hits": [], "candidates": len(labels),
                                   "total": 0, "rounds": rnd})
                return

            yield out("log", {"msg": f"resolving {len(labels)} candidates…"})
            t_dns = time.monotonic()
            hits = await loop.run_in_executor(None, lambda: resolve_all(labels, req.domain))
            dns_ms = round((time.monotonic() - t_dns) * 1000, 1)
            new = {k: v for k, v in hits.items() if k not in total_hits}
            for label, ip in sorted(new.items()):
                yield out("hit", {"fqdn": f"{label}.{req.domain}", "ip": ip,
                                  "round": rnd}, MOD_RESOLVE)
            yield out("round_done", {
                "round": rnd, "new": len(new), "tested": len(labels),
                "rate": round(len(hits) / max(len(labels), 1), 4),
                "elapsed_ms": dns_ms,
            }, MOD_RESOLVE)
            total_hits.update(new)
            if not new:
                yield out("log", {"msg": "no new resolutions — stopping recursion"})
                break
            known |= set(new)

        yield out("done", {
            "hits": [{"fqdn": f"{k}.{req.domain}", "ip": v} for k, v in sorted(total_hits.items())],
            "total": len(total_hits),
            "rounds": rnd,
        })
    except Exception as exc:  # surface failures to the UI instead of hanging
        yield out("error", {"msg": f"{type(exc).__name__}: {exc}"})


@app.post("/api/predict")
async def predict(req: PredictRequest):
    return StreamingResponse(run_pipeline(req), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
