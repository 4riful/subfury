"""Budgeted DNS evidence collection for authorized SubFury runs."""

from __future__ import annotations

import concurrent.futures
import math
import re
import secrets
import threading
import time

try:
    import dns.exception
    import dns.resolver
except ImportError:  # DNS is optional for offline prediction and research.
    dns = None


RECORD_TYPES = ("A", "AAAA", "CNAME")
LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")


class QueryBudget:
    """Thread-safe global DNS query cap with a simple start-rate limit."""

    def __init__(self, limit: int, max_qps: float = 50.0):
        if limit < 1:
            raise ValueError("query budget must be at least 1")
        if not math.isfinite(max_qps) or max_qps < 0.1:
            raise ValueError("max_qps must be finite and at least 0.1")
        self.limit = limit
        self.max_qps = max_qps
        self.used = 0
        self._lock = threading.Lock()
        self._next_start = 0.0

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used

    def acquire(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + (1.0 / self.max_qps)
        delay = start - now
        if delay > 0:
            time.sleep(delay)
        return True


def normalize_domain(domain: str) -> str:
    """Return a normalized ASCII DNS name, rejecting malformed scope input."""
    value = str(domain).strip().lower().rstrip(".")
    if not value or len(value) > 253 or "." not in value:
        raise ValueError("domain must be an apex name such as example.com")
    labels = value.split(".")
    if any(not label or len(label) > 63 or label[0] == "-" or label[-1] == "-"
           or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label)
           for label in labels):
        raise ValueError("domain must be a valid lowercase ASCII DNS name")
    return value


def _zone_for_label(label: str, domain: str) -> str:
    parts = label.split(".")
    return domain if len(parts) == 1 else ".".join(parts[1:] + [domain])


def _query(resolver, fqdn: str, record_type: str, budget: QueryBudget) -> dict:
    item = {"fqdn": fqdn, "type": record_type, "outcome": "budget_exhausted",
            "answers": []}
    if not budget.acquire():
        return item
    try:
        answer = resolver.resolve(fqdn, record_type, raise_on_no_answer=False,
                                  search=False)
        if answer.rrset is None:
            item["outcome"] = "nodata"
            return item
        item["outcome"] = "answer"
        item["answers"] = sorted({str(value).rstrip(".") for value in answer})
        canonical = str(getattr(answer, "canonical_name", "")).rstrip(".")
        if canonical and canonical != fqdn.rstrip("."):
            item["canonical_name"] = canonical
        return item
    except dns.resolver.NXDOMAIN:
        item["outcome"] = "nxdomain"
    except dns.resolver.NoAnswer:
        item["outcome"] = "nodata"
    except dns.resolver.NoNameservers:
        item["outcome"] = "servfail"
    except (dns.exception.Timeout, dns.resolver.LifetimeTimeout):
        item["outcome"] = "timeout"
    except Exception as exc:
        item["outcome"] = "error"
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def _classify(questions: list[dict], wildcard_profile: dict) -> str:
    answered = {q["type"]: set(q["answers"])
                for q in questions if q["outcome"] == "answer"}
    if answered:
        matched = []
        uncertain = []
        for record_type, values in answered.items():
            profile = wildcard_profile.get(record_type)
            overlap = bool(profile) and bool(values & set(profile["answers"]))
            matched.append(overlap)
            uncertain.append(bool(profile) and profile["dynamic"] and not overlap)
        if all(matched):
            return "probable_wildcard"
        if any(uncertain) and all(match or unsure for match, unsure in zip(matched, uncertain)):
            return "inconclusive"
        return "resolved"
    outcomes = {q["outcome"] for q in questions}
    if "budget_exhausted" in outcomes:
        return "budget_exhausted"
    if outcomes <= {"nxdomain", "nodata"}:
        return "not_resolved"
    return "inconclusive"


def resolve_with_evidence(labels, domain: str, budget: QueryBudget, workers: int = 16,
                          record_types=RECORD_TYPES, wildcard_probes: int = 2,
                          resolver=None, probe_token_factory=None,
                          wildcard_cache=None, answer_cache=None) -> dict:
    """Resolve labels under one apex and return auditable per-question evidence.

    Wildcard controls are sent to each relevant parent zone. A record type is
    considered wildcard-active only when every random control for that zone
    answers, reducing the chance that one accidental name poisons recursion.
    """
    if dns is None and resolver is None:
        raise RuntimeError("dnspython not installed")
    domain = normalize_domain(domain)
    if not 1 <= wildcard_probes <= 10:
        raise ValueError("wildcard_probes must be between 1 and 10")
    labels = list(dict.fromkeys(str(label).strip().lower() for label in labels))
    invalid = [label for label in labels if not LABEL_RE.fullmatch(label)]
    if invalid:
        raise ValueError(f"invalid candidate label: {invalid[0]!r}")
    resolver = resolver or dns.resolver.Resolver()
    resolver.lifetime = 3.0
    token = probe_token_factory or (lambda: "sf-" + secrets.token_hex(8))

    zones = sorted({_zone_for_label(label, domain) for label in labels})
    controls = {}
    wildcard_profiles = {}
    wildcard_cache = wildcard_cache if wildcard_cache is not None else {}
    answer_cache = answer_cache if answer_cache is not None else {}
    for zone in zones:
        cached = wildcard_cache.get(zone)
        if cached is None:
            probes = []
            for _ in range(wildcard_probes):
                fqdn = f"{token()}.{zone}"
                probes.append([_query(resolver, fqdn, rt, budget) for rt in record_types])
            profile = {}
            for rt in record_types:
                answer_sets = [
                    {answer for q in probe
                     if q["type"] == rt and q["outcome"] == "answer"
                     for answer in q["answers"]}
                    for probe in probes
                ]
                if all(answer_sets):
                    profile[rt] = {
                        "answers": sorted(set().union(*answer_sets)),
                        "dynamic": len({frozenset(values) for values in answer_sets}) > 1,
                    }
            cached = {"profile": profile, "controls": probes}
            wildcard_cache[zone] = cached
        controls[zone] = cached["controls"]
        wildcard_profiles[zone] = cached["profile"]

    def one(label):
        fqdn = f"{label}.{domain}"
        questions = []
        for rt in record_types:
            key = (fqdn, rt)
            if key in answer_cache:
                question = dict(answer_cache[key], cached=True)
            else:
                question = _query(resolver, fqdn, rt, budget)
                if question["outcome"] in {"answer", "nxdomain", "nodata"}:
                    answer_cache[key] = question
            questions.append(question)
        zone = _zone_for_label(label, domain)
        status = _classify(questions, wildcard_profiles[zone])
        answers = sorted({answer for q in questions for answer in q["answers"]})
        return label, {"label": label, "fqdn": fqdn, "zone": zone,
                       "status": status, "answers": answers,
                       "records": {q["type"]: q["answers"] for q in questions
                                   if q["outcome"] == "answer"},
                       "wildcard_types": sorted(wildcard_profiles[zone]),
                       "questions": questions}

    evidence = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for label, item in pool.map(one, labels):
            evidence[label] = item

    counts = {}
    for item in evidence.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "domain": domain,
        "query_budget": budget.limit,
        "queries_used": budget.used,
        "queries_remaining": budget.remaining,
        "record_types": list(record_types),
        "wildcard_probes": wildcard_probes,
        "wildcards": {
            zone: {"active_types": sorted(wildcard_profiles[zone]),
                   "profile": wildcard_profiles[zone],
                   "controls": controls[zone]}
            for zone in zones
        },
        "counts": counts,
        "results": evidence,
    }


def preferred_answer(item: dict) -> tuple[str | None, str | None]:
    """Pick a display value without presenting a CNAME target as an IP."""
    records = item.get("records", {})
    for record_type in RECORD_TYPES:
        values = records.get(record_type, [])
        if values:
            return record_type, values[0]
    return None, None
