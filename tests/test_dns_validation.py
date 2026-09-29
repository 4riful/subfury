import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "subfury"))

import dns_validation
from dns_validation import (QueryBudget, normalize_domain, preferred_answer,
                            resolve_with_evidence)


class NXDOMAIN(Exception):
    pass


class NoAnswer(Exception):
    pass


class NoNameservers(Exception):
    pass


class Timeout(Exception):
    pass


FAKE_DNS = SimpleNamespace(
    resolver=SimpleNamespace(NXDOMAIN=NXDOMAIN, NoAnswer=NoAnswer,
                             NoNameservers=NoNameservers, LifetimeTimeout=Timeout),
    exception=SimpleNamespace(Timeout=Timeout),
)


class FakeAnswer:
    def __init__(self, fqdn, values):
        self.rrset = values or None
        self.canonical_name = fqdn
        self._values = values

    def __iter__(self):
        return iter(self._values)


class FakeResolver:
    lifetime = 0

    def resolve(self, fqdn, record_type, **_kwargs):
        if record_type != "A":
            raise NoAnswer
        if ((fqdn.startswith("probe") and fqdn.count(".") == 2)
                or fqdn == "www.example.com"):
            return FakeAnswer(fqdn, ["192.0.2.10"])
        if fqdn == "api.dev.example.com":
            return FakeAnswer(fqdn, ["192.0.2.20"])
        if fqdn == "real.example.com":
            return FakeAnswer(fqdn, ["192.0.2.20"])
        raise NXDOMAIN


class DynamicWildcardResolver(FakeResolver):
    def resolve(self, fqdn, record_type, **kwargs):
        if record_type == "A" and fqdn == "probe1.example.com":
            return FakeAnswer(fqdn, ["192.0.2.1"])
        if record_type == "A" and fqdn == "probe2.example.com":
            return FakeAnswer(fqdn, ["192.0.2.2"])
        if record_type == "A" and fqdn == "real.example.com":
            return FakeAnswer(fqdn, ["192.0.2.3"])
        return super().resolve(fqdn, record_type, **kwargs)


class TransientResolver(FakeResolver):
    def __init__(self):
        self.failed = False

    def resolve(self, fqdn, record_type, **kwargs):
        if fqdn == "flaky.example.com" and record_type == "A" and not self.failed:
            self.failed = True
            raise Timeout
        return super().resolve(fqdn, record_type, **kwargs)


class DnsValidationTests(unittest.TestCase):
    def setUp(self):
        self.real_dns = dns_validation.dns
        dns_validation.dns = FAKE_DNS

    def tearDown(self):
        dns_validation.dns = self.real_dns

    def test_normalize_domain_rejects_out_of_scope_shapes(self):
        self.assertEqual(normalize_domain("Example.COM."), "example.com")
        for value in ("example", "-example.com", "example..com", "https://example.com"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_domain(value)

    def test_wildcards_are_excluded_and_nested_zone_is_checked(self):
        probes = iter(("probe1", "probe2", "probe3", "probe4"))
        budget = QueryBudget(18, max_qps=1_000_000)
        report = resolve_with_evidence(
            ["www", "api.dev"], "example.com", budget,
            resolver=FakeResolver(), probe_token_factory=lambda: next(probes),
        )

        self.assertEqual(report["queries_used"], 18)
        self.assertEqual(report["results"]["www"]["status"], "probable_wildcard")
        self.assertEqual(report["results"]["api.dev"]["status"], "resolved")
        self.assertEqual(report["wildcards"]["example.com"]["active_types"], ["A"])
        self.assertEqual(report["wildcards"]["dev.example.com"]["active_types"], [])

    def test_explicit_answer_different_from_stable_wildcard_is_kept(self):
        probes = iter(("probe1", "probe2"))
        report = resolve_with_evidence(
            ["real"], "example.com", QueryBudget(9, max_qps=1_000_000),
            resolver=FakeResolver(), probe_token_factory=lambda: next(probes),
        )

        self.assertEqual(report["results"]["real"]["status"], "resolved")

    def test_unmatched_answer_in_dynamic_wildcard_zone_is_inconclusive(self):
        probes = iter(("probe1", "probe2"))
        report = resolve_with_evidence(
            ["real"], "example.com", QueryBudget(9, max_qps=1_000_000),
            resolver=DynamicWildcardResolver(), probe_token_factory=lambda: next(probes),
        )

        self.assertEqual(report["results"]["real"]["status"], "inconclusive")

    def test_query_budget_is_never_exceeded(self):
        probes = iter(("probe1", "probe2"))
        budget = QueryBudget(2, max_qps=1_000_000)
        report = resolve_with_evidence(
            ["api"], "example.com", budget, resolver=FakeResolver(),
            probe_token_factory=lambda: next(probes),
        )

        self.assertEqual(report["queries_used"], 2)
        self.assertEqual(report["queries_remaining"], 0)
        self.assertEqual(report["results"]["api"]["status"], "budget_exhausted")

    def test_wildcard_controls_are_reused_across_rounds(self):
        probes = iter(("probe1", "probe2"))
        budget = QueryBudget(12, max_qps=1_000_000)
        cache = {}
        first = resolve_with_evidence(
            ["www"], "example.com", budget, resolver=FakeResolver(),
            probe_token_factory=lambda: next(probes), wildcard_cache=cache,
        )
        second = resolve_with_evidence(
            ["other"], "example.com", budget, resolver=FakeResolver(),
            probe_token_factory=lambda: next(probes), wildcard_cache=cache,
        )

        self.assertEqual(first["queries_used"], 9)
        self.assertEqual(second["queries_used"], 12)
        self.assertEqual(second["results"]["other"]["status"], "not_resolved")

    def test_candidate_answers_are_reused_across_rounds(self):
        probes = iter(("probe1", "probe2"))
        budget = QueryBudget(12, max_qps=1_000_000)
        wildcard_cache = {}
        answer_cache = {}
        first = resolve_with_evidence(
            ["other"], "example.com", budget, resolver=FakeResolver(),
            probe_token_factory=lambda: next(probes), wildcard_cache=wildcard_cache,
            answer_cache=answer_cache,
        )
        second = resolve_with_evidence(
            ["other"], "example.com", budget, resolver=FakeResolver(),
            probe_token_factory=lambda: next(probes), wildcard_cache=wildcard_cache,
            answer_cache=answer_cache,
        )

        self.assertEqual(first["queries_used"], 9)
        self.assertEqual(second["queries_used"], 9)
        self.assertTrue(all(q["cached"] for q in second["results"]["other"]["questions"]))

    def test_transient_failures_are_retried(self):
        budget = QueryBudget(6, max_qps=1_000_000)
        wildcard_cache = {"example.com": {"profile": {}, "controls": []}}
        answer_cache = {}
        resolver = TransientResolver()
        first = resolve_with_evidence(
            ["flaky"], "example.com", budget, resolver=resolver,
            wildcard_cache=wildcard_cache, answer_cache=answer_cache,
        )
        second = resolve_with_evidence(
            ["flaky"], "example.com", budget, resolver=resolver,
            wildcard_cache=wildcard_cache, answer_cache=answer_cache,
        )

        self.assertEqual(first["results"]["flaky"]["status"], "inconclusive")
        self.assertEqual(second["results"]["flaky"]["status"], "not_resolved")
        self.assertEqual(second["queries_used"], 4)

    def test_invalid_candidate_never_reaches_resolver(self):
        with self.assertRaises(ValueError):
            resolve_with_evidence(["../outside"], "example.com", QueryBudget(10),
                                  resolver=FakeResolver())

    def test_non_finite_qps_is_rejected(self):
        with self.assertRaises(ValueError):
            QueryBudget(10, max_qps=float("nan"))
        with self.assertRaises(ValueError):
            QueryBudget(10, max_qps=5e-324)

    def test_preferred_answer_keeps_record_type(self):
        self.assertEqual(preferred_answer({"records": {"CNAME": ["target.example"]}}),
                         ("CNAME", "target.example"))


if __name__ == "__main__":
    unittest.main()
