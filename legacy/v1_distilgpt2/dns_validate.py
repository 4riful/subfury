"""DNS validation for predicted subdomains.

Concurrent resolution via dnspython with a threaded fallback to the
simple socket.gethostbyname approach shown in the README.

Only run this against domains you are authorized to test.
"""

import argparse
import concurrent.futures
import socket

try:
    import dns.resolver
    HAVE_DNSPYTHON = True
except ImportError:
    HAVE_DNSPYTHON = False


def dns_resolve(subdomain):
    """README's simple resolver: (fqdn, ip | None)."""
    try:
        ip = socket.gethostbyname(subdomain)
        return subdomain, ip
    except socket.gaierror:
        return subdomain, None


def resolve_dnspython(fqdn, timeout=3.0):
    resolver = dns.resolver.Resolver()
    resolver.lifetime = timeout
    try:
        answers = resolver.resolve(fqdn, "A")
        return fqdn, str(answers[0])
    except Exception:
        return fqdn, None


def validate(labels, domain, workers=20):
    """Resolve label.domain for each label. Returns {fqdn: ip|None}."""
    fqdns = [f"{label}.{domain}" for label in labels]
    fn = resolve_dnspython if HAVE_DNSPYTHON else dns_resolve
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for fqdn, ip in pool.map(fn, fqdns):
            results[fqdn] = ip
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", help="target domain (must be authorized)")
    ap.add_argument("--wordlist", required=True, help="file with one label per line")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    with open(args.wordlist) as f:
        labels = [ln.strip() for ln in f if ln.strip()]

    results = validate(labels, args.domain, workers=args.workers)
    hits = {k: v for k, v in results.items() if v}
    for fqdn, ip in sorted(hits.items()):
        print(f"[+] {fqdn} -> {ip}")
    print(f"\n{len(hits)}/{len(results)} resolved")


if __name__ == "__main__":
    main()
