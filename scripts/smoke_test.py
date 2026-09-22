#!/usr/bin/env python3
"""Smoke test for a deployed MonopEli instance.

Usage:
    python scripts/smoke_test.py --url http://localhost
    python scripts/smoke_test.py --url http://your-server.com
"""

import argparse
import json
import sys
import urllib.request


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="MonopEli smoke test")
    parser.add_argument("--url", default="http://localhost", help="Base URL of the deployment")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    results: list[bool] = []

    print(f"\nSmoke testing: {base}\n")

    # 1. Health check
    try:
        resp = urllib.request.urlopen(f"{base}/health", timeout=10)
        data = json.loads(resp.read())
        results.append(
            check("Health check", data.get("status") == "healthy", f"v{data.get('version', '?')}")
        )
    except Exception as e:
        results.append(check("Health check", False, str(e)))

    # 2. API info
    try:
        resp = urllib.request.urlopen(f"{base}/api", timeout=10)
        data = json.loads(resp.read())
        results.append(check("API info", data.get("name") == "MonopEli API"))
    except Exception as e:
        results.append(check("API info", False, str(e)))

    # 3. API docs
    try:
        resp = urllib.request.urlopen(f"{base}/api/docs", timeout=10)
        results.append(check("API docs (Swagger)", resp.status == 200))
    except Exception as e:
        results.append(check("API docs (Swagger)", False, str(e)))

    # 4. Create session
    session_id = None
    try:
        req = urllib.request.Request(
            f"{base}/api/players/session",
            data=json.dumps({"display_name": "SmokeTest"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        session_id = data.get("session_id")
        results.append(check("Create session", session_id is not None, f"id={session_id}"))
    except Exception as e:
        results.append(check("Create session", False, str(e)))

    # 5. Create lobby
    lobby_id = None
    if session_id:
        try:
            req = urllib.request.Request(
                f"{base}/api/lobbies",
                data=json.dumps(
                    {
                        "host_name": "SmokeTest",
                        "lobby_name": "Smoke Test Lobby",
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            lobby_id = data.get("id")
            results.append(check("Create lobby", lobby_id is not None, f"id={lobby_id}"))
        except Exception as e:
            results.append(check("Create lobby", False, str(e)))

    # 6. Frontend
    try:
        resp = urllib.request.urlopen(f"{base}/", timeout=10)
        body = resp.read().decode()
        results.append(
            check("Frontend loads", "<html" in body.lower() or "<!doctype" in body.lower())
        )
    except Exception as e:
        results.append(check("Frontend loads", False, str(e)))

    # 7. Metrics
    try:
        resp = urllib.request.urlopen(f"{base}/metrics", timeout=10)
        body = resp.read().decode()
        results.append(check("Metrics endpoint", "http_requests_total" in body))
    except Exception as e:
        results.append(check("Metrics endpoint", False, str(e)))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("All smoke tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
