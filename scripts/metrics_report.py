# Reads the JSON-lines request log written and reports success/failure counts
import json
import sys
from pathlib import Path

ALERT_WINDOW = 50
ALERT_ERROR_RATE = 0.20


def load_lines(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"skipping unparseable line {line_no}", file=sys.stderr)
    return rows


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "requests.log")
    if not path.exists():
        print(f"no log file at {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_lines(path)
    total = len(rows)
    failures = [r for r in rows if r.get("status", 0) >= 400]
    successes = total - len(failures)
    durations = [r["duration_ms"] for r in rows if "duration_ms" in r]

    print(f"requests:       {total}")
    print(f"successes:      {successes}")
    print(f"failures:       {len(failures)}")
    print(f"error rate:     {(len(failures) / total * 100) if total else 0:.1f}%")
    print(f"p95 latency:    {p95(durations):.1f} ms")

    # alert rule: error rate over the last ALERT_WINDOW requests exceeds
    # ALERT_ERROR_RATE
    last_window = rows[-ALERT_WINDOW:]
    if last_window:
        window_failures = sum(1 for r in last_window if r.get("status", 0) >= 400)
        window_rate = window_failures / len(last_window)
        status = "ALERT" if window_rate > ALERT_ERROR_RATE else "ok"
        print(
            f"last {len(last_window)} requests: "
            f"{window_failures} failed ({window_rate * 100:.1f}%) -> {status}"
        )

    if failures:
        print("\nfailed requests:")
        for r in failures:
            print(
                f"  {r.get('ts')}  {r.get('method')} {r.get('route')} "
                f"-> {r.get('status')}  request_id={r.get('request_id')}"
            )


if __name__ == "__main__":
    main()
