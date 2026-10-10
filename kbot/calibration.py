import json
import math
import sqlite3
from typing import Optional


def brier(rows: list) -> Optional[float]:
    if not rows:
        return None
    return sum((p - o) ** 2 for p, o in rows) / len(rows)


def platt_coeffs(rows: list) -> Optional[tuple]:
    """Fit sigmoid a,b for calibrated p = 1/(1+exp(a*logit(p)+b)) via simple Newton steps."""
    rows = [(min(max(p, 1e-4), 1 - 1e-4), o) for p, o in rows]
    if len(rows) < 20:
        return None
    xs = [math.log(p / (1 - p)) for p, _ in rows]
    ys = [o for _, o in rows]
    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(500):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            z = a * x + b
            pr = 1.0 / (1.0 + math.exp(z))
            ga += (pr - y) * x
            gb += (pr - y)
        ga /= len(xs)
        gb /= len(xs)
        a -= lr * ga
        b -= lr * gb
    return a, b


def platt_apply(p: float, a: float, b: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    x = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(a * x + b))


def bucket_report(db: sqlite3.Connection, source: str = "ensemble") -> str:
    rows = db.execute(
        "SELECT p, outcome FROM calib WHERE source=?", (source,)
    ).fetchall()
    if not rows:
        return "no calibration data"
    buckets = {}
    for p, o in rows:
        k = int(p * 10) / 10
        buckets.setdefault(k, []).append((p, o))
    lines = [f"calibration for {source}: n={len(rows)} brier={brier(rows):.4f}"]
    for k in sorted(buckets):
        rs = buckets[k]
        avg_p = sum(p for p, _ in rs) / len(rs)
        avg_o = sum(o for _, o in rs) / len(rs)
        lines.append(f"  p~{k:.1f}: n={len(rs)} predicted={avg_p:.3f} actual={avg_o:.3f}")
    return "\n".join(lines)
