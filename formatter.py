from __future__ import annotations

from strategy import Signal


def _fmt_num(x: float) -> str:
    """Fixed-point string, never scientific notation (Python's default float
    formatting switches to "7e-05" style below ~0.0001, which a signal
    parser like Cornix can't read as a price) -- precision scaled to
    magnitude the same way strategy.py/gem_strategy.py round prices, then
    trailing zeros trimmed for a clean look."""
    x = float(x)
    ax = abs(x)
    if ax >= 1000:
        d = 2
    elif ax >= 1:
        d = 4
    elif ax >= 0.01:
        d = 6
    elif ax >= 0.0001:
        d = 8
    else:
        d = 10
    s = f"{x:.{d}f}".rstrip("0").rstrip(".")
    return s or "0"


def format_signal(sig: Signal, funding_pct: float | None = None) -> str:
    """Cornix-facing signal text. Kept minimal on purpose -- symbol,
    leverage, direction, entry ladder, take-profits, stop-loss -- since
    that's the exact shape a working Cornix custom-signal template expects.
    The bracketed engine tag is the only addition beyond that template: it
    lets a human tell ZENITH and GEM signals apart in the channel without
    being parsed as anything Cornix cares about (it comes after the
    #SYMBOL token, on its own visual slot)."""
    arrow = "🟢" if sig.side == "LONG" else "🔴"
    engine = (sig.extras or {}).get("engine") or "ZENITH"

    entries = list(sig.entries)
    while len(entries) < 4:
        entries.append(entries[-1] if entries else sig.reference)
    entries = entries[:4]

    tps = list(sig.tps)
    while len(tps) < 5:
        tps.append(tps[-1] if tps else sig.sl)
    tps = tps[:5]

    lines = [
        f"#{sig.symbol} [{engine}]",
        "",
        f"Leverage: {sig.leverage}X",
        f"Direction: {arrow} {sig.side}",
        "",
        "",
        f"🎯 Entry targets:{'-'.join(_fmt_num(e) for e in entries)}",
        "💰 Take-Profit Targets:",
        f"TP 1: {_fmt_num(tps[0])}",
        f"TP 2: {_fmt_num(tps[1])}",
        f"TP 3: {_fmt_num(tps[2])}",
        f"TP 4: {_fmt_num(tps[3])}",
        f"TP 5: {_fmt_num(tps[4])}",
        f"🛑 Stop Loss: {_fmt_num(sig.sl)}",
    ]
    return "\n".join(lines)
