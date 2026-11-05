#!/usr/bin/env python3
"""E20: SLO-aware mini control plane (stdlib-only HTTP/1.1 reverse proxy).

Combines the repo's measured findings into one gateway:

  1. prefix-affinity routing      (D3: hash of leading prompt words)
  2. short/long class split       (E19/SRF: prompt_len <= threshold -> short)
  3. admission control            (E9: cap inflight per class; over-cap ->
     instant 429 + Retry-After instead of unbounded queueing)
  4. per-class observability      (/stats on port+1)

A "429" is returned within ~1ms so callers can retry against another
endpoint or back off — the SLO-relevant contrast vs a naive proxy that
queues every request until the engine chokes (E8's admission wall).

Usage: python3 e20_control_plane.py --port 8092 \
         --backends 127.0.0.1:8010,127.0.0.1:8001 \
         --short-threshold 256 --long-cap 24 --short-cap 32
"""
import argparse, asyncio, hashlib, json, time

BACKENDS = []
N_WORDS = 256
SHORT_WORDS = 192        # ~256 tokens at ~1.3 tok/word
LONG_CAP = 1 << 30
SHORT_CAP = 1 << 30
STATS = {
    "total": 0, "rejected": 0,
    "by_class": {"short": 0, "long": 0},
    "rej_by_class": {"short": 0, "long": 0},
    "by_backend": {},
    "inflight": {"short": 0, "long": 0},
    "lat_ms": {"short": [], "long": []},
}


def parse_body(body: bytes):
    try:
        d = json.loads(body)
    except Exception:
        return "", 0
    text = d.get("prompt") or ""
    if not text:
        for m in d.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                text += c + " "
            elif isinstance(c, list):
                text += " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return text, len(text.split())


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter):
    try:
        while True:
            chunk = await r.read(1 << 16)
            if not chunk:
                break
            w.write(chunk)
            await w.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


def reject(writer):
    body = json.dumps({"error": {"type": "overloaded",
                                 "message": "control-plane admission cap",
                                 "retry_after_ms": 500}}).encode()
    writer.write(b"HTTP/1.1 429 Too Many Requests\r\n"
                 b"content-type: application/json\r\n"
                 b"retry-after: 1\r\n"
                 + f"content-length:{len(body)}\r\n\r\n".encode() + body)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except Exception:
        writer.close(); return
    headers = {}
    lines = head.decode(errors="replace").split("\r\n")
    req_line = lines[0]
    for l in lines[1:]:
        if ":" in l:
            k, v = l.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    cl = int(headers.get("content-length", 0) or 0)
    body = await reader.readexactly(cl) if cl else b""

    text, n_words = parse_body(body)
    cls = "short" if n_words <= SHORT_WORDS else "long"
    cap = SHORT_CAP if cls == "short" else LONG_CAP
    STATS["total"] += 1

    if STATS["inflight"][cls] >= cap:
        STATS["rejected"] += 1
        STATS["rej_by_class"][cls] += 1
        reject(writer)
        await writer.drain(); writer.close(); return

    key = " ".join(text.split()[:N_WORDS])
    idx = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % len(BACKENDS)
    host, port = BACKENDS[idx].split(":")
    STATS["by_class"][cls] += 1
    bkey = f"{host}:{port}"
    STATS["by_backend"][bkey] = STATS["by_backend"].get(bkey, 0) + 1
    STATS["inflight"][cls] += 1
    t0 = time.perf_counter()

    try:
        br, bw = await asyncio.open_connection(host, int(port))
    except Exception:
        STATS["inflight"][cls] -= 1
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length:0\r\n\r\n")
        await writer.drain(); writer.close(); return

    out = [req_line]
    for l in lines[1:]:
        if l and l.split(":")[0].strip().lower() not in (
                "host", "connection", "keep-alive", "proxy-connection",
                "transfer-encoding", "te", "upgrade"):
            out.append(l)
    out.append(f"Host: {host}:{port}")
    out.append("Connection: close")
    bw.write(("\r\n".join(out) + "\r\n\r\n").encode() + body)
    await bw.drain()
    try:
        await pipe(br, writer)
    finally:
        STATS["inflight"][cls] -= 1
        STATS["lat_ms"][cls].append(round((time.perf_counter() - t0) * 1000))


async def _stats(reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    snap = {k: v for k, v in STATS.items() if k != "lat_ms"}
    snap["lat_summary_ms"] = {}
    for cls, xs in STATS["lat_ms"].items():
        if xs:
            s = sorted(xs)
            snap["lat_summary_ms"][cls] = {
                "n": len(s), "p50": s[len(s)//2],
                "p90": s[int(0.9*(len(s)-1))], "max": s[-1]}
    body = json.dumps(snap).encode()
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                 + f"content-length:{len(body)}\r\n\r\n".encode() + body)
    await writer.drain(); writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--backends", required=True)
    ap.add_argument("--prefix-words", type=int, default=256)
    ap.add_argument("--short-threshold", type=int, default=192,
                    help="max prompt WORDS counted as short (~1.3 tok/word)")
    ap.add_argument("--long-cap", type=int, default=1 << 30)
    ap.add_argument("--short-cap", type=int, default=1 << 30)
    args = ap.parse_args()
    global N_WORDS, SHORT_WORDS, LONG_CAP, SHORT_CAP
    N_WORDS = args.prefix_words
    SHORT_WORDS = args.short_threshold
    LONG_CAP = args.long_cap
    SHORT_CAP = args.short_cap
    BACKENDS.extend(args.backends.split(","))

    async def run():
        srv = await asyncio.start_server(handle, "0.0.0.0", args.port)
        st = await asyncio.start_server(
            lambda r, w: _stats(r, w), "0.0.0.0", args.port + 1)
        print(f"control-plane :{args.port} (stats :{args.port+1}) -> "
              f"{BACKENDS} short<={SHORT_WORDS}w caps l/s={LONG_CAP}/{SHORT_CAP}",
              flush=True)
        async with srv, st:
            await srv.serve_forever()
    asyncio.run(run())


if __name__ == "__main__":
    main()
