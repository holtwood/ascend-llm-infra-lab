#!/usr/bin/env python3
"""D3: prefix-aware router (stdlib-only asyncio HTTP/1.1 reverse proxy).

Hashes the first N words of the JSON `prompt`/`messages` body and picks a
backend, so same-prefix traffic always lands on the same replica -> its
prefix cache gets the hit instead of another replica's cold prefill.

Usage: python3 d3_router.py --port 8090 \
         --backends 127.0.0.1:8010,127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003
"""
import argparse, asyncio, hashlib, json

BACKENDS = []
N_WORDS = 256
STATS = {"total": 0, "by_backend": {}}


def prefix_key(body: bytes) -> str:
    try:
        d = json.loads(body)
    except Exception:
        return ""
    text = d.get("prompt") or ""
    if not text:
        for m in d.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                text += c + " "
            elif isinstance(c, list):
                text += " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return " ".join(text.split()[:N_WORDS])


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

    key = prefix_key(body)
    idx = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % len(BACKENDS)
    host, port = BACKENDS[idx].split(":")
    STATS["total"] += 1
    bkey = f"{host}:{port}"
    STATS["by_backend"][bkey] = STATS["by_backend"].get(bkey, 0) + 1

    try:
        br, bw = await asyncio.open_connection(host, int(port))
    except Exception:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length:0\r\n\r\n")
        await writer.drain(); writer.close(); return

    # strip hop-by-hop headers, fix Host
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
    # single direction is enough (backend will close after response)
    await pipe(br, writer)


async def stats_server(port):
    app = await asyncio.start_server(
        lambda r, w: _stats(r, w), "0.0.0.0", port + 1)


async def _stats(reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    body = json.dumps(STATS).encode()
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                 + f"content-length:{len(body)}\r\n\r\n".encode() + body)
    await writer.drain(); writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--backends", required=True)
    ap.add_argument("--prefix-words", type=int, default=256)
    args = ap.parse_args()
    global N_WORDS
    N_WORDS = args.prefix_words
    BACKENDS.extend(args.backends.split(","))

    async def run():
        await stats_server(args.port)
        srv = await asyncio.start_server(handle, "0.0.0.0", args.port)
        print(f"router :{args.port} (stats :{args.port+1}) -> {BACKENDS}",
              flush=True)
        async with srv:
            await srv.serve_forever()
    asyncio.run(run())


if __name__ == "__main__":
    main()
