#!/usr/bin/env python3
"""D3: prefix-aware router — LPR (locality-aware) request routing.

Sits in front of the E4 replica set and routes by a hash of the prompt's
leading tokens, so semantically-identical prefixes land on the same
backend -> prefix-cache hits instead of cold prefill elsewhere.

Modes:
  --policy lpr        : first-K-token hash affinity (default)
  --policy least_conn : pass-through to nginx (baseline comparator)

Design: two routing tiers.
  1. new-conversation requests  -> pick replica by first-chunk hash
  2. follow-up/same-prefix      -> same replica (hash is stable)
This is the classic SGLang-style "LPR" idea distilled: hash only the
PREFIX region (first N words), not the whole prompt.
"""
import argparse, asyncio, hashlib, json, time
import httpx
from fastapi import FastAPI, Request
import uvicorn

BACKENDS = []
app = FastAPI()
STATS = {"by_backend": {}, "prefix_hits": 0, "total": 0}


def prefix_key(body: dict, n_words: int) -> str:
    """Stable key from the first n_words of the prompt/message content."""
    text = body.get("prompt") or ""
    if not text:
        for m in body.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                text += c + " "
            elif isinstance(c, list):
                text += " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    words = text.split()[:n_words]
    return " ".join(words)


def choose_backend(body: dict, n_words: int) -> str:
    key = prefix_key(body, n_words)
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return BACKENDS[h % len(BACKENDS)]


@app.get("/stats")
async def stats():
    return STATS


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except Exception:
        payload = {}
    backend = choose_backend(payload, request.app.state.n_words)
    STATS["total"] += 1
    STATS["by_backend"][backend] = STATS["by_backend"].get(backend, 0) + 1

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=None) as c:
        if "stream" in payload and payload["stream"]:
            req = c.build_request(request.method, f"{backend}/{path}",
                                  content=body, headers=headers)
            r = await c.send(req, stream=True)
            from starlette.responses import StreamingResponse
            return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                                     headers=dict(r.headers))
        r = await c.request(request.method, f"{backend}/{path}",
                            content=body, headers=headers)
        from starlette.responses import Response
        return Response(r.content, status_code=r.status_code,
                        headers=dict(r.headers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--backends", required=True, help="comma list of base URLs")
    ap.add_argument("--prefix-words", type=int, default=256,
                    help="hash uses first N words of prompt")
    ap.add_argument("--policy", default="lpr")
    args = ap.parse_args()
    BACKENDS.extend(b.rstrip("/") for b in args.backends.split(","))
    app.state.n_words = args.prefix_words
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
