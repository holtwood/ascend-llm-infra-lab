#!/usr/bin/env python3
"""E37 — does client disconnect free KV blocks immediately?

Interview question: "A client disconnects mid-reply. What must happen?"
Expected: KV blocks freed immediately, not at end-of-generation.

Method: read /metrics for vllm:num_requests_running + gpu_cache_usage_perc,
start K streaming decodes (long outputs), wait for steady state, abort all
connections, then sample metrics every 0.2s to find when requests disappear
and KV is released. Compare against the same streams left to finish.
"""
import argparse, json, re, time, urllib.request
import threading, http.client


def metrics(base):
    out = {}
    try:
        txt = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                     "vllm:kv_cache_usage_perc"):
            m = re.findall(rf"^{name}.*?\s([\d.eE+-]+)$", txt, re.M)
            out[name.split(":")[1]] = float(m[0]) if m else None
    except Exception as e:
        out["error"] = str(e)
    return out


def stream_request(base, model, conns):
    conn = http.client.HTTPConnection(base.replace("http://", "").split("/")[0])
    body = json.dumps({"model": model, "prompt": "Write a very long story " * 3,
                       "max_tokens": 2000, "temperature": 0.9, "stream": True})
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read(1)  # ensure started
    conns.append(conn)
    try:
        while resp.read(64):
            pass
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    timeline = []
    t0 = time.monotonic()

    def snap(tag=""):
        m = metrics(args.base_url)
        m["t"] = round(time.monotonic() - t0, 2)
        if tag:
            m["tag"] = tag
        timeline.append(m)
        return m

    snap("idle")
    conns = []
    ths = [threading.Thread(target=stream_request, args=(args.base_url, args.model, conns), daemon=True)
           for _ in range(args.clients)]
    for t in ths:
        t.start()
    deadline = time.monotonic() + args.settle
    while len(conns) < args.clients and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)  # ensure all are generating
    snap("running")

    # abort all client connections: close sockets -> server sees disconnect
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
    for t in ths:
        t.join(timeout=3)

    # poll until requests drop to 0 or 15s elapse
    deadline = time.monotonic() + 15
    freed_at = None
    while time.monotonic() < deadline:
        m = snap()
        if m.get("num_requests_running", 1) == 0 and freed_at is None:
            freed_at = m["t"]
            snap("freed")
            break
        time.sleep(0.2)
    time.sleep(0.5)
    snap("final")

    res = {"clients": args.clients, "freed_at_s": freed_at, "timeline": timeline}
    json.dump(res, open(args.out, "w"))
    print(json.dumps({"clients": args.clients, "freed_at_s": freed_at}))
    for m in timeline[:20]:
        print(m)


if __name__ == "__main__":
    main()
