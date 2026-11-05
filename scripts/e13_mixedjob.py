import asyncio, random, string, time, httpx

WORDS = ("apple banana cedar delta ember fjord granite harbor ivory juniper "
         "kettle lumen mosaic nectar onyx prism quartz ridge saffron timber "
         "umbra violet walnut xenon yonder zephyr atlas beryl cobalt dune").split()

def mk(n, rng):
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=10)) + " " + " ".join(rng.choices(WORDS, k=n))

async def one(c, p, mt, job):
    t0 = time.time()
    try:
        r = await c.post("http://localhost:8000/v1/completions",
                         json={"model": "qwen35-4b",
                               "prompt": f"#job_name[{job}]# " + p,
                               "max_tokens": mt, "temperature": 0})
        await r.aread()
        return {"job": job, "e2e": time.time() - t0, "ok": r.status_code == 200}
    except Exception as e:
        return {"job": job, "e2e": time.time() - t0, "ok": False, "err": str(e)[:80]}

async def main():
    rng = random.Random(1)
    sem = asyncio.Semaphore(96)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=128), timeout=1200) as c:
        async def t(i):
            async with sem:
                if i < 48:
                    return await one(c, mk(2800, rng), 256, "long")
                return await one(c, mk(120, rng), 256, "short")
        res = await asyncio.gather(*(t(i) for i in range(96)))
    for job in ("long", "short"):
        xs = sorted(r["e2e"] for r in res if r["job"] == job and r["ok"])
        if xs:
            print(job, "n=", len(xs), "mean=", round(sum(xs)/len(xs), 2),
                  "p50=", round(xs[len(xs)//2], 2), "p90=", round(xs[int(0.9*(len(xs)-1))], 2))
    print("fails:", sum(1 for r in res if not r["ok"]))

asyncio.run(main())
