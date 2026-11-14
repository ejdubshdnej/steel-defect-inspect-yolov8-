"""从 hf-mirror 下载 GC10-DET 数据集。8 线程并发，支持断点续传（跳过已存在的文件）。

用法：
    python scripts/download_gc10.py

产物：data/gc10/images/{train,valid,test}/*.jpg 与 data/gc10/labels/...
（data/ 已在 .gitignore 中排除，本仓库不重新分发数据集）

为什么用线程池：
- 单线程下载约 0.30 MB/s，全量需要 50–60 min；
- 8 线程并发约 2.62 MB/s（约 9 倍提速），全量约 5 min。
- 再往上加线程会触发 hf-mirror 的速率限制（HTTP 429），不要盲目加大 WORKERS；
  若已触发限流，用 scripts/retry_missing.py 低并发补传。

注意：脚本关闭了 TLS 证书校验（check_hostname=False / CERT_NONE），
这是为了绕开部分 Windows 环境的证书链问题。对来源可信的公开数据集可以接受，
但不建议把这种连接方式照搬到其他场景。
"""
import sys, os, json, ssl, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "gc10")
API = "https://hf-mirror.com/api/datasets/dronefreak/GC10-DET"
RAW = "https://hf-mirror.com/datasets/dronefreak/GC10-DET/resolve/main/"
WORKERS = 8
RETRY = 3

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def fetch_one(rel):
    """下载单个文件，返回 'ok' / 'skip' / 'fail'。"""
    dst = os.path.join(OUT, rel.replace("data/", "").replace("/", os.sep))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last = None
    for _ in range(RETRY):
        try:
            data = get(RAW + rel).read()
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return "ok"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    print(f"  FAIL {rel}: {last}", flush=True)
    return "fail"


def main():
    d = json.loads(get(API).read())
    sib = [s["rfilename"] for s in d["siblings"]]
    images = sorted(f for f in sib if f.startswith("data/images/"))
    labels = set(f for f in sib if f.startswith("data/labels/"))
    jobs = []
    for img in images:
        jobs.append(img)
        lbl = img.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if lbl in labels:
            jobs.append(lbl)
    print(f"远端 {len(images)} 图 / {len(labels)} 标注 → {len(jobs)} 个文件", flush=True)

    t0 = time.time()
    cnt = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            cnt[fu.result()] += 1
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el/60:.1f} min  {cnt}", flush=True)
    el = time.time() - t0
    print(f"完成 {cnt} 耗时 {el/60:.1f} min ({el:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
