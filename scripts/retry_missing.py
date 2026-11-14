"""补传 download_gc10.py 因 429 限流而失败的文件。低并发 + 请求间隔。

用法：
    python scripts/retry_missing.py

背景：download_gc10.py 用 8 线程并发下载时会触发 hf-mirror 的
`HTTP Error 429: Too Many Requests`，曾经有 563 个文件失败（绝大多数是
labels/*.txt）。限流是按请求速率算的，所以本脚本改成 2 线程 + 每请求 0.35s
间隔，速率约为并发版的 1/16，可以稳定补完。

行为：先列出远端文件与本地缺失的差异，只下缺失的那部分，已存在且非空的跳过；
没有缺失就直接退出。

注意：与 download_gc10.py 一样关闭了 TLS 证书校验（check_hostname=False /
CERT_NONE），用于绕开部分 Windows 环境的证书链问题。对来源可信的公开数据集
可以接受，不建议把这种连接方式照搬到其他场景。
"""
import sys, os, json, ssl, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "gc10")
API = "https://hf-mirror.com/api/datasets/dronefreak/GC10-DET"
RAW = "https://hf-mirror.com/datasets/dronefreak/GC10-DET/resolve/main/"
WORKERS = 2
RETRY = 6
GAP = 0.35

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def fetch_one(rel):
    dst = os.path.join(OUT, rel.replace("data/", "").replace("/", os.sep))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last = None
    for i in range(RETRY):
        try:
            time.sleep(GAP)
            data = get(RAW + rel).read()
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return "ok"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            # 429 要退避得更狠
            time.sleep(2.0 + 3.0 * i)
    print(f"  FAIL {rel}: {last}", flush=True)
    return "fail"


def main():
    d = json.loads(get(API).read())
    sib = [s["rfilename"] for s in d["siblings"]]
    images = sorted(f for f in sib if f.startswith("data/images/"))
    labels = set(f for f in sib if f.startswith("data/labels/"))
    want = []
    for img in images:
        want.append(img)
        lbl = img.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if lbl in labels:
            want.append(lbl)
    missing = [r for r in want
               if not (os.path.exists(os.path.join(OUT, r.replace("data/", "").replace("/", os.sep)))
                       and os.path.getsize(os.path.join(OUT, r.replace("data/", "").replace("/", os.sep))) > 0)]
    print(f"远端 {len(want)} 个文件，本地缺 {len(missing)} 个", flush=True)
    if not missing:
        print("没有缺失文件")
        return

    t0 = time.time()
    cnt = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, j): j for j in missing}
        for i, fu in enumerate(as_completed(futs), 1):
            cnt[fu.result()] += 1
            if i % 50 == 0 or i == len(missing):
                el = time.time() - t0
                print(f"  {i}/{len(missing)}  {el/60:.1f} min  {cnt}", flush=True)
    print(f"补传完成 {cnt} 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
