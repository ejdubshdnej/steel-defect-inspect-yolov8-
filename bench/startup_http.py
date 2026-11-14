"""实测「进程启动 → HTTP 端口可用」耗时。

用法：
    python bench/startup_http.py

测的是从 serve/main.py 的子进程被拉起，到 /health 返回 200 之间的墙钟时间——
这是用户/产线实际感知的启动延迟，比进程内的 import 时间更接近真实体验。

做法：起一个子进程跑 serve/main.py，stdout/stderr 都重定向到 DEVNULL，
然后每 10 ms 轮询一次 /health，记下第一次成功的时间。

注意事项：
- 跑之前必须先停掉已经占着 8000 端口的服务，否则测到的是旧进程。
- 不要用 capture_output=True：管道缓冲区会让子进程的写操作阻塞，测出来的
  启动耗时偏大且不稳定。这里用 DEVNULL 就是为了不干扰被测进程。
- 单次测量噪声较大，需要可信数字时请多跑几次取中位数。
"""
import sys, os, time, socket, subprocess, urllib.request, urllib.error
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
URL = "http://127.0.0.1:8000/health"


def port_free(host="127.0.0.1", port=8000):
    s = socket.socket()
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


if not port_free():
    print("端口 8000 被占用，先停掉服务再跑本脚本")
    raise SystemExit(1)

t0 = time.perf_counter()
p = subprocess.Popen([PY, "-X", "utf8", os.path.join(BASE, "serve", "main.py")],
                     cwd=BASE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

deadline = t0 + 60
ready = None
try:
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=0.2) as r:
                if r.status == 200:
                    ready = time.perf_counter() - t0
                    break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.01)
finally:
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()

if ready is None:
    print("60 秒内没起来，检查 serve/main.py")
    raise SystemExit(1)

print(f"进程启动 -> /health 可用   {ready*1e3:8.2f} ms")
