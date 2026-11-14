"""不依赖 requests 的 HTTP 接口冒烟测试：只用标准库 urllib 拼 multipart 请求。

用法：
    python serve/main.py &            # 先起服务（或另开一个终端）
    python scripts/test_api.py

流程：GET /health → POST /predict 上传 docs/sample.jpg → 打印 JSON 响应。

注意事项：
- 服务地址写死 127.0.0.1:8000；若用 DSI_PORT 换了端口，请同步改本脚本里的 URL。
- /predict 的表单字段名必须是 file（见 post_file 的 field 参数），写成 image
  会返回 422。
- docs/sample.jpg 不存在时会在最后一步抛 FileNotFoundError，先跑
  hmi/_shot.py 或放一张图进去。
"""
import sys, os, io, json, urllib.request, uuid
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post_file(url, path, field="file"):
    """手工拼 multipart/form-data 并 POST。field 必须与后端参数名一致。"""
    boundary = "----" + uuid.uuid4().hex
    fn = os.path.basename(path).encode("utf-8")
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{fn.decode()}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, body, {
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get_json(url):
    return json.loads(urllib.request.urlopen(url, timeout=10).read())


if __name__ == "__main__":
    print("HEALTH", json.dumps(get_json("http://127.0.0.1:8000/health"), ensure_ascii=False))
    img = os.path.join(BASE, "docs", "sample.jpg")
    print("SAMPLE", img, os.path.getsize(img), "B")
    res = post_file("http://127.0.0.1:8000/predict", img)
    print(json.dumps(res, ensure_ascii=False, indent=2))
