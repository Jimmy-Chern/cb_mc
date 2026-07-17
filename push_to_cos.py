"""
腾讯云COS存储模块 — 替代GitHub Gist
====================================
上传: 服务器用SecretId/SecretKey写入
下载: QMT客户端用公开URL直接GET (bucket需设public-read)
"""
import json
from qcloud_cos import CosConfig, CosS3Client

# ===== 配置（需要填写）=====
SECRET_ID  = "$COS_SECRET_ID"
SECRET_KEY = "$COS_SECRET_KEY"
BUCKET     = "cb-signal-1454417314"
REGION     = "ap-shanghai"

# 公开读取地址（私有桶需带签名, QMT客户端用SDK读取）
PUBLIC_BASE = f"{BUCKET}.cos.{REGION}.myqcloud.com"

_config = None
_client = None

def get_client():
    global _client
    if _client is None:
        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
        _client = CosS3Client(config)
    return _client

def upload_json(filename: str, data: dict) -> str:
    """上传JSON到COS, 返回公开URL"""
    client = get_client()
    body = json.dumps(data, ensure_ascii=False).encode('utf-8')
    client.put_object(
        Bucket=BUCKET,
        Key=filename,
        Body=body,
        ContentType='application/json',
    )
    url = f"{PUBLIC_BASE}/{filename}"
    print(f"  ✅ COS: {url}")
    return url

def upload_signal(signal: dict):
    """上传加密信号"""
    return upload_json('cb_signal.json', signal)

def upload_log(log: dict):
    """上传明文日志"""
    return upload_json('cb_log.json', log)
