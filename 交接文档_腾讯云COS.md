# 可转债MC定价自动交易 — 腾讯云COS版 交接文档

## 一、系统概述

A800服务器每天15:30自动MC定价(5000路径, 324只转债) → 选出Top10 → AES加密 → 上传腾讯云COS。

你的腾讯云Windows轻量服务器上跑QMT，每天14:50从COS拉取信号，解密后自动调仓。

**不需要外网，全走腾讯云内网。**

## 二、COS存储信息

| 项目 | 值 |
|------|-----|
| 桶名 | `cb-signal-1454417314` |
| 地域 | `ap-shanghai` (上海) |
| 权限 | 私有读写 |
| SecretId | `<COS_SECRET_ID>` |
| SecretKey | `<COS_SECRET_KEY>` |
| 内网域名 | `cb-signal-1454417314.cos.ap-shanghai.myqcloud.com` |

桶内文件：
- `cb_signal.json` — AES加密信号（QMT调仓用）
- `cb_log.json` — 明文运行日志（监控用）

## 三、QMT客户端部署

### 3.1 安装依赖

在QMT的Python终端或cmd执行：
```
pip install cos-python-sdk-v5 pyaes
```

### 3.2 QMT策略代码

**全部复制到QMT策略编辑器，不改任何代码直接运行：**

```python
# ===== 可转债MC定价策略 — 腾讯云COS版 =====
import json, hashlib, base64
from qcloud_cos import CosConfig, CosS3Client
import pyaes

# COS配置
SECRET_ID  = "<COS_SECRET_ID>"
SECRET_KEY = "<COS_SECRET_KEY>"
BUCKET     = "cb-signal-1454417314"
REGION     = "ap-shanghai"
PASSWORD   = "336018"

_cos_client = None

def get_cos():
    global _cos_client
    if _cos_client is None:
        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
        _cos_client = CosS3Client(config)
    return _cos_client

def get_signal():
    """从COS拉取AES密文 → 解密 → 返回Top10代码列表"""
    try:
        client = get_cos()
        resp = client.get_object(Bucket=BUCKET, Key='cb_signal.json')
        data = json.loads(resp['Body'].read().decode('utf-8'))
        signal = decrypt_aes(data["encrypted"], PASSWORD)
        print(f"[{signal['date']} {signal['time']}] Top10: {signal['buy']}")
        for p in signal['positions']:
            print(f"  {p['code']} {p['name']:<10s} MC={p['model_price']:.0f} MKT={p['market_price']:.0f} disc={p['discount']:+.1f}%")
        return signal['buy']
    except Exception as e:
        print(f"信号获取失败: {e}")
        return []

def decrypt_aes(encrypted, password):
    key = hashlib.sha256(password.encode()).digest()
    raw = base64.b64decode(encrypted)
    iv, ct = raw[:16], raw[16:]
    cipher = pyaes.AESModeOfOperationCBC(key, iv=iv)
    plain = b''
    for i in range(0, len(ct), 16):
        plain += cipher.decrypt(ct[i:i+16])
    pad_len = plain[-1]
    return json.loads(plain[:-pad_len].decode('utf-8'))

def init():
    global g_target_codes
    g_target_codes = get_signal()

def handlebar(ContextInfo):
    current_time = ContextInfo.get_current_time()
    if current_time.strftime('%H:%M') != '14:50':
        return

    global g_target_codes
    target = set(g_target_codes)
    current = set()
    vol_map = {}
    for code in ContextInfo.get_stock_code_list():
        if code.startswith(('11','12')):
            vol = ContextInfo.get_volume(code)
            if vol > 0:
                current.add(code)
                vol_map[code] = vol

    for code in current - target:
        vol = vol_map.get(code, 0)
        if vol > 0:
            print(f"  卖出 {code} x{vol}")
            passorder(24, 1101, ContextInfo.accID, code, 5, 0, vol, '', 1, '', ContextInfo)

    to_buy = target - current
    if to_buy:
        cash = ContextInfo.get_account_asset('STOCK').m_dCash * 0.95
        per = cash / len(to_buy)
        for code in to_buy:
            price = ContextInfo.get_full_tick(code).m_nMatch
            if price > 0:
                vol = int(per / price / 10) * 10
                if vol >= 10:
                    print(f"  买入 {code} x{vol}")
                    passorder(23, 1101, ContextInfo.accID, code, 5, price, vol, '', 1, '', ContextInfo)

    g_target_codes = get_signal()
```

## 四、健康监控

从COS拉取日志检查是否有异常：

```python
from qcloud_cos import CosConfig, CosS3Client
import json

config = CosConfig(Region='ap-shanghai', SecretId=SECRET_ID, SecretKey=SECRET_KEY)
client = CosS3Client(config)
resp = client.get_object(Bucket='cb-signal-1454417314', Key='cb_log.json')
log = json.loads(resp['Body'].read().decode('utf-8'))

if log['errors']:
    print(f"⚠️ {len(log['errors'])} 个错误")
    for e in log['errors']:
        print(f"  [{e['ts']}] {e['msg']}")
else:
    print(f"✅ 正常 运行{log['run_at']} 耗时{log['duration_s']}s")
```

## 五、数据管线

```
每天15:30 A800服务器自动执行:
  [1] akshare.bond_zh_cov()           → 1034只转债列表+转股价
  [2] akshare.bond_zh_hs_cov_spot()   → 304只CB当日实时价
  [3] FTShare MCP                      → 321只正股当日收盘价+写缓存
  [4] 波动率从parquet缓存计算           → 712只正股有历史
  [5] GPU MC定价 (M=5000)              → 324只转债定价
  [6] 选Top10 → AES加密 → 上传COS      → cb_signal.json
  [7] 上传运行日志                      → cb_log.json
```

## 六、回测结论

| 时期 | 市场 | 基准 | LSM策略 | 超额 |
|------|------|------|---------|------|
| 2023H1 | 震荡市 | -0.51% | +4.41% | **+4.92%** |
| 2025-2026 | 牛市 | +16.37% | -0.06% | -16.43% |

**牛市关策略满仓，震荡市/熊市开策略。** 当前是牛市，可能跑输基准——这是预期内的。

## 七、常见问题

**Q: 第一天没有信号？**
服务器上手工跑一次：
```bash
cd /home/xujiayang2/chenjunming/cb_mc && .venv/bin/python push_signal.py
```

**Q: COS读取失败？**
检查Windows是否能访问腾讯云内网。在cmd执行：
```
curl http://cb-signal-1454417314.cos.ap-shanghai.myqcloud.com
```

**Q: 想换选股数量？**
告诉服务器管理员改 `push_signal.py` 里的 `N_POS = 10`。

---

**文档版本**: 2026-07-16  
**服务器**: A800, cron 每日15:30  
**COS桶**: cb-signal-1454417314  
**AES密码**: 336018
