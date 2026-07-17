# 可转债MC定价自动交易 — QMT部署文档

## 一、是什么

浙大A800服务器每天15:30自动MC定价(5000路径, 212只转债)，选出Top10，AES加密后推送到GitHub Gist。

你现在要把QMT客户端跑起来：每天14:50从Gist拉取→解密→用passorder调仓。

## 二、准备

| 项目 | 内容 |
|------|------|
| Gist地址 | `https://gist.githubusercontent.com/Jimmy-Chern/4f3d9550b2c5e874019272b025a58408/raw/cb_signal.json` |
| 解密密码 | `336018` |

## 三、安装依赖（QMT Python终端）

```
pip install pyaes requests
```

`pyaes` 是纯Python实现的AES，不需要C编译，QMT内置Python 3.6也能装。

## 四、完整QMT策略代码

**全部复制到QMT策略编辑器，改一处：`TARGET_CODES` 初始值留空即可，其他不用改。**

```python
# ===== 可转债MC定价策略 QMT客户端 =====
import requests, json, hashlib, base64, os
import pyaes

GIST_URL = "https://gist.githubusercontent.com/Jimmy-Chern/4f3d9550b2c5e874019272b025a58408/raw/cb_signal.json"
PASSWORD = "336018"

def decrypt_aes(encrypted, password):
    """AES-CBC解密 (pyaes, Python 3.6兼容)"""
    key = hashlib.sha256(password.encode()).digest()
    raw = base64.b64decode(encrypted)
    iv, ct = raw[:16], raw[16:]
    cipher = pyaes.AESModeOfOperationCBC(key, iv=iv)
    plain = b''
    for i in range(0, len(ct), 16):
        plain += cipher.decrypt(ct[i:i+16])
    pad_len = plain[-1]
    return json.loads(plain[:-pad_len].decode('utf-8'))

def get_signal():
    try:
        resp = requests.get(GIST_URL, timeout=15)
        signal = decrypt_aes(resp.json()["encrypted"], PASSWORD)
        print(f"[{signal['date']}] Top10: {[p['code'] for p in signal['positions']]}")
        return [p['code'] for p in signal['positions']]
    except Exception as e:
        print(f"信号获取失败: {e}")
        return []

def init():
    global g_target_codes
    g_target_codes = get_signal()

def handlebar(ContextInfo):
    # 每天14:50检查是否需要调仓
    current_time = ContextInfo.get_current_time()
    if current_time.strftime('%H:%M') != '14:50':
        return
    
    global g_target_codes
    target = set(g_target_codes)
    
    # 当前转债持仓
    current = set()
    vol_map = {}
    for code in ContextInfo.get_stock_code_list():
        if code.startswith(('11','12')):
            vol = ContextInfo.get_volume(code)
            if vol > 0:
                current.add(code)
                vol_map[code] = vol
    
    to_sell = current - target
    to_buy  = target - current
    
    # 卖出
    for code in to_sell:
        vol = vol_map.get(code, 0)
        if vol > 0:
            print(f"  卖出 {code} x{vol}")
            passorder(24, 1101, ContextInfo.accID, code, 5, 0, vol, '', 1, '', ContextInfo)
    
    # 买入
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
    
    # 更新信号（收盘后Gist已更新，为明天准备）
    g_target_codes = get_signal()
```

## 五、信号格式

```json
{
  "date": "2026-07-15",
  "buy": ["110092", "113042", ...],
  "positions": [{
    "code": "110092", "name": "三房转债",
    "model_price": 112.29, "market_price": 74.40,
    "discount": 50.92, "conv_price": 3.02
  }, ...]
}
```

## 六、部署检查清单

- [ ] `pip install pyaes requests` 成功
- [ ] QMT策略编辑器粘贴代码
- [ ] 运行策略，看日志是否有 `[2026-xx-xx] Top10: [...]` 输出
- [ ] 确认14:50时QMT在线、策略在运行

## 七、常见问题

**Q: Gist拉不到？**
GitHub被墙的话需要代理。或者让服务器换国内文件托管（联系管理员）。

**Q: 第一天没有历史信号怎么办？**
让服务器管理员手工跑一次推送：
```bash
cd /home/xujiayang2/chenjunming/cb_mc && .venv/bin/python push_signal.py
```

**Q: passorder报错？**
确认参数：23=买入, 24=卖出, 1101=转债, 5=对手价, accID用ContextInfo.accID。

## 八、策略原理

- 蒙特卡洛5000条路径 + 最小二乘回归 → 每条转债理论价
- 折价率 = (理论价-市场价)/市场价 → 买最高的10只
- 每日调仓，等权分配
- **回测结论：震荡市有效(2023年超额+5%)，牛市需关策略满仓持有**

---

**管理员**: A800 cron 每天15:30自动推送
**密码**: 336018
