# History：Pi 记录、PC 保存、云盘目录副本

## 这版完成了什么

- Pi 默认每 5 秒保存温度、门状态和 TC001 的最高／最低／平均／中心温度。
- Events 记录报警等级变化、门状态变化、设备上线／离线以及正常服务启停。
- History 提供 Events / Readings、日期和传感器筛选、翻页、完整选区 CSV 导出。
- PC 默认每 60 秒增量保存，连接恢复和重开程序后从上次记录编号继续。
- 可额外写入 PC 上的云盘同步目录，由你的云盘客户端上传。
- 有线、Wi-Fi、TC001 默认检测、英文界面和可选开机启动均保留。

本版不直接登录云服务，不需要云端账号或 API 密钥，也没有 Pi 直传云端功能。
云盘备份需要 PC、Orchestrator 和云盘客户端持续运行。

## 1. 更新 Pi

先停止旧程序：手动启动的在终端按 Ctrl+C；已安装系统服务的执行
`sudo systemctl stop sentinel.service`。

把 Pi 更新包中的文件复制到原有目录。必须包含新文件 `history_store.py`。
保留以下运行数据：

- `userinfo.json`、`probeconfig.json`
- `event_log.csv`
- 如果已经运行过新版，保留 `sentinel_history.sqlite3`；停止服务后也不要删相关的
  `sentinel_history.sqlite3-wal` 和 `sentinel_history.sqlite3-shm` 文件。

首次运行会把旧 CSV 导入数据库一次，不会修改旧 CSV；格式错误的旧行会跳过，
History 显示跳过数量。原 CSV 文件继续保留，页面统一使用 Export CSV 导出。

没有新增第三方依赖，SQLite 使用 Python 自带的 `sqlite3`。沿用已能运行 TC001
的 conda 环境即可。

手动启动，按你目前的环境和目录：

```bash
conda activate envmo
cd /home/oybc1999/Desktop/Pi/Pi
python app.py
```

如果原来就使用系统服务，更新文件后改为执行：

```bash
sudo systemctl restart sentinel.service
```

不要同时手动启动和启动系统服务。开机启动仍是可选项，见 `AUTOSTART.md`。
Pico 固件无需更新。

## 2. 更新并运行 PC

解压完整的 Orchestrator 更新包，在原来的 PC Python 环境中执行：

```powershell
python MAIN_WINDOW.py
```

也可以单独运行 Sentinel：`python -m gui.pages.brain_page`。
沿用 PySide6，无新增第三方依赖。两个新增 PC 模块在 `gui/` 中，不能只替换
`brain_page.py`。

在 Sentinel 选择 Ethernet 或 Wi-Fi，连接 Pi 后进入 History。
默认 PC 目录为当前用户 Documents 下的 `Sentinel Logs`。
点击 **Log settings** 或 History 的 **PC → Configure** 可更换目录、关闭保存、
调整间隔（10–3600 秒）。修改目录会为新目录重新补传 Pi 仍保留的全部记录。

目录结构如下，日期均按 UTC 分组：

```text
Sentinel Logs/
  <Pi dataset ID>/
    archive.sqlite3
    2026-10-21-events.csv
    2026-10-21-readings.csv
```

数据库是完整的 PC 本地副本，CSV 可用于 Excel 等工具分析。
同一 Pi 改 IP 或在 Ethernet / Wi-Fi 间切换，仍使用同一数据库编号，不重复写入。
不要删 Pi 的数据库来更新代码；重建数据库会产生新的编号和独立的 PC 归档目录。

Orchestrator 关闭或点击 Disconnect 后不再同步，但 Pi 持续记录；重新连接会补传。
保存任务在后台执行，离开 Sentinel 页面不会停止已经建立连接的自动保存。

## 3. 云盘目录副本

在 PC 的 **Log settings** 中：

1. 勾选 **Also save a copy to a cloud sync folder**。
2. 用 **Browse…** 选择已由你的云盘客户端管理的现有目录。
3. 保存。Sentinel 首次立即写入，之后随 PC 保存任务约每 5 分钟刷新一次；
   如果 PC 保存间隔大于 5 分钟，则也使用较长的间隔。
4. 点击 History 的 **Sync now** 可以要求立即刷新。

云盘目录中会生成：

```text
<Selected folder>/Sentinel/<Pi dataset ID>/
  history.sqlite3
  2026-10-21-events.csv
  2026-10-21-readings.csv
  manifest.json
```

数据库使用 SQLite 一致性快照生成，不会直接复制正在写入的数据库文件。
CSV 和快照先写临时文件再替换，失败后会在后续保存任务重试。

**Folder saved · upload not verified** 只表示文件已写到 PC 的云盘目录。
请在云盘客户端检查实际上传状态；Sentinel 不会把本地写入成功显示成云端上传成功。
未安装同步客户端、客户端未登录或目录未被同步时，文件只保存在 PC 上。

只开云盘副本、关闭普通 PC 保存时，程序仍使用本机应用数据目录中的缓存数据库
暂存传输进度。云端账号和密码不放入 Pi 网页。

## 4. History 页面设置

- **Events**：状态变化和系统事件。
- **Readings**：每个采样时刻的传感器数据，热像数值不包含整段视频或截图。
- **Today / Last 7 days / All time / Custom dates**：按浏览器的本地日期筛选。
- **Export CSV**：导出整个选区，不限于页面显示的 200 行。CSV 时间为 UTC。
- **Load more**：查看较早记录；浏览较早记录时暂停列表自动替换，点击
  **Refresh latest** 返回实时更新。
- **Settings**：调整采样间隔和 Pi 保留天数。默认 0 天表示不自动删除。

保留天数大于 0 时，会在采集过程中删除 Pi 数据库中早于该时间的记录；
页面会在修改时确认。PC 已保存的记录和原始 CSV 不受影响。若 PC 太久未连接，
部分记录在补传前已经过期，PC 会显示缺失提示，无法恢复已删除的 Pi 原始数据。

事件按采样时刻比较状态，短于采样间隔的变化可能不会被记录；需要捕获较短变化时
可缩短间隔。断电无法正常写入“Service stopped”；重启后会有“Service started”。

## 5. 验证

Pi 本机浏览器：`http://127.0.0.1:8000/api/history/status`。
`recording` 应为 true，`record_count` 应随采样增长。

PC 连接后确认 History 的 PC 状态为 Saved，所选目录中生成数据库和当天 CSV。
断开连接、等待 Pi 记录几次，再连接，确认 CSV 补齐且没有重复记录。
更换 IP 后仍应写入相同的 Pi dataset ID 子目录。

本地测试已覆盖真实 HTTP / SQLite、页面筛选与下载、Qt 后台任务、断线续传、
导出失败重试及云盘快照；仍需要在你的实际 Pi、PC 和云盘客户端上验证。
