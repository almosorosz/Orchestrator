# Sentinel：Pi 与 PC 通过 Ethernet 或 Wi-Fi 连接

**History、PC 自动保存和云盘目录副本**：见 [两端更新与使用说明](pi/HISTORY.md)。
Pi 默认每 5 秒记录；PC 默认每 60 秒保存，断线后补传。
云盘目录副本由 PC 的同步客户端上传，页面区分本地写入与远端上传状态。

Sentinel 的采集、设备配置、事件 CSV 和摄像头服务运行在 Pi 上。PC 的 Sentinel
页面通过 HTTP 显示 Pi 提供的网页。PC 关闭或断线不会停止 Pi 的采集服务。

Sentinel 页面新增英文的 连接方式选项：`Ethernet` 和
`Wi-Fi`。两种方式分别记住 Pi 地址；点击 Connect 时保存当前
方式和地址。切换方式会停止旧连接，填好新地址后点击 Connect。
每种方式的 **Help** 按钮提供对应的英文配置步骤。

这里选择的是访问 Pi 的地址。Pi 加入哪个 Wi-Fi、使用什么学校账号，仍在 Pi
自己的系统网络设置中配置；PC 可以通过 Wi-Fi 或校园有线网络访问 Pi，前提是
校园网络允许两台设备之间通信。程序不会修改 PC 或 Pi 的网卡配置。

页面已精简为标题、连接控件、状态和仪表盘。配置步骤放在 Help 中。
旧的 `brain/` 文件路径与设置键保留，已有地址可以继续使用。

**TOPDON TC001 热像仪接入**：参见 [Pi 接线、更新与启动步骤](pi/TC001.md)。
该功能需要更新 Pi 的 `app.py`、网页，并新增 `tc001_reader.py`；
现在运行 `python app.py` 会默认自动检测 TC001，`THERMAL_DEVICE=off` 可关闭。
PC 端 Sentinel 的连接方式沿用原设置。开机自动运行参见 [Pi 开机启动](pi/AUTOSTART.md)。

## 文件与修改

- `gui/pages/brain_page.py`：嵌入网页、分别保存 Ethernet/Wi-Fi 地址、连接方式选择和帮助、异步检查传感器 API、
  超时提示、每 5 秒重连、刷新、外部浏览器入口、文件下载。
- `MAIN_WINDOW.py`：退出前停止 Sentinel 客户端定时器、请求和下载。
- `brain/pi/`：基于用户提供文件的 Pi 部署版本。
  其中 `fumehood_dashboard.html` 取消了请求失败后随机生成 Demo 数据的行为，
  改为显示断线、最后一次更新时间和离线状态；恢复连接后自动更新。
  `app.py` 已增加 TC001 接口及摄像头区分；`tc001_reader.py` 为新增读取模块。
  Pico 串口连接、重试、断线和解析错误提示已全部改为英文；固件功能保持原样。
  `install_autostart.py` 可在当前 conda 环境中安装 Pi 开机服务。
- `brain/requirements-desktop.txt`、`brain/pi/requirements.txt`：两端依赖说明。
- `tests/test_brain_page.py`、`tests/test_brain_dashboard.cjs`：本地模拟服务验证。

本次直接修改当前项目，没有生成备份。仅切换无线/网线时无需更新 Pi；
启用 TC001 则需要按 `pi/TC001.md` 部署新增文件。Pico 固件无需更新。

## 1. 接线与固定 IP

用普通网线连接 Pi 的 RJ45 网口与 PC 网口；Pi 单独使用合适的电源。
如果 PC 原有网口需要接校园网，可增加 USB 转以太网适配器，专门连接 Pi。

只对这两个直连接口设置以下地址：

| 设置 | PC | Pi |
| --- | --- | --- |
| IPv4 | `192.168.77.1` | `192.168.77.2` |
| 子网掩码 | `255.255.255.0` | `255.255.255.0` |
| 网关、DNS | 留空 | 留空 |

请先确认 `192.168.77.0/24` 未与已有校园网、VPN 或其他接口冲突；如冲突，
选择另一空闲私有网段，并同步修改 Sentinel 页面的地址。

Windows：运行 `ncpa.cpl` → 找到连接 Pi 的网卡 → 属性 →
Internet 协议版本 4 (TCP/IPv4) → 填写上述设置。
不要改动校园网网卡，也不需要启用网桥或 Internet 连接共享。

Pi：在网络设置中编辑有线连接，IPv4 选择手动，填写 `192.168.77.2/24`，
网关与 DNS 留空。使用 NetworkManager 命令行时，先运行 `nmcli device status`
确认实际的有线接口，再配置相应连接；不要直接套用未知接口名称。

两台机器可以同时通过其他接口上网。Sentinel 本地通信不需要互联网；网页中的
天气和 Google Fonts 则需要外网，外网不可用时不影响本地传感器 API。

## 2. Wi-Fi 连接配置

### Pi 已经连上校园 Wi-Fi 时

保留之前 `eth0` 的 `192.168.77.2/24`。Pi 的 Wi-Fi IPv4 使用
**Automatic (DHCP)**，不要把无线接口也设置成网线的地址。

1. 在 Pi 上运行以下命令，读取无线接口的地址：

   ```bash
   ip -4 addr show wlan0
   ```

   找到 `inet` 后的 IPv4，去掉 `/20`、`/24` 等后缀。不要使用 `lo` 的
   `127.0.0.1`，也不要使用 `eth0` 的 `192.168.77.2`。
   如果找不到 `wlan0`，运行 `nmcli device status`，找出 TYPE 为 `wifi` 的
   实际设备名，再用它替换命令中的 `wlan0`。

2. 保持现有 Sentinel 服务运行：在存放 `app.py` 的目录和原来能正常运行的
   Python 环境中执行 `python app.py`。如果服务已启动，不要再启动第二个实例。
   现有 `app.py` 已监听 `0.0.0.0:8000`，同时支持有线与无线访问。

3. PC 接入能访问 Pi 的实验室/校园网络。可以使用 Wi-Fi，也可以继续使用
   校园有线网；是否能互访取决于学校的网络设置。

4. 在 PC 浏览器访问 `http://Pi无线IP:8000`。例如，**仅当 Pi 实际显示的 IP
   是 `10.186.99.100` 时**，填写 `http://10.186.99.100:8000`。
   这个地址只是示例，不能直接当作你的 Pi 当前地址。

5. 打开 Orchestrator → Sentinel → 连接方式选择 **Wi-Fi** →
   地址栏 填写同一个完整地址 → **Connect**。

### Pi 尚未加入校园 Wi-Fi 时

从 Pi 桌面右上角网络图标选择学校的 Wi-Fi，并按学校要求完成账号和证书配置。
Purdue 的 PAL3.0/eduroam 当前推荐使用学校的
[SecureW2 配置入口](https://service.purdue.edu/TDClient/32/Purdue/KB/Article/399/How-to-connect-to-Wi-Fi-at-Purdue-West-Lafayette-including-Purdue-Indianapolis)。
加入 Wi-Fi 后，再执行上面的地址查询与连接步骤。

### 无线连接失败时

在 **PC PowerShell** 中，把示例 IP 替换为 Pi 当前的无线 IP 后执行：

```powershell
Test-NetConnection 10.186.99.100 -Port 8000
```

`TcpTestSucceeded : True` 表示 TCP 端口可达，再检查网页/API；False 时检查
Pi 服务是否运行、无线 IP 是否变化、Pi 防火墙的入站 TCP 8000 规则，以及
校园网是否限制设备互访。同一个 Wi-Fi 名称并不保证可互访。不要用“Pi 能上网”
作为“PC 能连接 Pi”的判断依据。

若学校网络阻止互访，需要学校 IT 调整访问权限，或在 Sentinel 中切回
**Ethernet**，使用 `http://192.168.77.2:8000`。
程序不会绕过学校的网络隔离，也不会在断线时自动切换到另一个地址。

校园 Wi-Fi 的 DHCP 地址可能在重新连接后变化；连接失败时重新查询并更新
Sentinel 地址。若需要长期固定无线地址，应向学校 IT 申请适合该网络的地址分配。

## 3. Pi 端部署

若 Pi 上已有正在运行的 Sentinel，可用本仓库
`brain/pi/fumehood_dashboard.html` 替换原网页，然后刷新页面。
启用 TC001 时还需要更新 `app.py` 和新增读取模块，详见上方专用说明。
保留现有的 `userinfo.json`、`probeconfig.json`、`event_log.csv`。

如果首次部署，将 `brain/pi/` 内文件复制到 Pi 的专用目录，在该目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

如使用摄像头，另安装 `opencv-python-headless`，并确认 Pi 已能读取对应
V4L2 摄像头。需要 Pi 的 1-Wire 配置和 USB 串口访问权限与原系统一致。

`pico_main.py` 是 **Pico 的 MicroPython 固件**：仅在首次安装或需要更新 Pico
时，将它作为 `main.py` 放到 Pico；不要在 PC 或 Pi 的普通 Python 中运行。

保持 `python app.py` 进程运行，PC 才能访问页面和数据。需要开机自动运行时，
按 `pi/AUTOSTART.md` 安装 systemd 服务；服务以当前用户和 Python/conda 环境
运行同一个主入口。后台采集与日志线程
由 `app.py` 的主入口启动，不要直接换成 `flask run` 或普通 WSGI 导入启动，
否则这些线程不会启动。

## 4. PC 端

在启动 Orchestrator 的同一个 Python 环境内安装完整的 PySide6：

```powershell
python -m pip install -r brain/requirements-desktop.txt
python MAIN_WINDOW.py
```

也可以单独打开 Sentinel 页验证连接，不初始化实验设备：

```powershell
python -m gui.pages.brain_page
```

网线连接：进入 Sentinel → 连接方式选择 **Ethernet** →
地址栏 填写 `http://192.168.77.2:8000` → Connect。
无线连接：连接方式选择 **Wi-Fi**，填写 Pi 的无线地址再连接。

两套地址保存在 PC 当前用户的 Qt 设置（Windows 为
`HKEY_CURRENT_USER\Software\Crystalloop\Orchestrator`），下次首次进入 Sentinel 时
恢复最近点击 Connect 使用的方式和地址并自动连接。Disconnect 停止本次自动
重连；保存的地址仍会保留。只切换下拉框而未点击 Connect 时，最近连接方式
不会被替换。两种方式未点击 Connect 的地址编辑在当前页面内暂存。

原来的 `brain/url` 设置会迁移保存为 Ethernet 地址；之后分别保存
`brain/ethernet_url`、`brain/wifi_url` 和 `brain/connection_mode`。
新安装时 Wi-Fi 地址为空，需要先输入实际地址，不会自动复用网线地址。

连接检查调用 `/api/sensors`，每 5 秒一次，单次超时 4 秒，不阻塞桌面界面。
“服务已连接”只表示 API 可访问，具体探头是否在线以网页显示为准。
断线后隐藏并停止旧网页和视频，显示最后成功检查时间，恢复后重新加载网页。
点击网页的 Download CSV 会弹出 PC 的保存对话框，取消时不会下载。

## 5. 排查与验收

1. Pi 本机打开 `http://localhost:8000/api/sensors`，确认得到 JSON 数组。
2. PC 浏览器打开 `http://192.168.77.2:8000`，确认网页和真实数据更新。
3. 如浏览器打不开，检查固定 IP、网线、服务进程和 Pi 的 TCP 8000 入站规则。
4. Sentinel 页连接成功后，拔掉网线；约一个检查周期加超时时间内（通常不超过
   9 秒）应显示不可用，不能继续显示旧读数为实时值。
5. 插回网线，确认自动恢复；测试温度、门磁、视频（如已配置）及 CSV 下载。
6. 直接用浏览器访问 Pi 网页时，API 失效也应显示断线，绝不生成随机 Demo 数据。
7. 在 Wi-Fi 模式连接 Pi 的无线地址，确认实际传感器数据更新。测试无线通信
   可暂时拔掉 PC 与 Pi 的直连网线（服务保持运行），验证无线地址仍能访问。
8. Wi-Fi 与 Ethernet 之间切换，确认分别恢复各自地址；点击 Connect 后再开始
   连接。模拟断线后应在同一模式重试，不会切换到另一套地址。

## 本地测试

以下测试使用模拟 HTTP 服务，不连接实验硬件。Python 测试需完整 PySide6。
旧的 `tests/test_controller.py` 引用了已不存在的 `app_control.Controller`，
所以请按下面命令单独运行本次测试：

```powershell
python -m unittest discover -s tests -p test_brain_page.py -v
node tests/test_brain_dashboard.cjs
```

TC001 解码及 Pi API 测试需要 `brain/pi/requirements-thermal.txt` 中的依赖：

```powershell
python -m unittest discover -s tests -p test_tc001.py -v
python -m unittest discover -s tests -p test_pi_autostart.py -v
python -m unittest discover -s tests -p test_sentinel_history.py -v
python -m unittest discover -s tests -p test_sentinel_history_ui.py -v
```

这些测试使用构造的原始帧，不代表实物 TC001 已校准或已在你的 Pi 上验证。

参考：[Qt WebEngine](https://doc.qt.io/qtforpython-6/PySide6/QtWebEngineWidgets/QWebEngineView.html)、
[Flask 部署说明](https://flask.palletsprojects.com/en/stable/deploying/)、
[Raspberry Pi 网络配置](https://www.raspberrypi.com/documentation/computers/configuration.html#networking)。
