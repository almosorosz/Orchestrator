# Sentinel 在 Pi 开机后自动运行

TC001 现在随 `python app.py` 默认自动检测。以下配置让整个 Pi 服务在开机后
自动运行，不需要打开终端或登录远程桌面；程序退出报错时会自动重启。

## 安装一次

把更新包内文件放进原来的程序目录。保留 `userinfo.json`、`probeconfig.json`、
`event_log.csv`、已有的 `sentinel_history.sqlite3`，无需更新 Pico 的固件。
确保新增的 `history_store.py` 也放在同一目录。先在旧服务终端按 Ctrl+C 停止手动运行
的 `app.py`，避免重复占用 8000 端口。

根据你截图里的目录和环境，在 **Pi 终端**执行：

```bash
conda activate envmo
cd /home/oybc1999/Desktop/Pi/Pi
python install_autostart.py
```

如果你改了目录或环境名，请使用实际位置。最后一行不要加 `sudo`，安装器在
写入系统服务时会自行调用 sudo，必要时输入 Pi 的系统密码。

安装器会：

- 检查当前环境里的 Flask、pyserial、NumPy、OpenCV 和必要程序文件。
- 自动记录当前用户、程序目录、Python 解释器与 conda 环境位置。
- 对 conda 环境使用 `conda run --no-capture-output --prefix ...` 启动，保留环境激活行为。
- 安装 `/etc/systemd/system/sentinel.service`，以当前普通用户运行。
- 设置开机启动、立即启动，并检查本机 API 是否就绪。

正常完成后显示 `Sentinel is running and will start automatically at boot.`。
它表示服务已启动；热像仪是否采集成功仍以 `/api/thermal` 的 `available` 为准。
关掉终端、断开远程桌面不会停止此服务。有线和 Wi-Fi 访问方式保持原样。

若提示端口已占用，先停止仍在手动运行的旧 `app.py`。安装完成后不要再手动
运行第二个 `python app.py`。

只预览将生成的系统服务、不安装：

```bash
python install_autostart.py --dry-run
```

## 查看状态、更新程序、关闭开机启动

```bash
systemctl status sentinel.service --no-pager
journalctl -u sentinel.service -n 50 --no-pager
```

实时日志：`journalctl -u sentinel.service -f`，按 Ctrl+C 仅退出日志查看。

更新程序文件后重启：

```bash
sudo systemctl restart sentinel.service
```

移动程序目录或更换 conda 环境后，进入新目录、激活新环境，重新运行安装器。
Python 包未变化时无需重复安装。

临时停止服务：

```bash
sudo systemctl stop sentinel.service
```

关闭开机启动并停止：

```bash
sudo systemctl disable --now sentinel.service
```

重新开启：`sudo systemctl enable --now sentinel.service`。

重启 Pi 后检查状态和 `http://127.0.0.1:8000/api/thermal`。设备缺失不会阻止
网页启动，Pico 和 TC001 会在后台重试；不要把服务启动成功当作设备已在线。

该安装器已在本地验证配置生成和安装流程的模拟测试；仍需在你的 Pi 上完成
实际安装与重启验证。

参考：[systemd 服务](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)、
[conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html)。
