@echo off
REM 切换到脚本自身所在目录，避免从其他位置双击时找不到模型和网页文件。
cd /d "%~dp0"

set "ANACONDA_ROOT=D:\software\develop-software\Anaconda3"

if not exist "%ANACONDA_ROOT%\envs\ai-common\python.exe" (
    echo 找不到 ai-common 环境，请先创建或修复该 Conda 环境。
    pause
    exit /b 1
)

REM 激活通用 AI 环境后启动 Flask 本地网页。
call "%ANACONDA_ROOT%\Scripts\activate.bat" "%ANACONDA_ROOT%\envs\ai-common"
python web_app.py

REM 如果依赖缺失或启动失败，暂停窗口以便看到错误提示。
pause
