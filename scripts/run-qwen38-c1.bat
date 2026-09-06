@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Put this launcher beside the v0.6.0 release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen38.bat first, or drag a qwen3_8_27b.ninfer file onto this launcher.
  exit /b 1
)

echo Starting Qwen3.8-27B at http://127.0.0.1:8080/v1
echo Profile: one request, 256K context, rk4v4-e8 KV, MTP3, ReplaySSM
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft
