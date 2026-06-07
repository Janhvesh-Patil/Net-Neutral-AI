@echo off
REM ============================================================
REM  Net-Neutral AI — Demo Launcher
REM  Run from the project root on the coordinator machine
REM  Requires: Windows Terminal installed
REM ============================================================

echo.
echo  ============================================================
echo   Net-Neutral AI   ^|   Demo Launcher
echo  ============================================================
echo.
echo  Opens 4 tiled terminals:
echo    Top Left     : Coordinator (server.py)
echo    Top Right    : Client A
echo    Bottom Left  : Client B
echo    Bottom Right : Client C
echo.
echo  BEFORE RUNNING — confirm all of the following:
echo    1. shared\config.py has correct COORDINATOR_IP
echo    2. coordinator\checkpoint.pt exists
echo    3. All machines are on the same WiFi network
echo    4. Port 5000 is reachable from client machines
echo    5. database.db has been reset if this is the recording run
echo.
echo  NOTE: checkpoint.pt is already committed to the repo.
echo  There is no need to run pretrain.py again.
echo  If you need to verify the checkpoint, run:
echo    python coordinator\pretrain.py data --verify-only
echo.

pause

REM ── Launch Windows Terminal with 4 panes ──────────────────────
REM Coordinator starts first.
REM Clients wait 8 seconds to ensure coordinator is ready before connecting.

wt.exe ^
  new-tab --title "Coordinator" --startingDirectory "%~dp0.." ^
    cmd /k "cd /d %~dp0.. && venv\Scripts\activate && echo [Coordinator] Starting... && python coordinator\server.py && pause" ^
  ; split-pane --vertical --title "Client A" ^
    cmd /k "cd /d %~dp0.. && venv\Scripts\activate && timeout /t 8 /nobreak && python client\client.py --client_id client_A && pause" ^
  ; split-pane --horizontal --title "Client B" ^
    cmd /k "cd /d %~dp0.. && venv\Scripts\activate && timeout /t 8 /nobreak && python client\client.py --client_id client_B && pause" ^
  ; move-focus previous ^
  ; split-pane --horizontal --title "Client C" ^
    cmd /k "cd /d %~dp0.. && venv\Scripts\activate && timeout /t 8 /nobreak && python client\client.py --client_id client_C && pause"

echo.
echo  All terminals launched.
echo  Wait for all 3 clients to register before recording starts.