@echo off
rem One-click launcher for the Object Counter app.
rem Uses the project's own virtual environment (.venv) so imports always work.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    echo [!] No .venv found - trying system Python...
    python app.py
)
pause
