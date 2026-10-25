@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
 echo.
 echo 起動に失敗しました。次のコマンドで必要なライブラリを導入してください。
 echo python -m pip install -r requirements.txt
 pause
)
