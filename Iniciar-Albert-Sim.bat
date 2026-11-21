@echo off
setlocal
cd /d "%~dp0"

title Albert-sim - cargando
cls
echo ================================================
echo              Albert-sim
echo ================================================
echo.
echo Iniciando la aplicacion...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1 de 3] Preparando el entorno de Python...
    echo         Esto puede tardar unos minutos la primera vez.
    echo.
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo No se pudo crear el entorno de Python.
    echo Instale Python 3.12 o una version posterior desde https://www.python.org/downloads/
    echo Durante la instalacion marque "Add Python to PATH".
    pause
    exit /b 1
)

echo [2 de 3] Revisando e instalando los componentes necesarios...
echo         Puede tardar mientras prepara la aplicacion.
echo.
if not exist ".venv\.requirements-installed" (
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo.
        echo No se pudieron instalar las dependencias.
        pause
        exit /b 1
    )
    type nul > ".venv\.requirements-installed"
    echo         Componentes instalados correctamente.
) else (
    echo         Los componentes ya estan preparados.
)

echo.
echo [3 de 3] Iniciando Albert-sim...
echo         La pagina se abrira automaticamente.
echo.
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run "src/app.py" --server.port=8501 --server.address=localhost --server.headless=true --browser.gatherUsageStats=false
if errorlevel 1 (
    echo.
    echo La aplicacion no pudo iniciarse.
    pause
)
endlocal
