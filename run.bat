@echo off
rem Builds ZawUpscale (Release) and runs it. Extra arguments are passed to the app, e.g.:
rem   run.bat --no-browser --port 5480
setlocal
cd /d "%~dp0"

where dotnet >nul 2>nul
if errorlevel 1 (
    echo The .NET SDK was not found. Install .NET 10 from https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

echo Building ZawUpscale...
dotnet build src\ZawUpscale\ZawUpscale.csproj -c Release -nologo -v q
if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

dotnet src\ZawUpscale\bin\Release\net10.0\ZawUpscale.dll %*
if errorlevel 1 pause
