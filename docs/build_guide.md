# Build & Distribution Guide

Compile Survey Studio Pro into a standalone distribution that runs without Python installed.

## Build Tool: PyInstaller (Recommended)

> **Note**: Nuitka was originally planned but crashes on Python 3.13 (`AssertionError: listcomp_1__.0_clone`). PyInstaller is the working alternative.

## Prerequisites

1. **Python 3.11+** with all project dependencies installed (Poetry environment)
2. **PyInstaller** (recommended):
   ```bash
   pip install pyinstaller
   ```
3. **Nuitka** (alternative — requires Python ≤ 3.12):
   ```bash
   pip install nuitka ordered-set
   ```
   Plus a C compiler:
   - **MSVC**: Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"
   - **MinGW-w64**: Nuitka can auto-download if MSVC is not found

### Installed packages (for reference)

```
pyinstaller          6.19.0
pyinstaller-hooks-contrib  2026.3
altgraph             0.17.5
pefile               2024.8.26
nuitka               4.0.5   (optional, does NOT work with Python 3.13)
ordered-set          4.1.0   (optional, for Nuitka)
```

## Building with PyInstaller

From the project root:

```bash
python build_pyinstaller.py
```

This takes **10-20 minutes** on a typical machine. Output goes to `dist/main_local/`.

### Building with Nuitka

Nuitka compiles Python to native C code — nearly impossible to reverse-engineer.
Output goes to `dist/main_local.dist/`.

#### Option A: Python 3.13 + MSVC (requires admin)

If you have admin rights to install Visual Studio Build Tools:

```powershell
# 1. Install Visual Studio Build Tools (requires admin / UAC approval)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# 2. Install Nuitka in your existing Poetry environment
poetry run pip install --upgrade nuitka ordered-set

# 3. Build
$env:NUITKA_ASSUME_YES = "1"
poetry run python build_nuitka.py
```

#### Option B: Python 3.12 + MinGW (no admin needed)

If you cannot install MSVC, use Python 3.12 — Nuitka will auto-download MinGW.

```powershell
# 1. Install Python 3.12 side-by-side (does not affect existing 3.13)
winget install Python.Python.3.12

# 2. Install Poetry for Python 3.12
py -3.12 -m pip install poetry

# 3. Create a Python 3.12 virtual environment
cd c:\SDC_EM
py -3.12 -m poetry env use 3.12

# 4. Install project dependencies
#    If `poetry lock` hangs on sdv/sdmetrics resolution, skip it and pip install directly:
py -3.12 -m poetry run pip install panel bokeh plotly pandas numpy scipy scikit-learn polars dask pyarrow matplotlib openpyxl pydantic pydantic-settings loguru param tornado jinja2 xyzservices contourpy PyNaCl

# 5. Install Nuitka
py -3.12 -m poetry run pip install --upgrade nuitka ordered-set

# 6. Build
$env:NUITKA_ASSUME_YES = "1"
py -3.12 -m poetry run python build_nuitka.py
```

> **Note**: The `ed25519` package fails to build on Python 3.12+ (`SafeConfigParser` removed). Use `PyNaCl` instead — `ed25519` is not imported by the app.

#### Known Nuitka 4.0.5 issues

- **Clone bug** (`AssertionError: listcomp_1__.0_clone`): Crashes on `panel` and `bokeh` modules. The build script works around this by excluding them from C compilation (`--nofollow-import-to=panel/bokeh`). They are bundled as bytecode instead.
- **First run**: Nuitka will ask to download Dependency Walker — set `$env:NUITKA_ASSUME_YES = "1"` to auto-accept.
- **Build time**: 1-2 hours (C cache is reused on subsequent builds).

## What Gets Built

```
dist/main_local/
├── main_local.exe          # Application executable
├── run_sdc.bat             # Launcher (pre-configured with license keys)
├── sdc_engine/
│   ├── config.yml          # App configuration
│   ├── config.ini
│   ├── strategy.yml
│   └── assets/
│       └── (assets)
├── python313.dll           # Python runtime (bundled)
├── *.pyd                   # Extension modules
└── [other bundled DLLs]
```

## Dev Server (Local Development)

Use the PowerShell launcher script — it sets all required env vars automatically:

```powershell
.\start_dev.ps1
```

This runs `panel serve` on `http://localhost:5006/main`.

On the `sdc-ai` branch, set the Cerebras API key first for AI features:

```powershell
$env:CEREBRAS_API_KEY = "your-cerebras-key"
.\start_dev.ps1
```

Or manually:

```powershell
$env:KEYGEN_ACCOUNT_ID = "de813b66-fa08-40d8-b9da-0fdc1c47b876"
$env:KEYGEN_PUBLIC_KEY = "49f345e346da8348c29d1311e40017366d9d27c574f5d5780f99e95a7c0141cf"
panel serve sdc_engine/main.py --port 5006 --allow-websocket-origin='*'
```

## Distributing to Clients

1. Run `python build_pyinstaller.py`
2. The launcher `run_sdc.bat` comes pre-configured with Keygen credentials
3. Zip the entire `dist/main_local/` folder
4. Ship to client

## Client Usage

1. Unzip to any directory
2. Double-click `run_sdc.bat`
3. Browser opens automatically at `http://localhost:80/sdc_engine`
4. To stop: close the console window

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `KEYGEN_ACCOUNT_ID` | Keygen.sh account identifier |
| `KEYGEN_PUBLIC_KEY` | Ed25519 public key for license validation |
| `CEREBRAS_API_KEY` | Cerebras Cloud API key (AI features, `sdc-ai` branch only) |
| `BOKEH_ADDRESS` | Bind address (default: `0.0.0.0`) |
| `BOKEH_PORT` | Server port (default: `80`) |

## Code Protection Options

PyInstaller bundles Python as `.pyc` bytecode (not source `.py` files). Clients cannot read your code directly, but `.pyc` can theoretically be decompiled. Below are options for stronger protection.

### Default (No extra setup)

- **Protection level**: Moderate
- `.pyc` bytecode only — no readable source shipped
- Python 3.13 bytecode is not yet supported by major decompilers (`uncompyle6`, `decompyle3`)
- Sufficient for most B2B clients

### PyArmor (Recommended for extra protection)

- **Protection level**: Strong
- Obfuscates + encrypts Python bytecode with per-machine binding
- Works with PyInstaller out of the box
- Free for personal use; commercial license ~$56/year

```bash
pip install pyarmor
# Obfuscate your code first, then run PyInstaller on the obfuscated output
pyarmor gen -O dist_protected -r sdc_engine/
```

### Cython Compilation

- **Protection level**: Very strong (native machine code)
- Compiles `.py` → `.c` → `.pyd` (native Windows DLLs)
- Nearly impossible to reverse-engineer
- Requires MSVC Build Tools (C compiler)
- Some Python patterns may need adjustment

```bash
pip install cython
# Compile sdc_engine package to .pyd, then bundle with PyInstaller
```

### PyInstaller `--key` (Removed)

> The `--key` AES bytecode encryption flag was **removed in PyInstaller 6.0** (see [PR #6999](https://github.com/pyinstaller/pyinstaller/pull/6999)). It is no longer available. Use PyArmor or Cython instead.

## Troubleshooting

- **Nuitka crashes on Python 3.13**: Use PyInstaller instead (`python build_pyinstaller.py`)
- **"Missing DLL" errors**: Ensure Visual C++ Redistributable is installed on the client machine
- **Port 80 in use**: Set `BOKEH_PORT=5006` in the batch file
- **Port 5006 in use** (dev server): Kill the old process — `Get-Process -Id (Get-NetTCPConnection -LocalPort 5006).OwningProcess | Stop-Process -Force`
- **Blank page**: Check Keygen credentials are set correctly
- **Build fails on imports**: Run `python -c "from sdc_engine.main_local import start"` to verify all imports resolve before building
- **AI features not working**: Ensure `CEREBRAS_API_KEY` env var is set (only on `sdc-ai` branch)
