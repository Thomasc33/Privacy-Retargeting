#!/usr/bin/env bash
#
# Privacy-centric Motion Retargeting (PMR) — Interactive Setup Script
#
# Downloads NTU RGB+D 60/120 skeleton data, sets up a Python 3.11 venv,
# installs all dependencies, and validates the environment.
#
set -euo pipefail

# ─── Colors & formatting ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_VERSION="3.11"
SKELETON_DIR="$PROJECT_DIR/NTU/SGN/raw_skeletons"

# Google Drive file IDs
NTU60_FILE_ID="1CUZnBtYwifVXS21yVg62T-vrPVayso5H"
NTU120_FILE_ID="1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w"

step_num=0
total_steps=5

step() {
    step_num=$((step_num + 1))
    echo ""
    echo -e "${BOLD}${BLUE}[$step_num/$total_steps]${NC} ${BOLD}$1${NC}"
    echo -e "${DIM}$(printf '%.0s─' {1..60})${NC}"
}

info()    { echo -e "  ${CYAN}▸${NC} $1"; }
success() { echo -e "  ${GREEN}✔${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()    { echo -e "  ${RED}✖${NC} $1"; }
ask()     { echo -en "  ${YELLOW}?${NC} $1"; }

# ─── Header ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   Privacy-centric Motion Retargeting (PMR) Setup    ║${NC}"
echo -e "${BOLD}${CYAN}║   ICCV 2025                                        ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${DIM}This script will:${NC}"
echo -e "${DIM}  1. Install Python $PYTHON_VERSION via pyenv (if needed)${NC}"
echo -e "${DIM}  2. Create a virtual environment${NC}"
echo -e "${DIM}  3. Install all Python dependencies${NC}"
echo -e "${DIM}  4. Download and extract NTU RGB+D skeleton data${NC}"
echo -e "${DIM}  5. Validate the environment${NC}"
echo ""

ask "Continue? [Y/n] "
read -r response
if [[ "$response" =~ ^[Nn] ]]; then
    echo "Aborted."
    exit 0
fi

# ─── Step 1: Python version ─────────────────────────────────────────────────
step "Checking Python $PYTHON_VERSION"

PYTHON_BIN=""

# Check if pyenv has the right version
if command -v pyenv &>/dev/null; then
    info "pyenv detected"

    # Check if a compatible 3.11.x is already installed
    INSTALLED_311=$(pyenv versions --bare 2>/dev/null | grep "^3\.11\." | sort -V | tail -1 || true)

    if [[ -n "$INSTALLED_311" ]]; then
        success "Python $INSTALLED_311 already installed via pyenv"
        PYTHON_BIN="$(pyenv prefix "$INSTALLED_311")/bin/python3"
    else
        info "Python 3.11 not found in pyenv. Installing..."
        LATEST_311=$(pyenv install --list 2>/dev/null | grep '^\s*3\.11\.' | tail -1 | tr -d ' ')
        info "Installing Python $LATEST_311 (this may take a few minutes)..."
        pyenv install "$LATEST_311"
        PYTHON_BIN="$(pyenv prefix "$LATEST_311")/bin/python3"
        success "Python $LATEST_311 installed"
    fi
else
    # No pyenv — check system Python
    for candidate in python3.11 python3; do
        if command -v "$candidate" &>/dev/null; then
            ver=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
            if [[ "$ver" == "3.11" ]]; then
                PYTHON_BIN="$(command -v "$candidate")"
                success "Found system Python 3.11 at $PYTHON_BIN"
                break
            fi
        fi
    done

    if [[ -z "$PYTHON_BIN" ]]; then
        fail "Python 3.11 not found and pyenv is not installed."
        echo ""
        echo -e "  ${YELLOW}PyTorch requires Python 3.8–3.12. Your system Python is $(python3 --version 2>&1).${NC}"
        echo -e "  ${YELLOW}Install pyenv first:${NC}"
        echo -e "    ${DIM}curl https://pyenv.run | bash${NC}"
        echo -e "  ${YELLOW}Then re-run this script.${NC}"
        echo ""
        ask "Try to continue with system Python anyway? (may fail) [y/N] "
        read -r response
        if [[ "$response" =~ ^[Yy] ]]; then
            PYTHON_BIN="$(command -v python3)"
            warn "Using $PYTHON_BIN ($($PYTHON_BIN --version 2>&1)) — PyTorch may not install"
        else
            exit 1
        fi
    fi
fi

info "Using: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# ─── Step 2: Virtual environment ────────────────────────────────────────────
step "Setting up virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    EXISTING_VER=$("$VENV_DIR/bin/python3" --version 2>&1 || echo "unknown")
    warn "Existing venv found: $EXISTING_VER"
    ask "Delete and recreate? [Y/n] "
    read -r response
    if [[ ! "$response" =~ ^[Nn] ]]; then
        rm -rf "$VENV_DIR"
        info "Removed old venv"
    else
        info "Keeping existing venv"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv with $($PYTHON_BIN --version 2>&1)..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
info "Activated venv ($(python3 --version 2>&1))"

# ─── Step 3: Install dependencies ───────────────────────────────────────────
step "Installing Python dependencies"

info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing gdown (needed for data download)..."
pip install 'gdown>=5.1.0' --quiet

info "Installing from requirements.txt..."
pip install -r "$PROJECT_DIR/requirements.txt" 2>&1 | tail -5

success "All dependencies installed"

# Verify critical packages
echo ""
info "Verifying critical packages:"
python3 -c "
import torch, numpy, gdown, click, h5py, sklearn
print(f'  torch     {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  numpy     {numpy.__version__}')
print(f'  gdown     {gdown.__version__}')
print(f'  click     {click.__version__}')
print(f'  h5py      {h5py.__version__}')
print(f'  sklearn   {sklearn.__version__}')
"

# ─── Step 4: Download and extract NTU skeleton data ─────────────────────────
step "NTU RGB+D skeleton data"

DOWNLOAD_DIR="$PROJECT_DIR/NTU/SGN"
mkdir -p "$DOWNLOAD_DIR"

NTU60_ZIP="$DOWNLOAD_DIR/nturgbd_skeletons_s001_to_s017.zip"
NTU120_ZIP="$DOWNLOAD_DIR/nturgbd_skeletons_s018_to_s032.zip"

# ── Check existing skeleton files ──
existing_skeletons=0
ntu60_extracted=false
ntu120_extracted=false
if [[ -d "$SKELETON_DIR" ]]; then
    existing_skeletons=$(find "$SKELETON_DIR" -name "*.skeleton" -type f 2>/dev/null | wc -l | tr -d ' ')
fi

if (( existing_skeletons > 0 )); then
    # Sample files to determine which setups are present
    setup_range=$(find "$SKELETON_DIR" -name "*.skeleton" -type f 2>/dev/null | head -5000 | \
        sed 's|.*/S\([0-9]*\).*|\1|' | sort -un)
    max_setup=$(echo "$setup_range" | tail -1)
    min_setup=$(echo "$setup_range" | head -1)

    if (( max_setup >= 18 )); then
        ntu120_extracted=true
    fi
    if (( min_setup <= 17 )); then
        ntu60_extracted=true
    fi
fi

# ── Report current state and ask what to do ──
echo ""
if (( existing_skeletons > 0 )); then
    info "Found ${BOLD}${existing_skeletons}${NC} existing .skeleton files in raw_skeletons/"
    $ntu60_extracted  && info "NTU-60 data (S001–S017): ${GREEN}present${NC}"
    $ntu60_extracted  || info "NTU-60 data (S001–S017): ${YELLOW}not found${NC}"
    $ntu120_extracted && info "NTU-120 data (S018–S032): ${GREEN}present${NC}"
    $ntu120_extracted || info "NTU-120 data (S018–S032): ${YELLOW}not found${NC}"

    ntu60_zip_exists=false
    ntu120_zip_exists=false
    [[ -f "$NTU60_ZIP" ]]  && ntu60_zip_exists=true
    [[ -f "$NTU120_ZIP" ]] && ntu120_zip_exists=true
    $ntu60_zip_exists  && info "NTU-60 zip: ${GREEN}downloaded${NC} ($(du -sh "$NTU60_ZIP" | cut -f1))"
    $ntu120_zip_exists && info "NTU-120 zip: ${GREEN}downloaded${NC} ($(du -sh "$NTU120_ZIP" | cut -f1))"

    echo ""
    echo -e "  ${BOLD}Options:${NC}"
    echo -e "    ${CYAN}1${NC}) Skip — keep existing data as-is"
    echo -e "    ${CYAN}2${NC}) Repair — re-download and re-extract only missing datasets"
    echo -e "    ${CYAN}3${NC}) Re-download NTU-60 only (fresh)"
    echo -e "    ${CYAN}4${NC}) Re-download NTU-120 only (fresh)"
    echo -e "    ${CYAN}5${NC}) Re-download everything (fresh)"
    echo ""
    ask "Choice? [1/2/3/4/5] (default: 1) "
    read -r data_choice
    data_choice="${data_choice:-1}"

    DOWNLOAD_NTU60=false
    DOWNLOAD_NTU120=false
    EXTRACT_NTU60=false
    EXTRACT_NTU120=false

    case "$data_choice" in
        1)
            info "Keeping existing data"
            ;;
        2)
            # Repair: download/extract only what's missing
            if ! $ntu60_extracted; then
                DOWNLOAD_NTU60=true
                EXTRACT_NTU60=true
            else
                info "NTU-60 skeletons already extracted — skipping"
            fi
            if ! $ntu120_extracted; then
                DOWNLOAD_NTU120=true
                EXTRACT_NTU120=true
            else
                info "NTU-120 skeletons already extracted — skipping"
            fi
            # If zip exists but not extracted, just extract
            if ! $ntu60_extracted && [[ -f "$NTU60_ZIP" ]]; then
                DOWNLOAD_NTU60=false
                info "NTU-60 zip already downloaded — will extract"
            fi
            if ! $ntu120_extracted && [[ -f "$NTU120_ZIP" ]]; then
                DOWNLOAD_NTU120=false
                info "NTU-120 zip already downloaded — will extract"
            fi
            ;;
        3) DOWNLOAD_NTU60=true;  EXTRACT_NTU60=true ;;
        4) DOWNLOAD_NTU120=true; EXTRACT_NTU120=true ;;
        5) DOWNLOAD_NTU60=true;  EXTRACT_NTU60=true;
           DOWNLOAD_NTU120=true; EXTRACT_NTU120=true ;;
        *)
            info "Invalid choice, keeping existing data"
            ;;
    esac
else
    # No existing data — ask which datasets to download
    [[ -f "$NTU60_ZIP" ]]  && info "NTU-60 zip found: $(du -sh "$NTU60_ZIP" | cut -f1) — will extract"
    [[ -f "$NTU120_ZIP" ]] && info "NTU-120 zip found: $(du -sh "$NTU120_ZIP" | cut -f1) — will extract"

    echo ""
    echo -e "  ${BOLD}Available datasets:${NC}"
    echo -e "    ${CYAN}1${NC}) NTU RGB+D 60  (S001–S017, ~5.8 GB)"
    echo -e "    ${CYAN}2${NC}) NTU RGB+D 120 (S018–S032, ~5.5 GB)"
    echo -e "    ${CYAN}3${NC}) Both (required for full NTU-120 experiments)"
    echo ""
    ask "Which datasets to download? [1/2/3] (default: 3) "
    read -r dataset_choice
    dataset_choice="${dataset_choice:-3}"

    DOWNLOAD_NTU60=false
    DOWNLOAD_NTU120=false
    EXTRACT_NTU60=false
    EXTRACT_NTU120=false

    case "$dataset_choice" in
        1) DOWNLOAD_NTU60=true;  EXTRACT_NTU60=true ;;
        2) DOWNLOAD_NTU120=true; EXTRACT_NTU120=true ;;
        *) DOWNLOAD_NTU60=true;  EXTRACT_NTU60=true;
           DOWNLOAD_NTU120=true; EXTRACT_NTU120=true ;;
    esac

    # Skip download if zip already exists
    if $DOWNLOAD_NTU60 && [[ -f "$NTU60_ZIP" ]]; then
        info "NTU-60 zip already downloaded — skipping download"
        DOWNLOAD_NTU60=false
    fi
    if $DOWNLOAD_NTU120 && [[ -f "$NTU120_ZIP" ]]; then
        info "NTU-120 zip already downloaded — skipping download"
        DOWNLOAD_NTU120=false
    fi
fi

# ── Download function ──
download_from_gdrive() {
    local file_id="$1"
    local output="$2"
    local label="$3"

    info "Downloading $label from Google Drive..."
    info "File ID: $file_id"
    python3 -c "
import gdown
import sys

url = 'https://drive.google.com/uc?id=$file_id'
output = '$output'
try:
    gdown.download(url, output, quiet=False, fuzzy=True)
except Exception as e:
    # Try folder download if direct download fails
    try:
        gdown.download(id='$file_id', output=output, quiet=False, fuzzy=True)
    except Exception as e2:
        print(f'Download failed: {e2}', file=sys.stderr)
        sys.exit(1)
"
    if [[ -f "$output" ]]; then
        local size
        size=$(du -sh "$output" | cut -f1)
        success "$label downloaded ($size)"
    else
        fail "Download of $label failed"
        return 1
    fi
}

# ── Extract function ──
extract_skeletons() {
    local zip_file="$1"
    local label="$2"

    if [[ ! -f "$zip_file" ]]; then
        fail "Zip not found: $(basename "$zip_file") — cannot extract $label"
        return 1
    fi

    info "Extracting $label..."
    local size
    size=$(du -sh "$zip_file" | cut -f1)
    info "Archive size: $size"

    mkdir -p "$SKELETON_DIR"

    python3 << PYEOF
import zipfile
import os
import shutil
import sys

zip_path = "$zip_file"
dest_dir = "$SKELETON_DIR"

print(f"  Inspecting zip structure...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()

    # Determine structure: files in root vs in a subdirectory
    skeleton_files = [n for n in names if n.endswith('.skeleton')]
    dirs_in_root = set()
    files_in_root = 0

    for n in names:
        parts = n.split('/')
        if len(parts) == 1 and not n.endswith('/'):
            files_in_root += 1
        elif len(parts) >= 2:
            dirs_in_root.add(parts[0])

    has_subdir = len(dirs_in_root) == 1 and files_in_root == 0
    subdir_name = list(dirs_in_root)[0] if has_subdir else None

    if has_subdir:
        print(f"  Structure: files inside '{subdir_name}/' directory")
    else:
        print(f"  Structure: files directly in zip root")

    print(f"  Found {len(skeleton_files)} .skeleton files in archive")
    print(f"  Extracting to {dest_dir}...")

    # Extract all files
    extracted = 0
    for member in zf.infolist():
        if member.is_dir():
            continue

        # Get just the filename (strip any leading directory)
        filename = os.path.basename(member.filename)
        if not filename:
            continue

        # Extract to destination
        target = os.path.join(dest_dir, filename)
        with zf.open(member) as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        extracted += 1

        if extracted % 5000 == 0:
            print(f"  Extracted {extracted}/{len(skeleton_files)} files...")

    print(f"  Extracted {extracted} files total")

PYEOF

    # Count skeleton files
    local count
    count=$(find "$SKELETON_DIR" -name "*.skeleton" -type f 2>/dev/null | wc -l | tr -d ' ')
    success "$label extracted — $count total .skeleton files in raw_skeletons/"

    # Delete the zip
    info "Removing zip archive to save disk space..."
    rm -f "$zip_file"
    success "Deleted $(basename "$zip_file")"
}

# ── Execute downloads ──
if $DOWNLOAD_NTU60; then
    download_from_gdrive "$NTU60_FILE_ID" "$NTU60_ZIP" "NTU RGB+D 60"
fi
if $DOWNLOAD_NTU120; then
    download_from_gdrive "$NTU120_FILE_ID" "$NTU120_ZIP" "NTU RGB+D 120"
fi

# ── Execute extractions ──
if $EXTRACT_NTU60 && [[ -f "$NTU60_ZIP" ]]; then
    extract_skeletons "$NTU60_ZIP" "NTU RGB+D 60"
fi
if $EXTRACT_NTU120 && [[ -f "$NTU120_ZIP" ]]; then
    extract_skeletons "$NTU120_ZIP" "NTU RGB+D 120"
fi

# ─── Step 6: Environment validation ─────────────────────────────────────────
step "Validating environment"

python3 << 'PYEOF'
import sys
import os

project_dir = os.environ.get("PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
# Use the script's directory context
project_dir = os.path.abspath(".")

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'

errors = []
warnings = []

def check(label, ok, detail=""):
    if ok:
        print(f"  {GREEN}✔{NC} {label}" + (f" {detail}" if detail else ""))
    else:
        print(f"  {RED}✖{NC} {label}" + (f" {detail}" if detail else ""))
        errors.append(label)

def warn_check(label, ok, detail=""):
    if ok:
        print(f"  {GREEN}✔{NC} {label}" + (f" {detail}" if detail else ""))
    else:
        print(f"  {YELLOW}⚠{NC} {label}" + (f" {detail}" if detail else ""))
        warnings.append(label)

print(f"\n  {CYAN}── Python Environment ──{NC}")

# Python version
ver = sys.version_info
check(f"Python version: {ver.major}.{ver.minor}.{ver.micro}",
      ver.major == 3 and 8 <= ver.minor <= 12)

# Critical packages
packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn',
    'click': 'Click',
    'gdown': 'gdown',
    'h5py': 'h5py',
    'tqdm': 'tqdm',
    'matplotlib': 'Matplotlib',
    'cv2': 'OpenCV',
    'seaborn': 'Seaborn',
}

for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        ver_str = getattr(mod, '__version__', 'unknown')
        check(f"{name}: {ver_str}", True)
    except ImportError:
        check(f"{name}: NOT INSTALLED", False)

# CUDA
import torch
if torch.cuda.is_available():
    check(f"CUDA available: {torch.cuda.get_device_name(0)}", True)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    check("MPS (Apple Silicon) available", True)
else:
    warn_check("No GPU acceleration detected (CPU only)", False,
               "— training will be slow")

print(f"\n  {CYAN}── Project Structure ──{NC}")

# Check key directories and files
checks = [
    ("models/pmr.py", "PMR model"),
    ("models/classifiers.py", "Classifiers"),
    ("configs/default_config.py", "Configuration"),
    ("cli.py", "CLI"),
    ("SGN/model.py", "SGN evaluation model"),
    ("NTU/SGN/get_raw_skes_data.py", "Skeleton data extractor"),
    ("NTU/SGN/get_raw_denoised_data.py", "Denoising script"),
    ("NTU/SGN/seq_transformation.py", "Sequence transformation"),
    ("NTU/SGN/statistics/skes_available_name.txt", "Skeleton name list"),
    ("requirements.txt", "Requirements file"),
]

for path, name in checks:
    full_path = os.path.join(project_dir, path)
    check(f"{name}: {path}", os.path.exists(full_path))

# Check pretrained models
pretrained_dir = os.path.join(project_dir, "pretrained")
if os.path.isdir(pretrained_dir):
    pt_files = [f for f in os.listdir(pretrained_dir) if f.endswith('.pt')]
    check(f"Pretrained checkpoints: {len(pt_files)} models", len(pt_files) > 0,
          f"({', '.join(sorted(pt_files))})")
else:
    warn_check("Pretrained checkpoints directory", False)

print(f"\n  {CYAN}── Skeleton Data ──{NC}")

# Check skeleton files
skel_dir = os.path.join(project_dir, "NTU", "SGN", "raw_skeletons")
if os.path.isdir(skel_dir):
    skel_files = [f for f in os.listdir(skel_dir) if f.endswith('.skeleton')]
    count = len(skel_files)

    # NTU 60 has ~56,880 files (S001-S017)
    # NTU 120 has ~114,480 files total (S001-S032)
    if count >= 100000:
        check(f"Skeleton files: {count:,} (NTU RGB+D 120 — full dataset)", True)
    elif count >= 50000:
        check(f"Skeleton files: {count:,} (NTU RGB+D 60)", True)
        warn_check("NTU-120 data not complete", False,
                    f"— need ~114k files for full NTU-120, have {count:,}")
    elif count > 0:
        warn_check(f"Skeleton files: {count:,}", True,
                    "— partial download, may need to re-download")
    else:
        check("Skeleton files: 0", False, "— download failed or not yet run")

    # Check for setup range
    if count > 0:
        setups = set()
        for f in skel_files[:1000]:  # sample first 1000
            if len(f) >= 8:
                try:
                    setup = int(f[1:4])
                    setups.add(setup)
                except ValueError:
                    pass
        if setups:
            min_s, max_s = min(setups), max(setups)
            check(f"Setup range: S{min_s:03d}–S{max_s:03d}", True)
else:
    check("Skeleton data directory", False, "— NTU/SGN/raw_skeletons/ not found")

# Check preprocessed data
for pkl_name, label in [("X_full.pkl", "NTU-60 preprocessed"),
                         ("X_full_120.pkl", "NTU-120 preprocessed")]:
    pkl_path = os.path.join(project_dir, "NTU", "SGN", pkl_name)
    if os.path.exists(pkl_path):
        size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        check(f"{label}: {pkl_name} ({size_mb:.0f} MB)", True)
    else:
        warn_check(f"{label}: {pkl_name}", False, "— run preprocessing pipeline to generate")

# Summary
print()
if errors:
    print(f"  {RED}{'─' * 50}{NC}")
    print(f"  {RED}✖ {len(errors)} error(s) found — some features may not work{NC}")
    for e in errors:
        print(f"    {RED}•{NC} {e}")
else:
    print(f"  {GREEN}{'─' * 50}{NC}")
    print(f"  {GREEN}✔ All checks passed!{NC}")

if warnings:
    print(f"  {YELLOW}{len(warnings)} warning(s):{NC}")
    for w in warnings:
        print(f"    {YELLOW}•{NC} {w}")

PYEOF

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                  Setup Complete!                     ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo -e "  ${CYAN}1.${NC} Activate the environment:"
echo -e "     ${DIM}source venv/bin/activate${NC}"
echo ""
echo -e "  ${CYAN}2.${NC} Preprocess skeleton data (generates X_full.pkl):"
echo -e "     ${DIM}cd NTU/SGN${NC}"
echo -e "     ${DIM}python get_raw_skes_data.py${NC}"
echo -e "     ${DIM}python get_raw_denoised_data.py${NC}"
echo -e "     ${DIM}python seq_transformation.py${NC}"
echo -e "     ${DIM}cd ../..${NC}"
echo ""
echo -e "  ${CYAN}3.${NC} Train the model:"
echo -e "     ${DIM}python cli.py train --dataset ntu60 --model pmr --device cuda:0${NC}"
echo ""
echo -e "  ${CYAN}4.${NC} Or use pretrained models:"
echo -e "     ${DIM}python cli.py evaluate --model-path pretrained/PMR.pt --dataset ntu60${NC}"
echo ""
