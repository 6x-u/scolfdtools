#!/usr/bin/env bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         SCOLFDTOOLS - Universal Setup Script              ║"
echo "║    Supports: Linux, macOS, Android (Termux), Windows      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

detect_platform() {
    if [ -n "$TERMUX_VERSION" ]; then
        echo "termux"
    elif [ "$(uname)" = "Linux" ]; then
        echo "linux"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "macos"
    elif [ "$(uname -s | grep -i 'MINGW\|MSYS\|CYGWIN')" ]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

install_python() {
    platform=$1
    echo "[*] Installing Python 3..."
    
    case $platform in
        termux)
            pkg update -y && pkg install -y python
            ;;
        linux)
            if command -v apt >/dev/null; then
                sudo apt update && sudo apt install -y python3 python3-pip
            elif command -v dnf >/dev/null; then
                sudo dnf install -y python3 python3-pip
            elif command -v pacman >/dev/null; then
                sudo pacman -S --noconfirm python python-pip
            fi
            ;;
        macos)
            if ! command -v brew >/dev/null; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python3
            ;;
        windows)
            echo "[!] Please install Python from https://www.python.org/downloads/"
            ;;
    esac
}

check_python() {
    if command -v python3 >/dev/null 2>&1; then
        version=$(python3 --version 2>&1 | awk '{print $2}')
        echo "[✓] Python 3 found: $version"
        return 0
    elif command -v python >/dev/null 2>&1; then
        version=$(python --version 2>&1 | awk '{print $2}')
        if [[ $version == 3.* ]]; then
            echo "[✓] Python 3 found: $version"
            return 0
        fi
    fi
    return 1
}

main() {
    platform=$(detect_platform)
    echo "[*] Platform detected: $platform"
    echo ""
    
    if ! check_python; then
        echo "[!] Python 3 not found"
        read -p "[?] Install Python 3 automatically? (y/n): " install
        if [ "$install" = "y" ] || [ "$install" = "Y" ]; then
            install_python $platform
        else
            echo "[!] Python 3 is required. Please install manually."
            exit 1
        fi
    fi
    
    echo ""
    echo "[*] Making scripts executable..."
    chmod +x scolfdtools.py eng.py xir.py 2>/dev/null || true
    
    echo ""
    echo "[✓] Setup complete!"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   HOW TO USE                               ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  Run the tool:                                             ║"
    echo "║    python3 scolfdtools.py                                  ║"
    echo "║                                                            ║"
    echo "║  Or create an alias (optional):                            ║"
    echo "║    alias scolfd='python3 $(pwd)/scolfdtools.py'            ║"
    echo "║    echo \"alias scolfd='python3 $(pwd)/scolfdtools.py'\" >> ~/.bashrc  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
}

main
