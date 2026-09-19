# Build and installation

## Supported environment

EIGCM is supported on 64-bit Linux and Ubuntu under WSL2. Python 3.8 or newer
is required for the scheduler. The scheduler itself uses only the Python
standard library.

## Ubuntu dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git perl \
  libx11-dev libxext-dev libgl1-mesa-dev libtiff-dev
```

`BUILD_HEADLESS=ON` avoids GUI programs. Omit it if the complete Radiance GUI
tool set is required.

## Compile

```bash
git clone https://github.com/a200626190111/EIGCM.git
cd EIGCM
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_HEADLESS=ON
cmake --build build -j"$(nproc)"
```

For a development build limited to the EIGCM executables:

```bash
cmake --build build --target \
  dcglare2 bsdf2reinhart directlobecontrast specularcontrast ttsuncontrast \
  -j"$(nproc)"
export PATH="$PWD/build/bin:$PATH"
```

## Test

```bash
python3 -m unittest discover -s tests -v

for case in cases/C*/eigcm.cfg; do
  python3 eigcm_scheduler.py inspect "$case" >/dev/null
  python3 eigcm_scheduler.py plan "$case" --summary >/dev/null
done
```

## Install

Build the complete selected Radiance configuration before installing:

```bash
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/.local/eigcm"

export PATH="$HOME/.local/eigcm/bin:$PATH"
export RAYPATH="$HOME/.local/eigcm/lib:."
```

The scheduler is installed as `eigcm`. Its template is installed under
`share/eigcm`, and the documentation under `share/doc/eigcm`.

Verify the installation:

```bash
eigcm --version
eigcm init my_case.cfg
dcglare2 --help
directlobecontrast --help
specularcontrast --help
ttsuncontrast --help
bsdf2reinhart --help
```

## WSL2

Clone the repository inside the Linux filesystem for the best compile and I/O
performance. A repository on a Windows drive also works, for example:

```bash
cd /mnt/e/path/to/EIGCM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_HEADLESS=ON
cmake --build build -j"$(nproc)"
```

Run `eigcm_scheduler.py` from WSL when using relative case paths. Paths in a
configuration file are resolved relative to that file.
