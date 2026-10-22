# Environment

Every study ran on a CPU, in one Python environment: CPython 3.12.14 with the 38 packages recorded in
`studies/1-feasibility-classification/06-Study-History/environment.json`, among them PyTorch 2.7.1 (CPU build),
NumPy 2.2.0, SciPy 1.16.0 and Matplotlib 3.10.3. Every run manifest also records the interpreter and library
versions it used. `requirements.txt` pins all of them. The environment also held Tracr and the CAE library,
installed from their repositories, which no code here imports; `requirements.txt` lists them commented out.

## A fresh install

```sh
make setup
```

creates `.venv` with uv's managed CPython 3.12.14, installs the pinned packages from PyPI and then PyTorch 2.7.1 from
PyTorch's CPU index (`https://download.pytorch.org/whl/cpu`). Installed, the environment takes about 1.3 GB, most of
it PyTorch (650 MB) and jaxlib (320 MB), which the recorded environment held although no code here uses it.

## A smaller environment

The quick check and the figures need only NumPy, SciPy and Matplotlib:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.2.0 scipy==1.16.0 matplotlib==3.10.3
```

`make test` and the study code import PyTorch.

Training runs in float64 (phase 2 and the larger confirmation) or float32 (phase 1) with deterministic operations
and one thread. In the recorded environment the saved networks are reproduced bit for bit; another CPU or library
build may differ in the last bits.
