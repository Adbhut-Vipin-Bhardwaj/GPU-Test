# Intel GPU Test (PyTorch XPU)

The purpose of this repository is to test access to the Intel Core Ultra 7 155H integrated GPU using Python and PyTorch. 

It utilizes PyTorch's native XPU backend to run computations directly on Intel integrated graphics.

## Project Structure

- `test.py` - A simple script to verify if the Intel XPU is recognized and available to PyTorch.
- `speed_comparison.py` - A benchmark script that compares the performance of matrix multiplication between the CPU and the Intel XPU.

## Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Compatible Intel Hardware (e.g., Intel Core Ultra processors with Intel Arc Graphics)

## Installation

This project uses `uv` for fast dependency management. The `pyproject.toml` is pre-configured to download the necessary PyTorch XPU wheels directly from the official PyTorch download index.

To install dependencies and set up the virtual environment, run:

```bash
uv sync
```

*(Alternatively, you can run scripts directly using `uv run <script_name>`)*

## Usage

### 1. Verify XPU Availability

Run the `test.py` script to ensure PyTorch can detect your Intel GPU:

```bash
uv run test.py
```

**Expected output:**
```
Intel XPU is available!
```

### 2. Run Performance Benchmark

To see the performance difference between your CPU and the Intel XPU, run the speed comparison script:

```bash
uv run speed_comparison.py
```

This will perform iterative 4096x4096 matrix multiplications on both devices and report the average execution time and the resulting speedup factor of the XPU over the CPU.
