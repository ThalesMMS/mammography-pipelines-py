# Multi-Environment GPU Support Guide

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes.**

## Solution Overview

**Challenge:** The MacBook Air M4 has no discrete GPU but provides an Apple Silicon integrated GPU.

**Resolution:** Automatic device detection with fallbacks across environments:
- ✅ **MPS** (Apple Silicon GPU) — running successfully on the MacBook Air M4
- ✅ **CUDA** (NVIDIA GPU) — recommended for a desktop with an RTX 5080
- ✅ **CPU** fallback — available on every system when GPUs are absent

## Recommendation

Keep development on the MacBook Air M4. It offers:
1. Automatic MPS support with PyTorch
2. Noticeably faster execution compared with CPU-only workflows
3. A great developer experience for experimentation and tests
4. Portability across macOS and Windows/Linux machines
5. No additional hardware cost for day-to-day work

For a PC with an RTX 5080:
- Superior throughput for heavy training
- ≥16 GB VRAM for larger batches and datasets
- Ideal for final experiments and production-grade runs

## Technical Implementation

### 1. Automatic Device Detection

```python
# src/utils/device_detection.py
from src.utils.device_detection import get_optimal_device, get_device_config

device = get_optimal_device()  # Returns MPS, CUDA, or CPU
config = get_device_config()   # Optimised batch size, precision, workers
```

### 2. Environment-Specific Settings

```yaml
mps:
  batch_size: 8
  mixed_precision: false
  gpu_memory_limit: 8
  num_workers: 4

cuda:
  batch_size: 16
  mixed_precision: true
  gpu_memory_limit: 16
  num_workers: 4

cpu:
  batch_size: 4
  mixed_precision: false
  num_workers: 8
```

### 3. Usage in Code

```python
import torch
from src.utils.device_detection import get_optimal_device, get_device_config

device = get_optimal_device()
config = get_device_config()

model = ResNet50().to(device)
inputs = inputs.to(device)

batch_size = config["batch_size"]
use_amp = config["mixed_precision"]
```

## Current Status

### MacBook Air M4
- Device: MPS (Apple Silicon GPU)
- Status: ✅ Operational
- Role: Development and experimentation

### PC with RTX 5080
- Device: CUDA (NVIDIA GPU)
- Status: ✅ Ready when available
- Role: Heavy experiments and production training

## Validation Steps

```bash
source venv/bin/activate
python scripts/check_status.py
```

Expected output snippet:
```
Checking GPU...
MPS (Apple Silicon GPU) available
```

Device-specific check:

```bash
python -c "from src.utils.device_detection import print_device_status; print_device_status()"
```

```
Selected device: MPS
Batch size: 8
Mixed precision: False
```

## Next Steps

1. Continue development on the MacBook Air M4 using the integrated GPU.
2. Switch to the RTX 5080 workstation for large-scale experiments when needed.
3. Keep `device_detection.py` as the single source of truth for device configuration.

## Performance Comparison

| Environment      | Device | Batch Size | Mixed Precision | Usage            |
|------------------|--------|------------|-----------------|------------------|
| MacBook Air M4   | MPS    | 8          | No              | Development      |
| PC with RTX 5080 | CUDA   | 16         | Yes             | Production       |
| Generic machine  | CPU    | 4          | No              | Fallback         |

## Conclusion

Staying on the MacBook Air M4 for development is efficient and cost-effective. Automatic detection keeps the project portable, and the RTX 5080 workstation remains available for high-throughput training.
