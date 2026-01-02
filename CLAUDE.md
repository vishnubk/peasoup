# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make              # Build peasoup binary (output: bin/peasoup)
make install      # Install to /usr/local/bin (configurable via INSTALL_DIR in Makefile.inc)
make clean        # Remove object files
```

**Prerequisites:** CUDA Toolkit (10.0-12.6), dedisp library (https://github.com/vishnubk/dedisp)

## Architecture Overview

Peasoup is a GPU-accelerated pulsar search pipeline written in CUDA C++. It processes radio telescope filterbank files to detect periodic signals from pulsars, outputting XML candidate files.

### Core Pipeline Flow

1. **Input**: Filterbank file → `include/data_types/filterbank.hpp`
2. **Dedispersion**: Removes frequency-dependent delays → `include/transforms/dedisperser.hpp` (wraps dedisp library)
3. **FFT**: Time-domain to frequency-domain → `include/transforms/ffter.hpp`
4. **Spectrum Formation**: Power spectrum generation → `include/transforms/spectrumformer.hpp`
5. **Red Noise Removal**: Baseline correction → `include/transforms/dereddener.hpp`
6. **Harmonic Summing**: Fold harmonics to boost SNR → `include/transforms/harmonicfolder.hpp` (incoherent) or `include/transforms/coherentharmonicfolder.hpp` (coherent)
7. **Peak Detection**: Find candidate signals → `include/transforms/peakfinder.hpp`
8. **Candidate Distillation**: Filter duplicates → `include/transforms/distiller.hpp`
9. **Output**: XML candidate file → `include/data_types/candidates.hpp`

### Search Modes

- **Acceleration Search**: Polynomial resampling for accelerated pulsars (`include/transforms/resampler.hpp`)
- **Template Bank Search**: Keplerian orbital parameter matching for binaries (`include/transforms/template_bank_reader.hpp`)

### Harmonic Summing Modes

- **Incoherent (default)**: Sums power spectrum values at harmonic frequencies. Fast but loses phase information, sensitivity drops for duty cycles < 1/nharmonics (~3% for 32 harmonics).
- **Coherent (`--coherent_harmonic_sums`)**: Preserves phase via FFT-based phase search over complex Fourier coefficients. Optimal sensitivity for ANY duty cycle, but computationally more expensive. Uses inline radix-2 FFT (2, 4, 8, 16, 32 points) per frequency bin.

### Key Source Files

- `src/pipeline_multi.cu` - Main entry point with multi-threaded DM trial distribution
- `src/kernels.cu` - GPU kernel implementations (harmonic sums, template correlation)
- `src/folding_kernels.cu` - Pulse folding kernels

### Thread Model

- `DMDispenser` class: Thread-safe distribution of DM trials across workers
- `Worker` class: GPU worker threads processing DM ranges in parallel
- Each worker owns a GPU stream for concurrent execution

### Data Types (`include/data_types/`)

- `filterbank.hpp` - Input file handling
- `timeseries.hpp` - Dedispersed time series
- `fourierseries.hpp` - FFT output representation
- `candidates.hpp` - Pulsar candidate storage with XML serialization
- `header.hpp` - Observation metadata

### Python Tools (`tools/`)

- `peasoup_tools.py` - Utilities for parsing XML output
- `peasoup_plot_cand.py` - Candidate visualization

## Important Flags

- `--fft_size` - Always specify explicitly; affects search sensitivity
- `--dm_file` - Preferred over `--dm_start/--dm_end` for DM trial control
- `--cdm` - Coherent DM value if data is coherently dedispersed
- `-K` - Template bank file for Keplerian searches
- `-p` - Progress bar (useful for long searches)
- `--coherent_harmonic_sums` - Enable coherent harmonic summing for optimal sensitivity at any duty cycle
- `--single_precision_harmonic_sums` - Use single precision for incoherent harmonic sums (faster, slightly less accurate)
