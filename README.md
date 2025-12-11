# REASON PDS Review Library

This is a quick set of code I put together for a PDS review of a preview of data from the REASON instrument on Europa Clipper. The vast majority of this code was written with AI assistance and there are known issues. Feel free to take this as a starting point for anything if it's useful to you, but be aware that this was only intended for breifly reviewing the data formats and providing feedback on the documentation. The remainder of this README is entirely AI generated.

## Features

- **Easy Data Loading**: Load REASON partially processed data directories into organized xarray DataTrees
- **PDS4 Label Parsing**: Reads channel configurations directly from XML label files (no hardcoded values!)
- **Automatic Channel Detection**: Detects and loads all channels (HF, VHF NEGX, VHF POSX, VHF FULL)
- **Science & Engineering Data**: Loads both science data (.BIN files) and engineering data (.TAB files)
- **Complex I/Q Data**: Loads complex I/Q data for radar signal processing
- **xarray Integration**: Full xarray compatibility for powerful data analysis and manipulation

## Installation

```bash
# Clone or copy the library
cd reason_pds_review

# Install with uv
uv sync
```

## Quick Start

```python
from reason_pds_review import load_ppdp

# Load data from a partially processed data directory
tree = load_ppdp("path/to/DATA/000MGA/2025060T1736")

# Access HF science data
hf_science = tree['HF/science'].ds

# Get complex I/Q data
complex_iq = hf_science['complex']

# Calculate amplitude
amplitude = np.abs(complex_iq)

# Calculate amplitude in dB
amplitude_db = 20 * np.log10(amplitude + 1e-10)

# Access engineering data
hf_eng = tree['HF/engineering'].ds
chirp_start = hf_eng['Chirp_start_frequency']
```

## DataTree Structure

The library organizes data into a hierarchical xarray DataTree:

```
/
├── HF/
│   ├── science (Dataset with I/Q data)
│   └── engineering (Dataset with chirp params)
├── VHF_NEGX/
│   ├── science
│   └── engineering
├── VHF_POSX/
│   ├── science
│   └── engineering
└── VHF_FULL/
    ├── science
    └── engineering
```

### Science Dataset Variables

Each science dataset contains:
- `complex`: Complex I/Q data (I + 1j*Q)

Additional quantities can be easily computed:
```python
amplitude = np.abs(hf_science['complex'])
amplitude_db = 20 * np.log10(amplitude + 1e-10)
i_component = hf_science['complex'].real
q_component = hf_science['complex'].imag
```

### Coordinates

- `slow_time`: Pulse number (along-track)
- `fast_time`: Fast time in microseconds (range/depth)

### Attributes

Each dataset includes metadata:
- `channel_name`: Channel identifier (HF, VHF_NEGX, etc.)
- `frequency`: Center frequency
- `sample_rate_hz`: Sampling rate
- `description`: Channel description
- `n_slow_time`, `n_fast_time`: Data dimensions

## Examples

### Example 1: Basic Data Exploration

```python
from reason_pds_review import load_ppdp

# Load data
tree = load_ppdp("DATA/000MGA/2025060T1736")

# Print structure
print(tree)

# Access HF data
hf = tree['HF/science'].ds
print(f"HF data shape: {hf['complex'].shape}")
print(f"Fast time range: {hf.coords['fast_time'].min():.2f} to {hf.coords['fast_time'].max():.2f} μs")

# Calculate amplitude
amplitude_db = 20 * np.log10(np.abs(hf['complex']) + 1e-10)
```

### Example 2: Uncompressed Radargrams

See `demo_uncompressed_radargrams.py` for a complete example that:
- Loads all four REASON channels
- Applies incoherent stacking
- Generates uncompressed radargrams (no pulse compression)
- Saves publication-quality figures

Run the demo:
```bash
uv run python demo_uncompressed_radargrams.py
```

### Example 2b: Pulse-Compressed Radargrams

See `demo_radargrams.py` for full processing pipeline:
- Dwell-based processing (chirp parameters vary by dwell)
- Pulse compression (matched filter per dwell)
- Coherent stacking within each dwell
- Delay alignment within each dwell
- Concatenation of all dwells
- Geometric correction (optional)
- Generates fully processed radargrams

Run the demo:
```bash
uv run python demo_radargrams.py
```

**Note:** Processing is done separately for each dwell because chirp parameters (frequency, length) may change between dwells. Each dwell typically contains a few seconds of data (~416 pulses for HF).

### Example 3: Processing with Stacking

```python
from reason_pds_review import load_ppdp, apply_stacking, calculate_amplitude_db
import matplotlib.pyplot as plt

tree = load_ppdp("DATA/000MGA/2025060T1736")

# Get HF complex data
hf_complex = tree['HF/science'].ds['complex']

# Apply 10x stacking to reduce noise
hf_stacked = apply_stacking(hf_complex, stack_factor=10)

# Calculate amplitude in dB
amp_db = calculate_amplitude_db(hf_stacked)

# Plot radargram
plt.figure(figsize=(12, 8))
plt.imshow(amp_db.T, aspect='auto', cmap='gray')
plt.xlabel('Slow Time (Pulse Number)')
plt.ylabel('Fast Time (μs)')
plt.title('HF Radargram (10x stacked)')
plt.colorbar(label='Amplitude (dB)')
plt.show()
```

### Example 4: Working with Engineering Data

```python
from reason_pds_review import load_ppdp

tree = load_ppdp("DATA/000MGA/2025060T1736")

# Access HF engineering data
hf_eng = tree['HF/engineering'].ds

# Get chirp parameters
chirp_start = hf_eng['Chirp_start_frequency'].values  # Hz
chirp_end = hf_eng['Chirp_end_frequency'].values      # Hz
chirp_length = hf_eng['Chirp_length_ticks'].values    # ticks

# Calculate bandwidth
bandwidth = (chirp_end - chirp_start) / 1e6  # MHz
print(f"Median bandwidth: {np.median(bandwidth):.1f} MHz")
```

## Channel Configurations

The library supports four REASON channels:

| Channel | Frequency | Fast Samples | Sample Rate | Description |
|---------|-----------|--------------|-------------|-------------|
| HF | 9 MHz | 964 | 1.2 MHz | Deep Sounding (30 km) |
| VHF_NEGX | 60 MHz | 6400 | 12 MHz | Shallow Sounding - Port (3 km) |
| VHF_POSX | 60 MHz | 6400 | 12 MHz | Shallow Sounding - Starboard (3 km) |
| VHF_FULL | 60 MHz | 9008 | 12 MHz | Deep Sounding - 1-bit (30 km) |

## How It Works

The library loads channel configurations directly from PDS4 XML label files:

- **XML Label Parsing**: Reads PDS4 XML label files (`.XML`) to extract:
  - Array dimensions (slow time, fast time samples)
  - Data type (IEEE754LSBSingle, SignedByte, etc.)
  - Channel name and description
  - This ensures configurations always match the actual data files

**Benefits:**
- No hardcoded values - always uses the authoritative PDS4 metadata
- Automatically adapts to different data products
- Ensures data integrity by matching configurations to actual files
- Sample rates are determined from REASON SIS documentation (1.2 MHz for HF, 12 MHz for VHF)

## API Reference

### `load_ppdp(data_dir: str) -> xr.DataTree`

Load REASON partially processed data from a directory.

The function automatically:
- Scans for `.BIN` (science) and `.TAB` (engineering) files
- Reads corresponding `.XML` label files for metadata
- Parses array dimensions and data types from PDS4 labels
- Organizes data into a hierarchical DataTree structure

**Parameters:**
- `data_dir`: Path to partially processed data directory

**Returns:**
- `xr.DataTree`: Hierarchical data structure containing all channels

**Example:**
```python
tree = load_ppdp("urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736")
```

### `apply_stacking(data, stack_factor, axis=0)`

Apply incoherent averaging (stacking) to reduce noise.

Stacking averages multiple consecutive pulses together to improve signal-to-noise ratio at the cost of reduced along-track resolution. Also known as incoherent integration or multi-look averaging.

**Parameters:**
- `data`: Input data (numpy array or xarray DataArray). For radar data, typically shape (slow_time, fast_time) with complex values
- `stack_factor`: Number of consecutive samples to average together (must be >= 1)
- `axis`: Axis along which to perform stacking (default: 0 for slow_time)

**Returns:**
- Stacked data with reduced dimension along stacking axis (output_size = input_size // stack_factor)
- If input is xarray DataArray, coordinates are updated and stacking metadata is added to attributes

**Example:**
```python
from reason_pds_review import apply_stacking

# Stack complex radar data 10x along slow time
hf_complex = tree['HF/science'].ds['complex']
stacked = apply_stacking(hf_complex, stack_factor=10)
print(stacked.shape)  # (969, 964) instead of (9698, 964)
print(stacked.attrs['stacking_factor'])  # 10
```

### `calculate_amplitude_db(complex_data, floor_value=1e-10)`

Calculate amplitude in decibels from complex radar data.

**Parameters:**
- `complex_data`: Complex-valued radar data (I + jQ), numpy array or xarray DataArray
- `floor_value`: Minimum value to avoid log(0) (default: 1e-10)

**Returns:**
- Amplitude in dB scale: 20*log10(|complex_data| + floor_value)
- If input is xarray DataArray, returns DataArray with metadata

**Example:**
```python
from reason_pds_review import calculate_amplitude_db

complex_iq = tree['HF/science'].ds['complex']
amp_db = calculate_amplitude_db(complex_iq)
amp_db.plot()  # Visualize radargram
```

### `generate_chirp(chirp_start_freq, chirp_end_freq, chirp_length_ticks, sample_rate, window='hann')`

Generate a complex reference chirp for pulse compression.

The chirp is centered at zero frequency (complex baseband) and spans the specified bandwidth with optional windowing to suppress range sidelobes.

**Parameters:**
- `chirp_start_freq`: Starting frequency of the chirp in Hz
- `chirp_end_freq`: Ending frequency of the chirp in Hz
- `chirp_length_ticks`: Duration of the chirp in sample ticks
- `sample_rate`: Sample rate in Hz
- `window`: Window function ('hann', 'hamming', 'blackman', or None). Default: 'hann'

**Returns:**
- Complex reference chirp signal (numpy array)

**Example:**
```python
from reason_pds_review import generate_chirp

# Generate HF chirp from engineering data
eng = tree['HF/engineering'].ds
chirp = generate_chirp(
    chirp_start_freq=eng['Chirp_start_frequency'].values[0],
    chirp_end_freq=eng['Chirp_end_frequency'].values[0],
    chirp_length_ticks=int(eng['Chirp_length_ticks'].values[0]),
    sample_rate=1.2e6
)
```

### `pulse_compress(data, chirp, axis=-1)`

Apply pulse compression via matched filter (convolution with reference chirp).

**Parameters:**
- `data`: Complex radar data to compress, typically shape (slow_time, fast_time)
- `chirp`: Complex reference chirp signal from `generate_chirp()`
- `axis`: Axis along which to apply compression (default: -1, fast time axis)

**Returns:**
- Pulse-compressed complex data with same shape as input

**Example:**
```python
from reason_pds_review import generate_chirp, pulse_compress

# Generate chirp and compress data
chirp = generate_chirp(chirp_start, chirp_end, chirp_len, sample_rate)
compressed = pulse_compress(raw_data, chirp, axis=1)
```

### `align_by_delay(data, hw_rx_opening_ticks, tx_start_ticks, chirp_length_ticks, rx_window_length_ticks, raw_active_mode_length, axis=0)`

Align fast time records by rolling to account for varying delays.

In the Partially Processed Data Product, dwell delays are varied to maintain tracking of the surface. This function aligns records by computing the delay offset and rolling each fast time record accordingly.

**Parameters:**
- `data`: Complex radar data, typically shape (slow_time, fast_time)
- `hw_rx_opening_ticks`: Hardware RX opening time in ticks (slow_time,)
- `tx_start_ticks`: TX start time in ticks (slow_time,)
- `chirp_length_ticks`: Chirp length in ticks (slow_time,)
- `rx_window_length_ticks`: RX window length in ticks (slow_time,)
- `raw_active_mode_length`: Number of fast time samples (slow_time,)
- `axis`: Axis corresponding to slow time (default: 0)

**Returns:**
- Aligned radar data with same shape as input

**Example:**
```python
from reason_pds_review import align_by_delay

eng = tree['HF/engineering'].ds
aligned = align_by_delay(
    data=compressed_data,
    hw_rx_opening_ticks=eng['HW_RX_opening_ticks'].values,
    tx_start_ticks=eng['TX_start_ticks'].values,
    chirp_length_ticks=eng['Chirp_length_ticks'].values,
    rx_window_length_ticks=eng['RX_window_length_ticks'].values,
    raw_active_mode_length=eng['Raw_active_mode_length'].values
)
```

### `geometric_correction(data, altitude_km, sample_rate, axis=0)`

Apply geometric correction by aligning records to reference ellipsoid range.

Uses spacecraft altitude to compute range to reference ellipsoid and rolls each fast time record to align surface returns geometrically.

**Parameters:**
- `data`: Complex radar data, typically shape (slow_time, fast_time)
- `altitude_km`: Spacecraft altitude above target ellipsoid in km (slow_time,)
- `sample_rate`: Sample rate in Hz
- `axis`: Axis corresponding to slow time (default: 0)

**Returns:**
- Geometrically corrected radar data with same shape as input

**Example:**
```python
from reason_pds_review import geometric_correction

# Note: MED (mission engineering data) contains altitude
# altitude_km = med['SC_altitude_above_target_ellipsoid'].values
corrected = geometric_correction(
    data=aligned_data,
    altitude_km=altitude_km,
    sample_rate=1.2e6
)
```

## Requirements

- Python >= 3.13
- xarray >= 2025.12.0
- numpy >= 2.3.5
- pandas >= 2.3.3
- scipy >= 1.16.3 (for pulse compression and signal processing)
- lxml >= 6.0.2 (for XML label parsing)
- matplotlib >= 3.10.7 (for demo scripts)

