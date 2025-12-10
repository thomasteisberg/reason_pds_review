"""Module for loading REASON partially processed data into xarray DataTrees."""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional
import re
from lxml import etree


# Sample rate lookup based on channel type (from REASON SIS)
SAMPLE_RATES = {
    'HF': 1.2e6,      # 1.2 MHz
    'FDVHF': 12e6,    # 12 MHz
    'VHF': 12e6       # 12 MHz (for NEGX and POSX)
}


def _parse_pds4_label(xml_path: Path) -> Optional[Dict]:
    """
    Parse PDS4 XML label to extract channel configuration.

    Args:
        xml_path: Path to XML label file

    Returns:
        Dictionary with channel configuration or None if parsing fails
    """
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        # Define XML namespaces
        ns = {
            'pds': 'http://pds.nasa.gov/pds4/pds/v1'
        }

        # Extract array metadata
        array = root.find('.//pds:Array_3D', ns)
        if array is None:
            return None

        # Get channel name from Array_3D name
        channel_name_elem = array.find('pds:name', ns)
        if channel_name_elem is None:
            return None
        xml_channel_name = channel_name_elem.text.strip()

        # Get description
        desc_elem = array.find('pds:description', ns)
        description = desc_elem.text.strip() if desc_elem is not None else ''

        # Get data type
        data_type_elem = array.find('.//pds:Element_Array/pds:data_type', ns)
        if data_type_elem is None:
            return None
        pds_dtype = data_type_elem.text.strip()

        # Convert PDS4 data type to numpy dtype
        dtype_map = {
            'IEEE754LSBSingle': '<f4',  # Little-endian float32
            'IEEE754MSBSingle': '>f4',  # Big-endian float32
            'SignedByte': 'i1',         # int8
        }
        numpy_dtype = dtype_map.get(pds_dtype)
        if numpy_dtype is None:
            print(f"Warning: Unknown data type '{pds_dtype}' in {xml_path.name}")
            return None

        # Get axis dimensions
        axes = array.findall('.//pds:Axis_Array', ns)
        dimensions = {}
        for axis in axes:
            axis_name = axis.find('pds:axis_name', ns).text.strip()
            elements = int(axis.find('pds:elements', ns).text.strip())
            dimensions[axis_name] = elements

        n_slow = dimensions.get('Slow Time', 0)
        n_fast = dimensions.get('Fast Time', 0)

        # Determine sample rate and frequency from channel name
        if xml_channel_name == 'HF':
            sample_rate = SAMPLE_RATES['HF']
            frequency = '9 MHz'
            channel_name = 'HF'
        elif xml_channel_name == 'FDVHF':
            sample_rate = SAMPLE_RATES['FDVHF']
            frequency = '60 MHz'
            channel_name = 'VHF_FULL'
        else:
            # VHF_NEGX or VHF_POSX - need to determine from filename
            sample_rate = SAMPLE_RATES['VHF']
            frequency = '60 MHz'
            # Will be set based on filename
            channel_name = None

        return {
            'xml_channel_name': xml_channel_name,
            'name': channel_name,
            'frequency': frequency,
            'n_slow': n_slow,
            'n_fast': n_fast,
            'dtype': numpy_dtype,
            'sample_rate': sample_rate,
            'description': description
        }

    except Exception as e:
        print(f"Warning: Failed to parse XML {xml_path.name}: {e}")
        return None


def _parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse REASON filename to extract metadata.

    Expected format: REA{flyby}_{timestamp}_{channel}_{product}.{ext}
    Example: REA000MGA_2025060T1736_09TCOMBSCI_PPD020.BIN

    Returns:
        Dictionary with parsed components or None if pattern doesn't match
    """
    pattern = r'REA(\w+)_(\w+)_(\w+)_(PPD\d+)\.(BIN|TAB)'
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return {
            'flyby': match.group(1),
            'timestamp': match.group(2),
            'channel': match.group(3),
            'product': match.group(4),
            'extension': match.group(5).upper()
        }
    return None


def _load_science_data(file_path: Path, channel_config: Dict) -> xr.Dataset:
    """
    Load science data from .BIN file into xarray Dataset.

    Args:
        file_path: Path to .BIN file
        channel_config: Channel configuration dictionary

    Returns:
        xarray Dataset with complex I/Q data
    """
    # Read binary data
    data = np.fromfile(file_path, dtype=channel_config['dtype'])

    # Calculate number of slow time samples from file size
    n_fast = channel_config['n_fast']
    n_iq = 2
    total_samples = len(data)
    n_slow = total_samples // (n_fast * n_iq)

    # Reshape to 3D array (slow_time, fast_time, IQ)
    data = data.reshape(n_slow, n_fast, n_iq)

    # Create complex data directly
    complex_data = data[:, :, 0] + 1j * data[:, :, 1]

    # Calculate time coordinates
    sample_rate = channel_config['sample_rate']
    fast_time = np.arange(n_fast) / sample_rate * 1e6  # in microseconds

    # Create xarray Dataset with only complex field
    ds = xr.Dataset(
        data_vars={
            'complex': (['slow_time', 'fast_time'], complex_data),
        },
        coords={
            'slow_time': np.arange(n_slow),
            'fast_time': fast_time,
        },
        attrs={
            'channel_name': channel_config['name'],
            'frequency': channel_config['frequency'],
            'sample_rate_hz': sample_rate,
            'description': channel_config['description'],
            'n_slow_time': n_slow,
            'n_fast_time': n_fast,
            'fast_time_units': 'microseconds',
            'source_file': file_path.name
        }
    )

    ds['complex'].attrs['description'] = 'Complex I/Q data'

    return ds


def _load_engineering_data(file_path: Path) -> xr.Dataset:
    """
    Load engineering data from .TAB file into xarray Dataset.

    Args:
        file_path: Path to .TAB file

    Returns:
        xarray Dataset with engineering parameters
    """
    # Read TAB file (tab-delimited)
    df = pd.read_csv(file_path, sep='\t')

    # Convert to xarray Dataset with record dimension
    ds = xr.Dataset.from_dataframe(df)

    # Rename index to slow_time to match science data
    if 'index' in ds.dims:
        ds = ds.rename({'index': 'slow_time'})

    ds.attrs['source_file'] = file_path.name
    ds.attrs['description'] = 'Engineering data (chirp parameters, telemetry)'

    return ds


def load_ppdp(data_dir: str) -> xr.DataTree:
    """
    Load REASON partially processed data from a directory into an xarray DataTree.

    This function scans a partially processed data directory and loads all .BIN
    (science data) and .TAB (engineering data) files, organizing them into a
    hierarchical xarray DataTree structure.

    Args:
        data_dir: Path to partially processed data directory
                 (e.g., "urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736")

    Returns:
        xarray DataTree with structure:
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

    Example:
        >>> tree = load_ppdp("urn-nasa-pds-clipper.rea.partiallyprocessed/DATA/000MGA/2025060T1736")
        >>> # Access HF science data
        >>> hf_data = tree['HF/science'].ds
        >>> # Get complex I/Q data
        >>> complex_iq = hf_data['complex']
        >>> # Calculate amplitude in dB
        >>> amp_db = 20 * np.log10(np.abs(complex_iq) + 1e-10)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Initialize dictionary to hold datasets organized by channel
    channels_data = {}

    # Scan directory for .BIN and .TAB files
    for file_path in sorted(data_path.glob('*.BIN')):
        parsed = _parse_filename(file_path.name)
        if not parsed:
            print(f"Warning: Could not parse filename: {file_path.name}")
            continue

        channel_key = parsed['channel']

        # Load configuration from XML label file
        xml_path = file_path.with_suffix('.XML')

        if not xml_path.exists():
            print(f"Warning: XML label file not found for {file_path.name}")
            continue

        config = _parse_pds4_label(xml_path)
        if config is None:
            print(f"Warning: Failed to parse XML label for {file_path.name}")
            continue

        # Determine channel name for VHF_NEGX and VHF_POSX from filename
        if config['name'] is None:
            if 'NEGX' in channel_key.upper():
                config['name'] = 'VHF_NEGX'
                config['description'] = 'Shallow Sounding - Port (3 km)'
            elif 'POSX' in channel_key.upper():
                config['name'] = 'VHF_POSX'
                config['description'] = 'Shallow Sounding - Starboard (3 km)'
            else:
                print(f"Warning: Could not determine VHF channel type for {file_path.name}")
                continue

        channel_name = config['name']

        # Initialize channel entry if needed
        if channel_name not in channels_data:
            channels_data[channel_name] = {}

        # Load science data
        print(f"Loading {channel_name} science data from {file_path.name}...")
        channels_data[channel_name]['science'] = _load_science_data(file_path, config)

    # Load engineering files (only *ENG*.TAB files)
    for file_path in sorted(data_path.glob('*ENG*.TAB')):
        parsed = _parse_filename(file_path.name)
        if not parsed:
            continue

        channel_key = parsed['channel']

        # Determine channel name from filename
        channel_name = None
        if '09TCOMB' in channel_key.upper():
            channel_name = 'HF'
        elif '60TFULL' in channel_key.upper():
            channel_name = 'VHF_FULL'
        elif 'NEGX' in channel_key.upper():
            channel_name = 'VHF_NEGX'
        elif 'POSX' in channel_key.upper():
            channel_name = 'VHF_POSX'

        if channel_name and channel_name in channels_data:
            print(f"Loading {channel_name} engineering data from {file_path.name}...")
            channels_data[channel_name]['engineering'] = _load_engineering_data(file_path)

    # Build DataTree structure using from_dict
    tree_dict = {}

    for channel_name, datasets in channels_data.items():
        if 'science' in datasets:
            tree_dict[f'{channel_name}/science'] = datasets['science']
        if 'engineering' in datasets:
            tree_dict[f'{channel_name}/engineering'] = datasets['engineering']

    # Create root with metadata
    root_ds = xr.Dataset(attrs={
        'data_directory': str(data_path),
        'description': 'REASON Partially Processed Data'
    })
    tree_dict['/'] = root_ds

    dt = xr.DataTree.from_dict(tree_dict)

    print(f"\nLoaded {len(channels_data)} channels from {data_path.name}")

    return dt
