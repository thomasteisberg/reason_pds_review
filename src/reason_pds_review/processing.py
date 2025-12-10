"""Processing functions for REASON data."""

import numpy as np
import xarray as xr
from typing import Union


def apply_stacking(data: Union[np.ndarray, xr.DataArray],
                   stack_factor: int,
                   axis: int = 0) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply incoherent averaging (stacking) to reduce noise.

    Stacking averages multiple consecutive pulses together to improve
    signal-to-noise ratio at the cost of reduced along-track resolution.
    This is also known as incoherent integration or multi-look averaging.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Input data to stack. For radar data, typically has shape
        (slow_time, fast_time) with complex values.
    stack_factor : int
        Number of consecutive samples to average together along the
        stacking axis. Must be >= 1. If 1, returns data unchanged.
    axis : int, optional
        Axis along which to perform stacking. Default is 0 (slow_time).

    Returns
    -------
    np.ndarray or xr.DataArray
        Stacked data with reduced dimension along stacking axis.
        Output shape along axis is input_size // stack_factor.
        If input is xr.DataArray, coordinates are updated accordingly.

    Examples
    --------
    >>> # Stack complex radar data 10x along slow time
    >>> complex_data = hf_science['complex'].values  # shape: (9698, 964)
    >>> stacked = apply_stacking(complex_data, stack_factor=10)
    >>> stacked.shape  # (969, 964)

    >>> # Stack xarray DataArray preserving coordinates
    >>> stacked_da = apply_stacking(hf_science['complex'], stack_factor=10)
    >>> stacked_da.coords['slow_time']  # Updated to [0, 1, 2, ..., 968]

    Notes
    -----
    - Partial stacks at the end are discarded (uses floor division)
    - For complex data, averaging is done on complex values (coherent averaging),
      but typically applied to already-pulse-compressed data (incoherent)
    - Common stacking factors: HF 10x, VHF 20x
    """
    if stack_factor <= 1:
        return data

    is_dataarray = isinstance(data, xr.DataArray)

    # Get numpy array
    arr = data.values if is_dataarray else data

    # Calculate output size
    input_size = arr.shape[axis]
    output_size = input_size // stack_factor

    # Create index slices for reshaping
    # Move stacking axis to front, reshape to (output_size, stack_factor, ...)
    arr = np.moveaxis(arr, axis, 0)
    original_shape = arr.shape

    # Reshape: (input_size, ...) -> (output_size, stack_factor, ...)
    new_shape = (output_size, stack_factor) + original_shape[1:]
    arr_reshaped = arr[:output_size * stack_factor].reshape(new_shape)

    # Average along stack_factor dimension (axis 1)
    arr_stacked = np.mean(arr_reshaped, axis=1)

    # Move axis back to original position
    arr_stacked = np.moveaxis(arr_stacked, 0, axis)

    # If input was DataArray, return DataArray with updated coordinates
    if is_dataarray:
        # Get dimension name for the stacking axis
        dim_name = data.dims[axis]

        # Create new coordinates
        new_coords = {}
        for coord_name, coord_data in data.coords.items():
            if dim_name in coord_data.dims:
                # Update coordinate for stacked dimension
                if coord_name == dim_name:
                    # For the main coordinate, just use indices
                    new_coords[coord_name] = np.arange(output_size)
                else:
                    # For other coordinates, take first value of each stack
                    coord_axis = coord_data.dims.index(dim_name)
                    new_coords[coord_name] = np.take(coord_data.values,
                                                     np.arange(0, output_size * stack_factor, stack_factor),
                                                     axis=coord_axis)
            else:
                # Coordinate doesn't depend on stacked dimension
                new_coords[coord_name] = coord_data

        # Create new DataArray
        result = xr.DataArray(
            arr_stacked,
            coords=new_coords,
            dims=data.dims,
            attrs=data.attrs.copy()
        )

        # Add stacking info to attributes
        result.attrs['stacking_factor'] = stack_factor
        result.attrs['stacking_axis'] = dim_name

        return result
    else:
        return arr_stacked


def calculate_amplitude_db(complex_data: Union[np.ndarray, xr.DataArray],
                          floor_value: float = 1e-10) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate amplitude in decibels from complex radar data.

    Parameters
    ----------
    complex_data : np.ndarray or xr.DataArray
        Complex-valued radar data (I + jQ)
    floor_value : float, optional
        Minimum value to avoid log(0). Default is 1e-10.

    Returns
    -------
    np.ndarray or xr.DataArray
        Amplitude in dB scale: 20*log10(|complex_data| + floor_value)

    Examples
    --------
    >>> complex_iq = hf_science['complex']
    >>> amp_db = calculate_amplitude_db(complex_iq)
    >>> amp_db.plot()  # Radargram visualization

    Notes
    -----
    Uses 20*log10 for power-in-amplitude (voltage) conversion.
    For power values, use 10*log10 instead.
    """
    is_dataarray = isinstance(complex_data, xr.DataArray)

    # Get numpy array
    arr = complex_data.values if is_dataarray else complex_data

    # Calculate amplitude in dB
    amplitude = np.abs(arr)
    amplitude_db = 20 * np.log10(amplitude + floor_value)

    if is_dataarray:
        result = xr.DataArray(
            amplitude_db,
            coords=complex_data.coords,
            dims=complex_data.dims,
            attrs=complex_data.attrs.copy()
        )
        result.attrs['description'] = 'Amplitude in dB scale'
        result.attrs['units'] = 'dB'
        return result
    else:
        return amplitude_db
