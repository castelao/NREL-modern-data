
import logging

import h5py
import pandas as pd
import xarray as xr

module_logger = logging.getLogger(__name__)

def extract_meta(filename):
    """Extract variables embeded in 'meta'"""
    with h5py.File(filename, "r") as h5f:
        for vname in h5f["meta"].dtype.names:
            yield xr.DataArray(
                h5f["meta"][vname][:],
                name=vname,
                dims=("location",),
                attrs=dict(
                    description="Extracted from meta variable",
                ),
            )


def fix_variable(da):
    attrs = da.attrs
    encoding = da.encoding
    encoding = {}
    if "fill_value" in da.attrs:
        print(f"Fixing fill value for {da.name}, {da.attrs['fill_value']}")
        da = da.where(da != da.attrs["fill_value"])
        attrs.pop("fill_value")
        # encoding["_FillValue"] = da.attrs.pop("fill_value")

    if "scale_factor" in da.attrs:
        if "adder" in da.attrs:
            da.attrs["offset"] = da.attrs["adder"]
            da = da + da.attrs["offset"]

        else:
            print(f"Fixing scaling factor for {da.name}, {da.attrs['scale_factor']}")
            da = da / da.attrs["scale_factor"]
            # da.encoding["scale_factor"] = 1 / attrs.pop("scale_factor")

        attrs.pop("scale_factor")

    da.attrs = attrs
    # da.encoding = encoding
    return da

def fix_time(ds):
    """Fix dimension name and data type for time

    Using phony_dim_X is not informative. Instead let's call it time.

    The actual time is stored as a string. Although that is easy for a
    human to read, it is not actionable. Any operation with time would
    require first validate if that string is a valid date/time, then
    convert that to some actionable data type, such as np.datetime64.
    Another issue is the space used. This string takes 19B versus the
    8B used by np.datetime64 which has ns resolution.
    """
    assert "time_index" in ds

    # Rename the phony dimension to time
    assert len(ds["time_index"].dims) == 1
    ds = ds.rename_dims({ds["time_index"].dims[0]: "time"})

    assert ds["time_index"].dtype.kind == "U", "Expected a `time_index` of type unicode"
    ds["time"] = pd.DatetimeIndex(ds["time_index"].values)

    # Figure out which type of calendar it is
    # ds.["time"].attrs["calendar"]

    # We don't need this anymore
    module_logger.debug(f"Removing time_index variable with time as a string.")
    ds = ds.drop_vars(["time_index"])

    return ds


standard_attributes = {
    "time": {
        "long_name": "time",
        "calendar": "proleptic_gregorian",
    }
    "latitude": {
        "standard_name": "latitude",
        "units": "degree_north",
    },
    "longitude": {
        "standard_name": "longitude",
        "units": "degree_east",
    },
    "temperature": {
        "standard_name": "air_temperature",
        "units": "C",
    },
    "windspeed": {
        "standard_name": "wind_speed",
        "units": "m s-1",
    },
    "winddirection": {
        "standard_name": "wind_to_direction",
        "units": "degree",
    },
    "pressure": {
        "standard_name": "air_pressure",
        "units": "Pa",
    },
    "relativehumidity": {
        "standard_name": "relative_humidity",
        "units": 1,
    },
}

def add_attributes(ds):
    for v in ds:
        if attrs:=v.split("_")[0] in standard_attributes:
            for k, v in standard_attributes[attrs].items():
                ds[v].attrs[k] = v

