"""Convert NSRDB data model"""


import xarray as xr
from mylib import *

h5filename = "/datasets/NSRDB/current/nsrdb_2000.h5"

ds = xr.open_mfdataset(h5filename, mask_and_scale=False, engine="netcdf4")
ds = fix_time(ds)
# What to do with timezone??
ds["time"].attrs = {
    "standard_name": "time",
    "long_name": "Time",
    # Should we keep this?
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "gregorian",
}

# Figure out which phony_dim is time and location.
# The other dimension that is not associated with time_index will be location.

# Think about if we want to call that dimension as `location`!


#ds["lat"] = ds.coordinates.isel(phony_dim_1=0)
#ds["lon"] = ds.coordinates.isel(phony_dim_1=1)
#ds = ds.drop_vars(["coordinates"])
ds = ds.rename_dims({"phony_dim_1": "location"})
# ds = ds.set_coords(["lat", "lon"])

# What is the correct name for 'timezone' coordinate
# Missing all the attributes for meta variables. For instance, attributes for lat/lon


for v in extract_meta(h5filename):
    ds[v.name] = v
    ds = ds.set_coords(v.name)

# Verify if we should indeed use f64 for everything.
# One idea is to look into the original data type. For instance, int8 doesn't
# have the range to take advantage of f64, so it would be a waste.
# Another point is to save space on writing. The example is alpha, which is
# originally uint8, so we would waste space recording as f32 (even worse as f64). Ideally we would like to encode back on the same data type since we can't recover precision.
for v in ds:
    ds[v] = fix_variable(ds[v])


