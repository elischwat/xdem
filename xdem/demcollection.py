"""DEM collection class and functions."""
from __future__ import annotations

import datetime
import warnings
from typing import Optional, Union

import geoutils as gu
import numpy as np
import pandas as pd

import xdem


class DEMCollection:
    """A temporal collection of DEMs."""

    def __init__(self, dems: Union[list[gu.georaster.Raster], list[xdem.DEM]],
                 timestamps: Optional[list[datetime.datetime]] = None,
                 outlines: Optional[Union[gu.geovector.Vector, dict[datetime.datetime, gu.geovector.Vector], dict[pd.Interval, gu.geovector.Vector]]] = None,
                 reference_dem: Union[int, gu.georaster.Raster] = 0):
        """
        Create a new temporal DEM collection.

        :param dems: A list of DEMs.
        :param timestamps: A list of DEM timestamps.
        :param outlines: Polygons to separate the changing area of interest. Could for example be glacier outlines.
        :param reference_dem: An instance or index of which DEM in the 'dems' list is the reference.

        :returns: A new DEMCollection instance.
        """
        # If timestamps is not given, try to parse it from the (potential) 'datetime' attribute of each DEM.
        if timestamps is None:
            timestamp_attributes = [dem.datetime for dem in dems]
            if any([stamp is None for stamp in timestamp_attributes]):
                raise ValueError("'timestamps' not provided and the given DEMs do not all have datetime attributes")

            timestamps = timestamp_attributes

        if not all(isinstance(dem, xdem.DEM) for dem in dems):
            dems = [xdem.DEM.from_array(dem.data, dem.transform, dem.crs, dem.nodata) for dem in dems]

        assert len(dems) == len(timestamps), "The 'dem' and 'timestamps' len differ."

        # Convert the timestamps to datetime64
        self.timestamps = np.array(timestamps).astype("datetime64[ns]")

        # Find the sort indices from the timestamps
        indices = np.argsort(self.timestamps.astype("int64"))
        self.dems = np.empty(len(dems), dtype=object)
        self.dems[:] = [dems[i] for i in indices]
        self.ddems: list[xdem.dDEM] = []
        self.ddems_are_intervalwise = False
        # The reference index changes place when sorted
        if isinstance(reference_dem, (int, np.integer)):
            self.reference_index = np.argwhere(indices == reference_dem)[0][0]
        elif isinstance(reference_dem, gu.georaster.Raster):
            self.reference_index = [i for i, dem in enumerate(self.dems) if dem is reference_dem][0]

        if outlines is None:
            self.outlines: dict[np.datetime64, gu.geovector.Vector] = {}
        elif isinstance(outlines, gu.geovector.Vector):
            self.outlines = {self.timestamps[self.reference_index]: outlines}
        elif all(isinstance(value, gu.geovector.Vector) for value in outlines.values()):
            self.outlines = dict(zip(np.array(list(outlines.keys())).astype("datetime64[ns]"), outlines.values()))
        else:
            raise ValueError(f"Invalid format on 'outlines': {type(outlines)},"
                             " expected one of ['gu.geovector.Vector', 'dict[datetime.datetime, gu.geovector.Vector']")

    @classmethod
    def from_files(cls, 
                    dem_filename_list, 
                    dem_datetime_list, 
                    reference_dem_date, 
                    bounds_vector, 
                    dst_res,
                    resampling)-> xdem.DEMCollection:
        """
        Create a new temporal DEM collection. 
        Args:
            dem_filename_list (list(str)): _description_
            dem_datetime_list (list(datetime)): _description_
            reference_dem_date (datetime): _description_
            bounds_vector (gu.Vector): vector containing bounds to crop all DEMs to
            dst_res (int): resolution for all dDEMS. If none is specified, the reference DEM resolution is used.

        Returns:
            _type_: A new DEMCollection instance.
        """
        assert len(dem_filename_list) == len(dem_datetime_list)
        reference_dem_index = dem_datetime_list.index(reference_dem_date)
        dem_list = [xdem.DEM(fn, datetime=dt) for (fn, dt) in zip(dem_filename_list, dem_datetime_list)]
        ref_dem = dem_list[reference_dem_index]
        if bounds_vector:
            _ = ref_dem.crop(bounds_vector)
        ref_dem = ref_dem.reproject(dst_res=dst_res, resampling=resampling)
        dem_list = [dem.reproject(ref_dem, resampling=resampling) for dem in dem_list]
        for dem,dt in zip(dem_list, dem_datetime_list):
            dem.datetime = dt
        return cls(dem_list, reference_dem=reference_dem_index)

    @property
    def reference_dem(self) -> gu.georaster.Raster:
        """Get the DEM acting reference."""
        return self.dems[self.reference_index]

    @property
    def reference_timestamp(self) -> np.datetime64:
        """Get the reference DEM timestamp."""
        return self.timestamps[self.reference_index]

    def subtract_dems(self, resampling_method: str = "cubic_spline") -> list[xdem.dDEM]:
        """
        Generate dDEMs by subtracting all DEMs to the reference.

        :param resampling_method: The resampling method to use if reprojection is needed.

        :returns: A list of dDEM objects.
        """
        ddems: list[xdem.dDEM] = []

        # Subtract every DEM that is available.
        for i, dem in enumerate(self.dems):
            # If the reference DEM is encountered, make a dDEM where dH == 0 (to keep length consistency).
            if dem == self.reference_dem:
                ddem_raster = self.reference_dem.copy()
                ddem_raster.data[:] = 0.0
                ddem = xdem.dDEM(
                    ddem_raster,
                    start_time=self.reference_timestamp,
                    end_time=self.reference_timestamp,
                    error=0,
                )
            else:
                ddem = xdem.dDEM(
                    self.reference_dem - dem.reproject(resampling=resampling_method, silent=True),
                    start_time=min(self.reference_timestamp, self.timestamps[i]),
                    end_time=max(self.reference_timestamp, self.timestamps[i]),
                    error=None
                )
            ddems.append(ddem)

        self.ddems = ddems
        return self.ddems
    
    def subtract_dems_intervalwise(self, create_bounding_dod: bool = False) -> list[xdem.dDEM]:
        """
        Generate dDEMs by subtracting sequential pairs of DEMs as to create a "time series" of dDEMs.

        Has side effect of making self.dems sorted by datetime.

        :returns: A list of dDEM objects.
        """
        ddems: list[xdem.dDEM] = []

        self.dems = sorted(self.dems, key=lambda x: x.datetime)

        for first_dem, last_dem in zip(self.dems, self.dems[1:]):
            ddem = xdem.dDEM(
                last_dem - first_dem,
                start_time = first_dem.datetime,
                end_time = last_dem.datetime,
                error = 0
            )
            ddems.append(ddem)
        if create_bounding_dod:
            bounding_interval = pd.Interval(pd.Timestamp(self.timestamps[0]), pd.Timestamp(self.timestamps[-1]))
            bounding_outlines = self.outlines.get(bounding_interval)

            bounding_dem_collection = xdem.DEMCollection(
                [self.dems[0], self.dems[-1]],
                [self.timestamps[0], self.timestamps[-1]],
                outlines = bounding_outlines
            )

            _ = bounding_dem_collection.subtract_dems_intervalwise()

            ddems.append(
                bounding_dem_collection.ddems[0]
            )
        
        self.ddems = ddems
        self.ddems_are_intervalwise = True
        return self.ddems

    def interpolate_ddems(self, method="linear", max_search_distance=10):
        """
        Interpolate all the dDEMs in the DEMCollection object using the chosen interpolation method.

        :param method: The chosen interpolation method.
        :param max_search_distance: int. Only applicable for linear method of interpolation.
        """
        # TODO: Change is loop to run concurrently
        for ddem in self.ddems:
            ddem.interpolate(method=method, reference_elevation=self.reference_dem, mask=self.get_ddem_mask(ddem), max_search_distance=max_search_distance)

        return [ddem.filled_data for ddem in self.ddems]

    def set_ddem_filled_data(self):
        """Set the filled (interpolated) data as the data for all ddems.
        """
        [ddem.set_filled_data() for ddem in self.ddems]
    

    def get_ddem_mask(self, ddem: xdem.dDEM, outlines_filter: Optional[str] = None) -> np.ndarray:
        """
        Get a fitting dDEM mask for a provided dDEM.

        The mask is created by evaluating these factors, in order:

        If self.outlines do not exist, a full True boolean mask is returned.
        If self.outlines have keys that are pd.Intervals (of start and end times), then the interval of the ddem is used to query the polygons. 
        If self.outlines have keys for the start and end time, their union is returned.
        If self.outlines only have contain the start_time, its mask is returned.
        If len(self.outlines) == 1, the mask of that outline is returned.

        :param ddem: The dDEM to create a mask for.
        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".

        :returns: A mask from the above conditions.
        """
        if not any(ddem is ddem_in_list for ddem_in_list in self.ddems):
            raise ValueError("Given dDEM must be a part of the DEMCollection object.")

        if outlines_filter is None:
            outlines = self.outlines
        elif type(self.outlines) == dict:
            outlines = {key: gu.Vector(outline.ds.copy()) for key, outline in self.outlines.items()}
            for key in outlines:
                outlines[key].ds = outlines[key].ds.query(outlines_filter)
        elif type(self.outlines) == gu.Vector:
            outlines = self.outlines.query(outlines_filter)

        # If the outlines are a gu.Vector
        if type(outlines) == gu.Vector:
            mask = outlines.create_mask(ddem)
        # If the outlines are a dictionary with pd.Intervals as keys
        elif outlines and type(list(outlines.keys())[0]) == pd.Interval:
            if len(outlines[ddem.interval].ds):
                mask = outlines[ddem.interval].create_mask(ddem)
            else:
                mask = ~np.ones(shape=ddem.data.shape, dtype=bool)
        # If both the start and end time outlines exist, a mask is created from their union.
        elif ddem.start_time in outlines and ddem.end_time in outlines:
            mask = np.logical_or(
                outlines[ddem.start_time].create_mask(ddem),
                outlines[ddem.end_time].create_mask(ddem)
            )
        # If only start time outlines exist, these should be used as a mask
        elif ddem.start_time in outlines:
            mask = outlines[ddem.start_time].create_mask(ddem)
        # If only one outlines file exist, use that as a mask.
        elif len(outlines) == 1:
            mask = list(outlines.values())[0].create_mask(ddem)
        # If no fitting outlines were found, make a full true boolean mask in its stead.
        else:
            mask = np.ones(shape=ddem.data.shape, dtype=bool)
        return mask.reshape(ddem.data.shape)

    def get_dh_series(self, outlines_filter: Optional[str] = None, mask: Optional[np.ndarray] = None,
                      nans_ok: bool = False) -> pd.DataFrame:
        """
        Return a dataframe of mean dDEM values and respective areas for every timestamp.

        The values are always compared to the reference DEM timestamp.

        :param mask: Optional. A mask for areas of interest. Overrides potential outlines of the same date.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A dataframe of dH values and respective areas with an Interval[Timestamp] index.
        """
        if len(self.ddems) == 0:
            raise ValueError("dDEMs have not yet been calculated")

        dh_values = pd.DataFrame(columns=["dh", "area"], dtype=float)
        for i, ddem in enumerate(self.ddems):
            # Skip if the dDEM is a self-comparison
            if not self.ddems_are_intervalwise:
                if float(ddem.time) == 0:
                    continue

            # Use the provided mask unless it's None, otherwise make a dDEM mask.
            ddem_mask = mask if mask is not None else self.get_ddem_mask(ddem, outlines_filter=outlines_filter)

            # Warn if the dDEM contains nans and that's not okay
            # Does this conditional actually guarantee therre arer nans? Its possible filled_data is None and 
            #  and data does not have nans!
            if ddem.filled_data is None and not nans_ok:
                warnings.warn(f"NaNs found in dDEM ({ddem.start_time} - {ddem.end_time}).")

            data = ddem.data[ddem_mask] if ddem.filled_data is None else ddem.filled_data[ddem_mask]

            # This line will through an error if all values are masked in data
            #     mean_dh = np.nanmean(data)
            # Updated if clause fixes.
            """
            
            File ~/.conda/envs/xdem/lib/python3.9/site-packages/numpy/lib/nanfunctions.py:950, in nanmean(a, axis, dtype, out, keepdims)
                948 cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
                949 tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
            --> 950 avg = _divide_by_count(tot, cnt, out=out)
                952 isbad = (cnt == 0)
                953 if isbad.any():

            File ~/.conda/envs/xdem/lib/python3.9/site-packages/numpy/lib/nanfunctions.py:212, in _divide_by_count(a, b, out)
                210 if isinstance(a, np.ndarray):
                211     if out is None:
            --> 212         return np.divide(a, b, out=a, casting='unsafe')
                213     else:
                214         return np.divide(a, b, out=out, casting='unsafe')

            ValueError: output array is read-only
            """
            if isinstance(data, np.ma.MaskedArray):
                if data.mask.all():
                    mean_dh = 0 
                else:
                    mean_dh = np.nanmean(data)
            else:
                mean_dh = np.nanmean(data)
                
            area = np.count_nonzero(ddem_mask) * self.reference_dem.res[0] * self.reference_dem.res[1]

            dh_values.loc[pd.Interval(pd.Timestamp(ddem.start_time), pd.Timestamp(ddem.end_time))] = mean_dh, area

        return dh_values
    
    def mask_ddems(self, mask_vector: gu.Vector):
        for ddem in self.ddems:
            mask = mask_vector.create_mask(ddem)
            ddem.data.mask = np.logical_or(ddem.data.mask, mask)
        
    def mask_dems(self):
        return None

    def get_dv_series(self, outlines_filter: Optional[str] = None,
                      mask: Optional[np.ndarray] = None, nans_ok: bool = False, return_area: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        Return a series of mean volume change (dV) for every timestamp.

        The values are always compared to the reference DEM timestamp.

        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".
        :param mask: Optional. A mask for areas of interest. Overrides potential outlines of the same date.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of dV values with an Interval[Timestamp] index.
        """
        dh_values = self.get_dh_series(outlines_filter=outlines_filter, mask=mask, nans_ok=nans_ok)
        if return_area:
            dh_values["volume"] = dh_values["area"] * dh_values["dh"]
            return dh_values
        else:
            return dh_values["area"] * dh_values["dh"]

    def get_cumulative_series(self, kind: str = "dh", outlines_filter: Optional[str] = None,
                              mask: Optional[np.ndarray] = None,
                              nans_ok: bool = False) -> pd.Series:
        """
        Get the cumulative dH (elevation) or dV (volume) since the first timestamp.

        :param kind: The kind of series. Can be dh or dv.
        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".
        :param mask: Optional. A mask for areas of interest.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of cumulative dH/dV with a Timestamp index.
        """
        if kind.lower() == "dh":
            # Get the dH series (where all indices are: "year to reference_year")
            d_series = self.get_dh_series(mask=mask, outlines_filter=outlines_filter, nans_ok=nans_ok)["dh"]
        elif kind.lower() == "dv":
            # Get the dV series (where all indices are: "year to reference_year")
            d_series = self.get_dv_series(mask=mask, outlines_filter=outlines_filter, nans_ok=nans_ok)
        else:
            raise ValueError("Invalid argument: '{dh=}'. Choices: ['dh', 'dv']")

        # Simplify the index to just "year" (implictly still the same as above)
        cumulative_dh = pd.Series(dtype=d_series.dtype)
        cumulative_dh[self.reference_timestamp] = 0.0
        for i, value in zip(d_series.index, d_series.values):
            non_reference_year = [date for date in [i.left, i.right] if date != self.reference_timestamp][0]
            cumulative_dh.loc[non_reference_year] = -value

        # Sort the dates (just to be sure. It should already be sorted)
        cumulative_dh.sort_index(inplace=True)
        # Subtract the entire series by the first value to
        cumulative_dh -= cumulative_dh.iloc[0]

        return cumulative_dh

    def plot_dems(self, max_cols = 4, figsize = (30,16), sharey=True, sharex=True, hillshade=False, cmap="terrain", interpolation=None):
        import matplotlib.pyplot as plt
        import math
        fig, axes = plt.subplots(
            math.ceil(len(self.dems)/max_cols), 
            max_cols, 
            figsize=figsize, 
            sharex=sharex,
            sharey=sharey            
        )
        axes_flat = axes.flatten()
        for dem, ax in zip(self.dems, axes_flat):
            alpha = 1.0
            if hillshade:
                hillshade_data = xdem.terrain.hillshade(dem.data, resolution=dem.res, azimuth=315.0, altitude=45.0)
                ax.imshow(hillshade_data.squeeze(), cmap = "Greys_r", interpolation=interpolation)
                alpha = 0.5
            ax.imshow(dem.data.squeeze(), cmap="terrain", alpha=alpha)
            ax.set_title(dem.datetime.strftime("%Y/%m/%d"))
        #remove plot grid from unused axes
        n_empty_plots = len(axes_flat) - len(self.dems)
        for i   in range(1, n_empty_plots + 1):
            axes_flat[-i].set_axis_off()
        return fig, axes

    def plot_ddems(self, max_cols = 4, figsize = (30,16), sharey=True, sharex=True, hillshade=False, cmap="RdYlBu", vmin=-30, vmax=30, cmap_alpha = 1.0, interpolation=None, plot_outlines=False, edgecolor='k', linewidth=1):
        import matplotlib.pyplot as plt
        import math
        fig, axes = plt.subplots(
            math.ceil(len(self.ddems)/max_cols), 
            max_cols, 
            figsize=figsize, 
            sharex=sharex,
            sharey=sharey   
        )
        axes_flat = axes.flatten()
        for (i,ax), ddem in zip(enumerate(axes_flat), self.ddems):
            ddem_xr = ddem.to_xarray()

            if hillshade:
                hillshade_xr = ddem_xr.copy()
                hillshade_xr.data = xdem.terrain.hillshade(self.dems[i].data, resolution=self.dems[i].res, azimuth=315.0, altitude=45.0)
                hillshade_xr.plot(ax=ax, cmap = "Greys_r", add_colorbar=False)
            
            ddem_xr.data = ddem.data.filled(np.nan) # THIS IS NECESSARY BECAUSE THE to_xarray() FUNCTION PASSES A DATASET READER - NOT THE ACTUAL DATA
            # ddem_xr = ddem_xr.where(ddem_xr.data != ddem_xr.attrs['_FillValue'])  
            ddem_xr.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, alpha=cmap_alpha)
            ax.set_title(
                pd.to_datetime(ddem.start_time).strftime("%Y/%m/%d") + '-\n' + pd.to_datetime(ddem.end_time).strftime("%Y/%m/%d")
                )
            if plot_outlines:
                # if outlines are a dictionary of pd.Intervals to gu.Vector objects, pick the interval that matches the ddem
                if type(self.outlines) == dict and type(list(self.outlines.keys())[0]) == pd.Interval:
                    self.outlines[ddem.interval].ds.plot(ax=ax, facecolor="none", edgecolor=edgecolor, linewidth=linewidth)
                # If outlines are just a gu.Vector object, plot it no matter the ddem plotted here
                elif type(self.outlines) == gu.geovector.Vector:
                    self.outlines.ds.plot(ax=ax, facecolor="none", edgecolor=edgecolor, linewidth=linewidth)
                
        #remove plot grid from unused axes
        n_empty_plots = len(axes_flat) - len(self.ddems)
        for i in range(1, n_empty_plots + 1):
            axes_flat[-i].set_axis_off()
        return fig, axes
