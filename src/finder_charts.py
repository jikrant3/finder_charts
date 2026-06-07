from pathlib import Path
from typing import Literal
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astroplan import Target
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.skyview import SkyView
from matplotlib.patches import Circle, Polygon


def ensure_wcs_axes(ax, wcs: WCS) -> WCSAxes:
    """
    Ensure the given axes object is a WCSAxes projection.

    If the input axes object ``ax`` is already a WCSAxes instance, it is returned
    unchanged. Otherwise, the original axes is removed and a new WCSAxes is created
    in the same position using the provided WCS projection ``wcs``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Input axes object to check or convert.
    wcs : astropy.wcs.WCS
        World Coordinate System projection to use for the output axes.

    Returns
    -------
    astropy.visualization.wcsaxes.WCSAxes
        Output WCSAxes object with the provided WCS projection.
    """
    if isinstance(ax, WCSAxes):
        return ax  # already correct

    fig = ax.figure
    pos = ax.get_position()  # preserve exact layout

    # remove old axis
    ax.remove()

    # create new WCS axis in same position
    ax_wcs = fig.add_axes(pos, projection=wcs)

    return ax_wcs


class InstrumentFOV:
    """
    Field of view definition for a specific instrument.

    Parameters
    ----------
    name : str
        Instrument name.
    shape : {'rectangle', 'circle'}, optional
        Shape of the field of view. Default is 'rectangle'.
    radius : astropy.units.Quantity, optional
        Radius of the circular FOV. Required when ``shape='circle'``.
    width : astropy.units.Quantity, optional
        Width of the rectangular FOV. Required when ``shape='rectangle'``.
    height : astropy.units.Quantity, optional
        Height of the rectangular FOV. Required when ``shape='rectangle'``.
    color : str, optional
        Color used for plotting the FOV. Default is 'g'.

    Attributes
    ----------
    label : str
        Description of the instrument and FOV size.
    width : astropy.units.Quantity
        Width in arcminutes for a rectangular FOV.
    height : astropy.units.Quantity
        Height in arcminutes for a rectangular FOV.
    radius : astropy.units.Quantity
        Radius in arcminutes for a circular FOV.

    Examples
    --------
    instrument_fov = InstrumentFOV(
        "ALFOSC@NOT",
        shape="rectangle",
        width=2048 * 0.2138 * u.arcsec,
        height=2064 * 0.2138 * u.arcsec,
    )
    instrument_fov = InstrumentFOV(
        "test",
        shape="circle",
        radius=2048 * 0.2138 * u.arcsec/2,
    )
    """

    @u.quantity_input
    def __init__(
        self,
        name,
        shape: Literal["rectangle", "circle"] = "rectangle",
        radius: u.deg = None,
        width: u.deg = None,
        height: u.deg = None,
        position_angle: u.deg = 0 * u.deg,
        color="g",
    ):
        self.name = name
        self.shape = shape.lower()
        self.color = color
        self.label = f"Instrument: {self.name}\n"
        if self.shape == "rectangle":
            self.width = width.to(u.arcmin)
            self.height = height.to(u.arcmin)
            self.height = height.to(u.arcmin)
            self.position_angle = position_angle.to(u.deg)
            self.label += (
                f"FOV:        {self.width.to_value(u.arcmin):.2f}′"
                + f"× {self.height.to_value(u.arcmin):.2f}′"
            )
            if position_angle != 0 * u.deg:
                self.label += f"\nPA:         {self.position_angle.to_value(u.deg):.2f}°"
        if self.shape == "circle":
            self.radius = radius.to(u.arcmin)
            self.label += f"FOV:        {self.radius.to_value(u.arcmin):.2f}′"
        else:
            NotImplementedError("Only 'rectangle' or 'circle' shapes are accepted.")


class Slit:
    """
    Represent a rectangular slit on the sky.

    Parameters
    ----------
    ra : `~astropy.coordinates.Angle` or `~astropy.coordinates.SkyCoord`
        Right ascension of the slit center.
    dec : `~astropy.coordinates.Angle` or `~astropy.coordinates.SkyCoord`
        Declination of the slit center.
    length : `~astropy.units.Quantity`
        Slit length.
    width : `~astropy.units.Quantity`
        Slit width.
    position_angle : `~astropy.units.Quantity`, optional
        Position angle of the slit in degrees. Default is 0 deg.
    color : str, optional
        Color used to draw the slit. Default is 'r'.

    Attributes
    ----------
    ra : same as input
        Right ascension of the slit center.
    dec : same as input
        Declination of the slit center.
    length : same as input
        Slit length.
    width : same as input
        Slit width.
    position_angle : same as input
        Position angle of the slit.
    color : same as input
        Drawing color.

    Examples
    --------
    slit_1 = Slit(
        target.ra,
        target.dec,
        6 * u.arcmin,
        1 * u.arcsec,
        position_angle=0 * u.deg,
    )
    slit_2 = Slit(
        target.ra + 1 * u.arcmin,
        target.dec,
        6 * u.arcmin,
        1 * u.arcsec,
        position_angle=20 * u.deg,
        color="b",
    )
    """

    def __init__(
        self,
        ra: Angle | u.Quantity,
        dec: Angle | u.Quantity,
        length: u.Quantity,
        width: u.Quantity,
        position_angle: u.Quantity = 0 * u.deg,
        color: str = "r",
    ) -> None:
        self.ra = ra
        self.dec = dec
        self.length = length
        self.width = width
        self.position_angle = position_angle
        self.color = color


class FinderChart:
    """
    Generate and annotate astronomical finder charts from SkyView survey images.

    The class provides utilities for:

    - Downloading survey images with :class:`astropy.wcs.WCS` support.
    - Plotting finder images with celestial coordinate overlays.
    - Overlaying slit geometries and instrument fields of view.
    - Rescaling views around arbitrary sky coordinates.

    Notes
    -----
    The downloaded image and associated WCS are stored on the instance after
    calling :meth:`get_image`.

    References
    ----------
    - https://astroplan.readthedocs.io/en/latest/api/astroplan.plots.plot_finder_image.html
    - https://github.com/Astro-Sean/finder_chart

    Examples
    --------
    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> target = Target.from_name("M51")
    >>> fc = FinderChart(target)
    >>> fc.get_image(5 * u.arcmin)
    >>> ax = fc.plot_image()
    >>> fc.plot_reticle(ax)
    >>> plt.show()
    """

    def __init__(self, target: Target) -> None:
        """
        Initialize a finder chart for a target.

        Parameters
        ----------
        target : Target
            Target object containing sky coordinates and metadata.
        """
        self.target = target
        self.name = target.name
        self.coord = target.coord.icrs

    @u.quantity_input
    def get_image(
        self,
        fov_radius: u.Quantity[u.deg],
        survey: str = "DSS",
        **kwargs,
    ):
        """
        Retrieve a survey image from SkyView.

        Parameters
        ----------
        fov_radius : `~astropy.units.Quantity`
            Angular radius of the requested image.
        survey : str, optional
            SkyView survey name, by default ``"DSS"``.
        **kwargs
            Additional keyword arguments passed to
            :func:`astroquery.skyview.SkyView.get_images`.

        Returns
        -------
        `~astropy.io.fits.PrimaryHDU`
            FITS HDU containing the image data.

        Notes
        -----
        The retrieved image, WCS, survey name, and field-of-view radius are
        cached on the instance.

        Examples
        --------
        >>> import astropy.units as u
        >>> fc.get_image(10 * u.arcmin, survey="DSS2 Red")
        """
        position = self.coord
        coordinates = "icrs"
        self.hdu = SkyView.get_images(
            position=position, coordinates=coordinates, survey=survey, radius=fov_radius, **kwargs
        )[0][0]
        self.wcs = WCS(self.hdu.header)
        self.survey = survey
        self.fov_radius = fov_radius

        self.image_label = f"Survey: {self.survey}\n"
        self.image_label += f"FOV:    {2 * fov_radius.to_value(u.arcmin):.2f}′× {2 * fov_radius.to_value(u.arcmin):.2f}′"
        return self.hdu

    def load_image(
        self,
        file_name: str,
        survey_name: str = "--",
    ):
        """
        Load a FITS image from disk and initialize the associated WCS.

        Parameters
        ----------
        file_name : str
            Path to the FITS image file.
        survey_name : str, optional
            Label describing the image survey or instrument.
            Default is ``"--"``.

        Returns
        -------
        `~astropy.io.fits.PrimaryHDU`
            Primary HDU containing the image data and header.

        Notes
        -----
        The loaded FITS HDU, WCS solution, survey label, and estimated
        field-of-view are cached on the instance.

        The field-of-view is estimated assuming square pixels using:

        .. math::

            \\mathrm{FOV} =
            N_\\mathrm{pix} \\times
            \\mathrm{pixel\\ scale}

        where the pixel scale is derived from
        :func:`astropy.wcs.utils.proj_plane_pixel_scales`.

        Examples
        --------
        Load a local FITS image:

        >>> fc.load_image("finder.fits")

        Load an Astropy tutorial image:

        >>> from astropy.utils.data import get_pkg_data_filename
        >>> file_name = get_pkg_data_filename(
        ...     "galactic_center/gc_msx_e.fits"
        ... )
        >>> fc.load_image(file_name, survey_name="MSX")
        """

        self.hdu = fits.open(file_name)[0]
        self.wcs = WCS(self.hdu.header)
        self.survey = survey_name
        self.fov_radius = proj_plane_pixel_scales(self.wcs)[0] * self.hdu.data.shape[0] * u.deg

        self.image_label = f"Survey: {self.survey}\n"
        self.image_label += (
            f"FOV:    {2 * self.fov_radius.to_value(u.arcmin):.2f}′× "
            + f"{2 * self.fov_radius.to_value(u.arcmin):.2f}′"
        )
        return self.hdu

    def plot_image(
        self,
        ax: plt.Axes | None = None,
        log: bool = False,
        show_degrees: bool = False,
        show_details: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the finder image.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Existing matplotlib axes. If `None`, a new figure and axes are
            created.
        log : bool, optional
            If `True`, display the image using logarithmic scaling.
        show_degrees : bool, optional
            If `True`, add overlay axes with coordinates formatted in degrees.
        show_details : bool, optional
            If `True`, display target and survey metadata on the plot.
        **kwargs
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        `~matplotlib.axes.Axes`
            WCS-aware matplotlib axes.

        Examples
        --------
        >>> ax = fc.plot_image(log=True, cmap="viridis")
        """
        if kwargs is None:
            kwargs = {}
        kwargs = dict(kwargs)
        kwargs.setdefault("cmap", "Greys")
        kwargs.setdefault("origin", "lower")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax = ensure_wcs_axes(ax, self.wcs)
        if log:
            image_data = np.log(self.hdu.data)
        else:
            image_data = self.hdu.data
        ax.imshow(image_data, **kwargs)

        x0 = ax.coords[0]
        y0 = ax.coords[1]
        # ONLY bottom/left
        x0.set_ticks_position("b")
        x0.set_ticklabel_position("b")
        x0.set_axislabel_position("b")
        y0.set_ticks_position("l")
        y0.set_ticklabel_position("l")
        y0.set_axislabel_position("l")

        if show_degrees:
            # Overlay axes (top/right)
            overlay = ax.get_coords_overlay("icrs")
            ra_deg = overlay["ra"]
            dec_deg = overlay["dec"]
            # ONLY top/right
            ra_deg.set_ticks_position("t")
            ra_deg.set_ticklabel_position("t")
            dec_deg.set_ticks_position("r")
            dec_deg.set_ticklabel_position("r")
            # Format in degrees
            ra_deg.set_format_unit(u.deg)
            dec_deg.set_format_unit(u.deg)
            ra_deg.set_major_formatter("d.ddd")
            dec_deg.set_major_formatter("d.ddd")
            ra_deg.set_ticks(number=6)
            dec_deg.set_ticks(number=6)
            # Labels
            ra_deg.set_axislabel("R.A. [°]")
            dec_deg.set_axislabel("Dec. [°]")
        if "ra" not in ax.coords:
            warnings.warn(
                "The image WCS is not in equatorial coordinates."
                + "The position angles are calculated relative to equatorial north pole."
            )

        if self.target.name is not None:
            label = f"Target:        {self.target.name}\n"
        else:
            label = ""
        label += (
            f"Image center:  {self.coord.ra.to_value(u.deg):3.6f}°, "
            + f"{self.coord.dec.to_value(u.deg):3.6f}°\n"
        )
        label += f"               {self.coord.ra.to_string(unit=u.hour, sep=':', precision=2)}, "
        label += f"{self.coord.dec.to_string(unit=u.degree, sep=':', precision=2)}"

        self.target_label = label

        if show_details:
            ax.text(
                0.02,
                0.98,
                label,
                va="top",
                ha="left",
                ma="left",
                transform=ax.transAxes,
                color="r",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
                weight="bold",
            )
            ax.text(
                0.98,
                0.98,
                self.image_label,
                va="top",
                ha="right",
                ma="left",
                transform=ax.transAxes,
                color="r",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
            )
        return ax

    def plot_reticle(
        self,
        ax: plt.Axes,
        **kwargs,
    ) -> None:
        """
        Draw a reticle centered on the image.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes on which to draw the reticle.
        **kwargs
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.axvline` and
            :meth:`matplotlib.axes.Axes.axhline`.

        Examples
        --------
        >>> ax = fc.plot_image()
        >>> fc.plot_reticle(ax, color="cyan")
        """
        pixel_width = self.hdu.data.shape[0]
        inner, outer = 0.03, 0.08

        if kwargs is None:
            kwargs = {}
        kwargs = dict(kwargs)
        kwargs.setdefault("linewidth", 2)
        kwargs.setdefault("color", "m")

        ax.axvline(x=0.5 * pixel_width, ymin=0.5 + inner, ymax=0.5 + outer, **kwargs)
        ax.axvline(x=0.5 * pixel_width, ymin=0.5 - inner, ymax=0.5 - outer, **kwargs)
        ax.axhline(y=0.5 * pixel_width, xmin=0.5 + inner, xmax=0.5 + outer, **kwargs)
        ax.axhline(y=0.5 * pixel_width, xmin=0.5 - inner, xmax=0.5 - outer, **kwargs)

    @u.quantity_input
    def plot_slits(
        self,
        slits: Slit | list[Slit],
        ax: plt.Axes | None = None,
        show_details: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot multiple slits on the finder chart.

        Parameters
        ----------
        slits : Slit or list of Slit
            Slit definitions to overlay.
        ax : `~matplotlib.axes.Axes`, optional
            Existing matplotlib axes.
        show_details : bool, optional
            If `True`, display slit metadata.
        **kwargs
            Additional keyword arguments passed to :meth:`plot_slit`.

        Examples
        --------
        >>> fc.plot_slits([slit1, slit2])
        """
        # Make sure slits is a list
        slits = [slits] if not isinstance(slits, (list, tuple)) else slits

        if ax is None:
            fig, ax = plt.subplots()
        ax = ensure_wcs_axes(ax, self.wcs)

        labels = ""
        for i, slit in enumerate(slits):
            self.plot_slit(slit, ax=ax, show_details=False, **kwargs)
            if i > 0:
                labels += "\n"
            labels += slit.label.replace("Slit", "Slit_%d" % i).replace(
                "               ", "                 "
            )

        if show_details:
            ax.text(
                0.98,
                0.02,
                labels,
                va="bottom",
                ha="right",
                ma="left",
                transform=ax.transAxes,
                color="r",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
            )

            for i, slit in enumerate(slits):
                ax.text(
                    slit.corners_in_pixels[0, 0],
                    slit.corners_in_pixels[0, 1],
                    "Slit_%d" % i,
                    color=slit.color,
                )
                pass

    @u.quantity_input
    def plot_slit(
        self,
        slit: Slit,
        ax: plt.Axes | None = None,
        show_details: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot a slit overlay on the finder chart.

        Parameters
        ----------
        slit : Slit
            Slit geometry definition.
        ax : `~matplotlib.axes.Axes`, optional
            Existing matplotlib axes.
        show_details : bool, optional
            If `True`, display slit metadata.
        **kwargs
            Additional keyword arguments passed to
            :class:`matplotlib.patches.Polygon`.

        Notes
        -----
        The slit is projected into pixel coordinates using the current image
        WCS.

        Examples
        --------
        >>> fc.plot_slit(slit, edgecolor="lime")
        """
        if kwargs is None:
            kwargs = {}
        kwargs = dict(kwargs)
        kwargs.setdefault("linewidth", 0.5)
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("edgecolor", slit.color)
        if ax is None:
            fig, ax = plt.subplots()
        ax = ensure_wcs_axes(ax, self.wcs)

        ra_offset = self.coord.ra - slit.ra
        dec_offset = self.coord.dec - slit.dec
        # Calculate the paragalactic angle for the RA/Dec position
        sky_coord_slit = SkyCoord(
            ra=slit.ra,
            dec=slit.dec,
        )
        paragalactic_angle = slit.position_angle
        # Define the half-lengths for easy calculation of corners
        half_length = slit.length / 2
        half_width = slit.width / 2

        # Define corner positions based on the central point and paragalactic angle
        # Offset directions: +paragalactic_angle (along length), ±π/2 (perpendicular for width)
        corner1 = sky_coord_slit.directional_offset_by(
            paragalactic_angle, half_length
        ).directional_offset_by(paragalactic_angle + np.pi * u.rad / 2, half_width)
        corner2 = sky_coord_slit.directional_offset_by(
            paragalactic_angle, half_length
        ).directional_offset_by(paragalactic_angle - np.pi * u.rad / 2, half_width)
        corner3 = sky_coord_slit.directional_offset_by(
            paragalactic_angle + np.pi * u.rad, half_length
        ).directional_offset_by(paragalactic_angle - np.pi * u.rad / 2, half_width)
        corner4 = sky_coord_slit.directional_offset_by(
            paragalactic_angle + np.pi * u.rad, half_length
        ).directional_offset_by(paragalactic_angle + np.pi * u.rad / 2, half_width)

        # Convert corners to pixel coordinates for plotting
        corner1_pix = self.wcs.world_to_pixel(corner1)
        corner2_pix = self.wcs.world_to_pixel(corner2)
        corner3_pix = self.wcs.world_to_pixel(corner3)
        corner4_pix = self.wcs.world_to_pixel(corner4)

        slit_polygon = Polygon(
            [corner1_pix, corner2_pix, corner3_pix, corner4_pix],
            closed=True,
            **kwargs,
        )
        ax.add_patch(slit_polygon)
        slit.corners_in_pixels = np.array([corner1_pix, corner2_pix, corner3_pix, corner4_pix])

        label = ""
        if (ra_offset == 0 * u.deg) & (dec_offset == 0 * u.deg):
            label += "Slit position: Image center\n"
        else:
            label += f"Slit position: {slit.ra.to_value(u.deg):.6f}°,"
            label += f"{slit.dec.to_value(u.deg):.6f}°\n"
            label += f"               {slit.ra.to_string(unit=u.hour, sep=':', precision=2)},"
            label += f"{slit.dec.to_string(unit=u.degree, sep=':', precision=2)}\n"
            label += (
                f"Slit offset:   ΔRa = {ra_offset.to_value(u.arcsec):.2f}″, "
                + f"ΔDec = {dec_offset.to_value(u.arcsec):.2f}″\n"
            )
        label += f"Slit size:     {slit.length.to_value(u.arcsec):.2f}″"
        label += f"× {slit.width.to_value(u.arcsec):.2f}″\n"
        label += f"Slit PA:       {slit.position_angle.to_value(u.deg)}°"
        if show_details:
            ax.text(
                0.98,
                0.02,
                label,
                va="bottom",
                ha="right",
                ma="left",
                transform=ax.transAxes,
                color=slit.color,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
            )
        slit.label = label

    def plot_instrument_fov(
        self,
        instrument_fov: InstrumentFOV,
        ax: plt.Axes | None = None,
        show_details: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot an instrument field-of-view overlay.

        Parameters
        ----------
        instrument_fov : InstrumentFOV
            Instrument field-of-view definition.
        ax : `~matplotlib.axes.Axes`, optional
            Existing matplotlib axes.
        show_details : bool, optional
            If `True`, display field-of-view metadata.
        **kwargs
            Additional keyword arguments passed to the matplotlib patch object.

        Examples
        --------
        >>> fc.plot_instrument_fov(fov)
        """
        if kwargs is None:
            kwargs = {}
        kwargs = dict(kwargs)
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("edgecolor", instrument_fov.color)
        if ax is None:
            fig, ax = plt.subplots()
        ax = ensure_wcs_axes(ax, self.wcs)

        if instrument_fov.shape == "rectangle":
            _fov_as_slit = Slit(
                self.coord.ra,
                self.coord.dec,
                instrument_fov.height,
                instrument_fov.width,
                position_angle=instrument_fov.position_angle,
                color=instrument_fov.color,
            )
            self.plot_slit(_fov_as_slit, ax=ax, show_details=False, **kwargs)
        if instrument_fov.shape == "circle":
            # Convert center to pixel coordinates for plotting
            center_pix = self.wcs.world_to_pixel(self.coord)
            pixel_scale = (self.wcs.proj_plane_pixel_scales() / u.pix).to(u.arcsec / u.pix)[0]
            radius = (instrument_fov.radius / pixel_scale).to(u.pix)
            circ = Circle(center_pix, radius=radius.to_value(u.pix), **kwargs)
            ax.add_artist(circ)
        if show_details:
            ax.text(
                0.02,
                0.02,
                instrument_fov.label,
                va="bottom",
                ha="left",
                ma="left",
                transform=ax.transAxes,
                color="r",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
            )

    @u.quantity_input
    def rescale_image(
        self,
        ax: plt.Axes,
        ra: u.Quantity[u.deg],
        dec: u.Quantity[u.deg],
        fov_radius: u.Quantity[u.arcsec] = 10 * u.arcsec,
        reticle: bool = True,
    ) -> None:
        """
        Rescale the displayed image around a sky position. Useful while showing zoom-in plots.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes containing the image.
        ra : `~astropy.units.Quantity`
            Right ascension of the new image center.
        dec : `~astropy.units.Quantity`
            Declination of the new image center.
        fov_radius : `~astropy.units.Quantity`, optional
            Radius of the displayed field of view.
        reticle : bool, optional
            If `True`, mark the center position with a reticle.

        Examples
        --------
        >>> fc.rescale_image(
        ...     ax,
        ...     ra=150.1 * u.deg,
        ...     dec=2.3 * u.deg,
        ...     fov_radius=20 * u.arcsec,
        ... )
        """
        center_pix = self.wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec))
        pixel_scale = (proj_plane_pixel_scales(self.wcs) * u.deg / u.pix).to(u.arcsec / u.pix)[0]
        radius = (fov_radius / pixel_scale).to_value(u.pix)
        ax.set_xlim(center_pix[0] - radius, center_pix[0] + radius)
        ax.set_ylim(center_pix[1] - radius, center_pix[1] + radius)

        if reticle:
            ax.scatter(*center_pix, marker="o", fc="none", ec="m", s=1000)
        return ax

    def plot_compass(
        self,
        ax: WCSAxes,
        frac_length: float = 0.12,
        x_frac: float = 0.08,
        y_frac: float = 0.08,
        color: str = "m",
        linewidth: float = 2.0,
        fontsize: float = 12,
    ):
        """
        Plot North/East compass arrows on a WCS image.

        The compass position and size scale automatically with the image size.

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            Axes containing the image.
        wcs : `~astropy.wcs.WCS`
            Celestial WCS associated with the image.
        frac_length : float, optional
            Arrow length as a fraction of the image size.
            Default is ``0.12``.
        x_frac : float, optional
            Horizontal anchor position as a fraction of image width.
            ``0`` corresponds to the left edge and ``1`` to the right edge.
            Default is ``0.08``.
        y_frac : float, optional
            Vertical anchor position as a fraction of image height.
            ``0`` corresponds to the bottom edge and ``1`` to the top edge.
            Default is ``0.08``.
        frame : str, optional
            Coordinate frame used for plotting.
            Default is ``"icrs"``.
        color : str, optional
            Arrow and label color.
        linewidth : float, optional
            Arrow line width.
        fontsize : float, optional
            Label font size.

        Returns
        -------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            Modified axes.

        Notes
        -----
        The compass orientation is computed from the local celestial geometry,
        ensuring correct orientation for rotated or distorted projections.

        Examples
        --------
        >>> plot_compass(ax, wcs)

        Place the compass in the lower-right corner:

        >>> plot_compass(
        ...     ax,
        ...     wcs,
        ...     x_frac=0.9,
        ...     y_frac=0.1,
        ... )

        Increase compass size:

        >>> plot_compass(
        ...     ax,
        ...     wcs,
        ...     frac_length=0.2,
        ... )
        """
        wcs = self.wcs
        nx = wcs.pixel_shape[0]
        ny = wcs.pixel_shape[1]

        size = min(nx, ny)

        # Anchor position in pixel coordinates
        x0 = x_frac * nx
        y0 = y_frac * ny

        # Arrow size in pixels
        arrow_pix = frac_length * size

        # Convert anchor pixel -> world coordinates
        sky0 = wcs.pixel_to_world(x0, y0).icrs

        # Local pixel scale
        pixscale = (abs(wcs.proj_plane_pixel_scales()[0]) / u.pix).to(u.arcsec / u.pix)

        # Convert arrow length to angular separation
        sep = (arrow_pix * u.pix * pixscale).to(u.arcsec)

        # North/East positions
        north = sky0.directional_offset_by(
            0 * u.deg,
            sep,
        ).icrs

        east = sky0.directional_offset_by(
            90 * u.deg,
            sep,
        ).icrs

        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        # Convert to pixel coordinates
        x0p, y0p = wcs.world_to_pixel(sky0)
        xnp, ynp = wcs.world_to_pixel(north)
        xep, yep = wcs.world_to_pixel(east)

        # Draw arrows in pixel coordinates
        ax.arrow(
            x0p,
            y0p,
            xnp - x0p,
            ynp - y0p,
            color=color,
            width=0,
            head_width=0.015 * size,
            length_includes_head=True,
            linewidth=linewidth,
        )

        ax.arrow(
            x0p,
            y0p,
            xep - x0p,
            yep - y0p,
            color=color,
            width=0,
            head_width=0.015 * size,
            length_includes_head=True,
            linewidth=linewidth,
        )

        # Labels
        north_label = sky0.directional_offset_by(
            0 * u.deg,
            1.15 * sep,
        ).icrs

        east_label = sky0.directional_offset_by(
            90 * u.deg,
            1.15 * sep,
        ).icrs

        xnlp, ynlp = wcs.world_to_pixel(north_label)
        xelp, yelp = wcs.world_to_pixel(east_label)

        ax.text(
            xnlp,
            ynlp,
            "N",
            color=color,
            fontsize=fontsize,
            ha="center",
            va="center",
            weight="bold",
        )

        ax.text(
            xelp,
            yelp,
            "E",
            color=color,
            fontsize=fontsize,
            ha="center",
            va="center",
            weight="bold",
        )
        return ax


def savefig(
    plot_name=None,
    close_plot: bool = False,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
    **kwargs,
) -> None:
    """
    Save the current plot to the specified file.

    Parameters
    ----------
    plot_name : str, optional
        The file name (including path) to save the plot. If None, the plot is not saved.
    close_plot : bool, default=False
        Whether to close the plot after saving.
    dpi : int, default=300
        The resolution in dots per inch.
    bbox_inches : str, default="tight"
        Bounding box option passed to `plt.savefig`.
    facecolor : str, default="white"
        The facecolor of the saved figure.
    **kwargs : dict
        Additional keyword arguments passed to `plt.savefig`.

    Notes
    -----
    If the directory in `plot_name` does not exist, it will be created automatically.
    """
    if plot_name is not None:
        plot_name = Path(plot_name)
        plot_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure folders exist
        plt.savefig(plot_name, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor, **kwargs)
        if close_plot:
            plt.close()
