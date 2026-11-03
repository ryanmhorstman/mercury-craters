"""Raster preparation: a tiled, compressed copy of the basemap."""
from osgeo import gdal

gdal.UseExceptions()


def build_cog(src, dst):
    """Copy src to a Cloud-Optimized GeoTIFF with overviews. Never touches src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.tif")
    gdal.Translate(
        str(tmp), str(src), format="COG",
        creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2", "BLOCKSIZE=512",
                         "OVERVIEWS=AUTO", "RESAMPLING=AVERAGE", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES"],
    )
    tmp.replace(dst)  # only a finished file ever has the final name
    return dst


if __name__ == "__main__":
    from .config import load_config

    paths = load_config()["paths"]
    print("building", paths["basemap"])
    build_cog(paths["basemap_raw"], paths["basemap"])
    print("done")
