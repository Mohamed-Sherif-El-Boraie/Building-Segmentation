import rasterio


def inspect_tif(path):
    with rasterio.open(path) as src:
        print(f"=== {path} ===")
        print(f"Bands:      {src.count}")
        print(f"Size:       {src.width}x{src.height}")
        print(f"Dtype:      {src.dtypes}")
        print(f"NoData:     {src.nodata}")
        print(f"CRS:        {src.crs}")
        print(f"Transform:  {src.transform}")

        # Check if geospatial
        has_crs = src.crs is not None
        has_transform = src.transform != rasterio.transform.Affine.identity()
        is_geotiff = has_crs and has_transform

        print(f"\n--- Geospatial Check ---")
        print(f"Has CRS:       {has_crs}")
        print(f"Has Transform: {has_transform}")
        print(f"Is GeoTIFF:    {'✅ Yes' if is_geotiff else '❌ No (plain TIFF)'}")

# Usage
inspect_tif("Image_Path.tif")