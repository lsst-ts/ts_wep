from lsst.daf.butler import Butler, DatasetType, DimensionUniverse, Timespan
from lsst.ip.isr import IntrinsicZernikes

butler = Butler("./", instrument="LSSTCam", collections="LSSTCam/aos/intrinsic", writeable=True)

intrinsic_zernikes_type = DatasetType(
    "intrinsic_zernikes",
    ("instrument", "physical_filter", "detector"),
    "IsrCalib",
    universe=DimensionUniverse(),
    isCalibration=True,
)
butler.registry.registerDatasetType(intrinsic_zernikes_type)

ds_src = butler.query_datasets("intrinsic_aberrations_temp")
ds_dest = butler.query_datasets("intrinsic_zernikes", explain=False)
for dataset in ds_src:
    if dataset.dataId in [dsd.dataId for dsd in ds_dest]:
        continue
    data = butler.get(dataset)
    izk = IntrinsicZernikes(table=data)
    ref = butler.put(izk, "intrinsic_zernikes", dataId=dataset.dataId, run="LSSTCam/aos/intrinsic")
    butler.registry.certify(
        "LSSTCam/calib",
        [ref],
        timespan=Timespan(None, None),
    )
