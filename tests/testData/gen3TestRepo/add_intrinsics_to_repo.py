# mypy: ignore-errors
from lsst.daf.butler import Butler, DatasetType, Timespan
from lsst.ip.isr import IntrinsicZernikes

butler = Butler("./", instrument="LSSTCam", collections="LSSTCam/aos/intrinsic", writeable=True)

intrinsic_zernikes_type = DatasetType(
    "intrinsicZernikes",
    ("instrument", "physical_filter", "detector"),
    "IsrCalib",
    universe=butler.dimensions,
    isCalibration=True,
)
butler.registry.registerDatasetType(intrinsic_zernikes_type)

ds_src = butler.query_datasets("intrinsic_aberrations_temp")
ds_dest = butler.query_datasets("intrinsicZernikes", explain=False)
for dataset in ds_src:
    if dataset.dataId in [dsd.dataId for dsd in ds_dest]:
        continue
    data = butler.get(dataset)
    izk = IntrinsicZernikes(table=data)
    ref = butler.put(izk, "intrinsicZernikes", dataId=dataset.dataId, run="LSSTCam/aos/intrinsic")
    butler.registry.certify(
        "LSSTCam/calib",
        [ref],
        timespan=Timespan(None, None),
    )
