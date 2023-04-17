from setuptools import find_packages, setup

from smc01 import SMC01_APPNAME, SMC01_VERSION

setup(
    name=SMC01_APPNAME,
    version=SMC01_VERSION,
    description="Data assimilation and other utilities for the SMC01 project.",
    author="Computer Research Institute of Montreal (CRIM)",
    author_email="david.landry@crim.ca",
    packages=find_packages(exclude="test"),
    include_package_data=True,
    install_requires=[
        "hydra-submitit-launcher==1.2.*",
        "luigi==3.2.*",
        "hydra-core==1.3.*",
        "pandas==1.5.*",
        "pyarrow==11.0.*",
        "pygrib==2.1.*",
        "pymongo==4.1.*",
        "pytorch_lightning==1.9.*",
        "requests==2.28.*",
        "scipy==1.10.*",
        "tensorboard==2.12.*",
        "torch==1.13.*",
        "tqdm==4.64.*",
        "xarray==2023.2.*",
    ],
    extras_require={
        "dev": [
            "black==23.1.*",
            "isort==5.12.*",
            "pyright==1.1.*",
        ]
    },
    entry_points={
        "console_scripts": [
            "smc01_datamart_consumer = smc01.datamart_consumer:cli",
            "smc01_crawl_iem = smc01.iem.crawler:cli",
            "smc01_interpolate = smc01.interpolate.interpolate:cli",
            "smc01_thin = smc01.thin:cli",
            "smc01_train = smc01.postprocessing.cli.train:cli",
            "smc01_test = smc01.postprocessing.cli.test:cli",
            "smc01_luigi = smc01.luigi.luigi:cli",
        ],
    },
)
