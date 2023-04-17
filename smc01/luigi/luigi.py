import hydra
import importlib.resources
import luigi
import os
import pandas as pd

from .interpolation import InterpolatePass


@hydra.main(config_path="conf", config_name="luigi", version_base=None)
def cli(cfg):
    if "LUIGI_CONFIG_PATH" in os.environ:
        pass
    elif "luigi_cfg" in cfg and cfg.luigi_cfg:
        luigi.configuration.add_config_path(cfg.luigi_cfg)
    else:
        cfg_file_path = importlib.resources.files("smc01.luigi").joinpath("luigi.cfg")
        luigi.configuration.add_config_path(str(cfg_file_path))

    dates = pd.date_range(cfg.begin, cfg.end, freq="12H")

    job_id = int(cfg.job_id)
    n_jobs = int(cfg.n_jobs)
    n_dates_per_job = (len(dates) // n_jobs) + 1
    first_date = job_id * n_dates_per_job
    last_date = (job_id + 1) * n_dates_per_job

    dates_to_work = dates[first_date:last_date]

    if cfg.local_scheduler:
        local_scheduler = True
        scheduler_url = ""
    else:
        local_scheduler = False
        scheduler_url = cfg.scheduler_url

    luigi.build(
        [InterpolatePass(time=x) for x in dates_to_work],
        workers=cfg.n_workers,
        local_scheduler=local_scheduler,
        scheduler_url=scheduler_url,
    )


if __name__ == "__main__":
    cli()
