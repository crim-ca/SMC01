# SMC01

SMC01 is a CRIM project realized under a Grants and Contributions (GnC) agreement with the
Meteorological Service of Canada.
The topic of the project was post-processing of weather forecast. More precisely, it studied ways
to make this post-processing integrate new information quicker, so that it 
performs better under model changes (add station, change resolution, etc).
The strategy to achieve this was to user Transformers to perform the post-processing,
where the attention mechanism is applied to the spatial axis.
We expect that spatial attention will allow the model to reuse information from 
station to station and allow an easier integration of new stations as a consequence.

The Numerical Weather Projection (NWP) model studied for this project was Global Deterministic Prediction System (GDPS). 
The model change studied for this project was the addition of a new station.

This package contains code to 
1) gather [METAR](https://www.aviationweather.gov/metar) observations to a [MongoDB](https://www.mongodb.com/) database
2) interpolate GDPS to these observations
3) train various models using this data
    - Model Output Statistics (MOS)
    - Transformer
4) Perform inference using the different models to compare their performance.

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) according to the version of the CUDA Runtime 
you are running. On CRIM infrastructure, PyTorch is installed with
```shell
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
```

Clone this repository 
```shell
git clone <TBD>
```
and install it with 
```shell
cd smc01
pip install .
```

## Data preparation

This package uses [Luigi](https://luigi.readthedocs.io/en/stable/) to orchestrate the 
data preparation.

Note that the Luigi pipelines of this package are idempotent.
That is, if a command goes wrong, Luigi should be able to run again and pickup where it left off.

### Configuration

First, the pipeline needs to be configured to better fit in the environment.
1) Copy the `smc01/luigi/luigi.cfg` file somewhere else on the system to modify it.
2) Modify the `mongo_uri` and `database_name` to point to a MongoDB database that will 
be used to store the observations.
3) Modify the `gdps_path` to point to the GDPS data.
4) Modify the `interpolated_steps_path` and `interpolated_pass_path` to a location
that will store the interpolated dataset.

### Fetch the observations

Run 
```shell
LUIGI_CONFIG_PATH=<PATH_TO_CFG> luigi --module smc01.luigi.luigi FetchMetar --workers <N_WORKERS> --local-scheduler
```
to fetch the METAR observations.

### Interpolate one pass

Run 
```shell
LUIGI_CONFIG_PATH=<PATH_TO_CFG> luigi --module smc01.luigi.luigi InterpolatePass --time 2021-01-01T00 
```
to interpolate the `2021-01-01 00:00` model run. 
This needs to be repeated for all model runs.
See [Data preparation](#data-preparation) down below for suggestions on how to parallelize this.

### Interpolate multiple passes in parallel

The package comes with a `smc01_luigi` script that wraps the pass interpolation to 
make it easier to parallelize.
Use
```shell
LUIGI_CONFIG_PATH=<PATH_TO_CFG> smc01_luigi job_id=0 n_jobs=100 n_workers=<N_WORKERS> local_scheduler=True
```
to prepare the first 100th of the dataset.
Note that it is advisable to spread the command over a minimal number of jobs.
For instance, running with `n_jobs=1` and `job_id=0` could lead to difficulties dues to a large number of pending tasks.

### Using the Luigi Central Scheduler

The Luigi scheduler provides a monitoring interface that is useful to see how the 
data preparation is doing. Instructions to launch one are
found [here](https://luigi.readthedocs.io/en/stable/central_scheduler.html#the-luigid-server).
Then, use
```shell
smc01_luigi job_id=0 n_jobs=100 n_workers=<N_WORKERS> local_scheduler=False scheduler_url=<SCHEDULER_URL>
```
to launch a data preparation job using the central scheduler.

## Model training

This package uses [Hydra](https://hydra.cc/docs/intro/) to configure the different experiments.
For any of the experiments below, you can type
`smc01_train <experiment options...> --cfg job` to see a list of available options.
You can change any such options using overrides:
```shell
smc01_train experiment=attention_gdps_metar experiment.optimizer.lr=1e-4
```
To change the default values inside the package, look for the `conf` directories in the various sub-packages.
Notably, `smc01/postprocessing/conf` contains the configuration options for the model training.


### MOS

To train a MOS model, use
```shell
smc01_train experiment=mos_gdps_metar experiment.dataset.dataset_dir=<PATH_TO_DATASET>
```
where `<PATH_TO_DATASET>` is the location of the `interpolated_pass_path` directory as configure 
in the data preparation pipeline.

Note that this MOS model is based on a rolling window of individual models, so its 
aggregation is slightly more complex that usual.
For more information, see the presentation in
[`docs/MasterclassPostTraitement.pdf`](docs/MasterclassPostTraitement.pdf).
The YouTube MasterClass Video 
[*Améliorer les prévisions de température à l'aide de transformers*](https://www.youtube.com/watch?v=7ROt3r04d8I)
of the corresponding presentation is also available.

Use 
```shell
smc01_train experiment=mos_gdps_metar --cfg job
```
top list all options available for the MOS model.


### Transformer

To train a transformer, run the following command, where `<PATH_TO_DATASET>` is replaced with
the `interpolated_pass_path` directory from the data preparation section.
```shell
smc01_train experiment=attention_gdps_metar experiment.dataset.dataset_dir=<PATH_TO_DATASET>
```
It is necessary to run this command from a machine equipped with a GPU. 
Running this from CPU will take orders of magnitude longer.

### Raw Model

To test how the model performs on the same dataset without postprocessing, use
```shell
smc01_test experiment=raw_model_gdps_metar
```

*Note*: you can also use the `smc01_test` command to run the test set on a pretrained 
model. Use the `checkpoint_path` configuration key to pass a checkpoint. For example:

```shell
smc01_test experiment=attention_gdps_metar checkpoint_path=<PATH_TO_CKPT>
```

#### Train on different subsets of station

To train a transformer on the **reference set**, use
```shell
smc01_train experiment=attention_gdps_metar experiment/dataset=gdps_metar_step_reference experiment.dataset.dataset_dir=<PATH_TO_DATASET> 
```

When this command is run, an `output` directory is generated, containing all the 
training outputs, which include the model checkpoint.
The output directory is displayed in the console output in this way:
```log
[2023-03-06 10:55:49,696][smc01.postprocessing.cli.train][INFO] - Working directory: <WORKING_DIRECTORY>
```
Make note of that directory.

#### Fine-tuning

To finetune a transformer on the **bootstrap set**, use
```shell
smc01_train experiment=attention_gdps_metar_finetune experiment.checkpoint_path=<REFERENCE_SET_TRAINING_DIR>
```
Make sure that the training on the reference set is over before you launch the fine-tuning.

#### Where to go from here

Look at the configuration of the various experiments and play with the values.
Notably, you can change the `experiment.split` options to change the dates on which the training and validation happen.
The parameters under `experiment.model` allow you to change the shape of the Transformer model.

You can get better experiment tracking by enabling the `mlflow` logger.
[MLFlow](https://mlflow.org/) is an experiment tracking solution.
To do so, add the `logging=full` override to your training command.
Note that this will require setting up a MLflow tracking server, 
and indicating the locating of the server using the `logging.full.mlflow.tracking_uri=<SERVER_URI>` override.
