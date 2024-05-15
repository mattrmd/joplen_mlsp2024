# General Descriptors Data

## About

This git repo contains the data to be used in the General Descriptors paper.
It includes all necessary preprocessing, and is tracked using DVC.
Once the General Descriptors paper is released, this data will be made public and we may transition to a different method than DVC.

Importing as a git submodule allows for easy and straightforward dataset reuse in our lab's future papers.

## Running Preprocessing

Assuming that you're in the current directory, create a virtual environment and install the dependencies using:

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

To run the preprocessing, use:

```bash
cd src
papermill data_preprocessing.ipynb data_preprocessing.ipynb
nb-clean clean data_preprocessing.ipynb
```

`nb-clean` removes metadata that interferes with version control.

## Info

Standard runtime is about 1 minute and 30 seconds on a consumer-level desktop:

| Attribute | Value                      |
|-----------|----------------------------|
| CPU       | Intel i9-11900KF @ 3.5GHz  |
| RAM       | 2x32GB DDR4 3600 MHz       |
| Memory    | 1TB M.2 NVMe @ 33MHz       |
| OS        | Ubuntu 20.04               |
| Python    | 3.10                       |
