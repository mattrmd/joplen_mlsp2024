#!/bin/bash
which python

# Change the urls of the submodules to HTTPS so they can be pulled without authentication
# This is necessary because HTTPS is pull-only, and cannot be used for development
bash switch_url.sh https

# set up virtual environment
conda create --prefix ./my_env python=3.10 --yes
echo -e "Virtual environment created\n\n"
eval "$(conda shell.bash hook)"
conda activate ./my_env
conda config --set env_prompt '(my_env)'

which python

# standard dependencies
pip install -r requirements.txt

# custom libraries
which python
pip install -e ./joplen
pip install --no-use-pep517 --no-build-isolation git+https://gitlab.eecs.umich.edu/mattrmd-public/gendesc/multitask-sklearn.git@57c1712b04cbffd1ea76d1073075b5dca0010e76#egg=scikit-learn
pip install git+https://gitlab.eecs.umich.edu/mattrmd-public/gendesc/mutar.git@5428bf20eb7f4818bd9386d411d27bb396a0b8e9#egg=mutar
pip install -e ./ml_utils
