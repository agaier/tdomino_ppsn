example_project
==============================

Example project structure.

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
    │
    ├── figures/            <- Figures saved by scripts or notebooks.
    │
    ├── notebooks/          <- Jupyter notebooks. Naming convention is a short `-` delimited 
    │                         description, a number (for ordering), and the creator's initials,
    │                        e.g. `initial-data-exploration-01-hg`.
    │
    ├── output/             <- Manipulated data, logs, etc.
    │
    ├── tests/              <- Unit tests.
    │
    ├── exampleproject/     <- Python module with source code of this project.
    │
    ├── environment.yml     <- conda virtual environment definition file.
    │
    ├── LICENSE
    │
    ├── Makefile            <- Makefile with commands like `make environment`
    │
    ├── README.md           <- The top-level README for developers using this project.
    │
    └── tox.ini             <- tox file with settings for running tox; see tox.testrun.org


--------


Set up
------------

Install the virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ conda activate tdomino
```

Install `bbq` (PyRibs boilerplate) in the virtual environment:

```bash
$ cd bbq; pip install -e .; cd ../
```

Install `tdomino` in the virtual environment:

```bash
$ pip install -e .
```

Test that everything runs

```bash
$ python test_run.py
```