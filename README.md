# rivertopo
Tools to help combine river profile data and DEMs

## Installation
A Python 3 environment is required. A suitable Conda environment (here called
"rivertopo") can be created with:

```
conda env create -n rivertopo -f environment.yml
```

For now, the tools support editable installation using `pip`. To install the
tools this way, use the following command in the root directory:

```
pip install -e .
```

Test that everything works by calling `pytest` from the root directory:

```
pytest
```
