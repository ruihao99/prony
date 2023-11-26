prony
==============================

Spectral function decomposition using time domain prony fitting. See [this JCP article](https://doi.org/10.1063/5.0095961) for details. 
The original author and developer of this package is **czh**. 
I create this repository for myself, to improve the reusablity (hopefully) of the [original `prony` code](https://git.lug.ustc.edu.cn/czh123/moscal2.0). 

Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. TODO, non existing yet.
    │
    ├── notebooks/          <- Jupyter notebooks. Examples on how to use the package
    │
    ├── tests/              <- Unit tests. TODO, non existing yet. 
    │
    ├── prony/              <- Python module with source code of this project.
    │
    ├── julia/              <- Julia module. Somewhat faster, especially useful if you are dealing with large Hankel matrix.
    │
    ├── LICENSE
    │
    ├── README.md           <- The top-level README for developers using this project.
    │


--------
Credits:

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</p>

The original `prony` code was developed and written by **czh** and it is available as an subproject of the [`moscal2.0`](https://git.lug.ustc.edu.cn/czh123/moscal2.0) package.


Set up
------------

Easiy install `prony` to your python environment:

```bash
$ pip install --editable .
```
