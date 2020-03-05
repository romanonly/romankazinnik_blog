# Build hyper-parameter search with Metaflow.

**This flow loads a CSV file, select best features based on basic correlation analysis, 
runs mutluple models trainig as Metaflow branches.Metaflow versions all its runs, 
one can view all the historical runs with the Metaflow client in a Notebook
(see other Metaflow tutrial examples).**

#### Showcasing:
- Including external files with 'IncludeFile'.
- Basic Metaflow Parameters.
- Running workflow branches in parallel and joining results.
- Using the Metaflow client in a Notebook.

#### Install:
1. ```python -m pip install notebook```

#### Run:
1. ```cd metaflow-folder```
2. ```python rk/search.py show```
3. ```python rk/search.py run```
4. ```python rk/search.py run --alpha 5```
5. ```jupyter-notebook rk/search.ipynb```
