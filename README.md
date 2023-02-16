# MLflow tutorial

## MLflow terms

Artifact - output file like plot, plickle, data file, etc.  

Flavors - modules/frameworks that allow mlflow to understand model (e.i. model library like sklearn).  

Model signature - defines the schema of model's input and output.  

Input example - as name says; also stored as artifact.  

MLmodel file - yaml that outlines multiple flavours the model can be viewed, model signature and input example.  

### Usage

There are few options how to store artifacts (logs, models, plots, ect.).
Default these will be stored in `mlruns` folder.
To run UI use command:

```console
mlflow ui
```

You can also use sqlite for logs.
First run mlflow server with command:

```console
mlflow server --backend-store-uri sqlite:///mlflow.sqlite  --default-artifact-root ./artifacts
```

Add line to your python code:

```python
mlflow.set_tracking_uri('http://127.0.0.1:5000')
```

### MLflow project

MLflow Projects are just a convention for organizing and describing your code to let other data scientists (or automated tools) run it.
Each project is simply a directory of files, or a Git repository, containing your code.

*MLproject* file, which is a YAML formatted text file. Each project can specify several properties:

- Name - project name.
- Entry Points - Commands that can be run within the project, and information about their parameters.
- Environment - The software environment that should be used to execute project entry points. This includes all library dependencies required by the project code.

Run project using:

- *MLproject* file from current dir,
- local venv (by default conda must be provided),
- run name set to *tutorial*:

```console
mlflow run . --env-manager local --run-name tutorial
```

## mypy

Library checking code as if python was static language.  

`mypy files` --ignore-missing-imports - won't prompt errors about missing library stubs.  

```console
mypy --ignore-missing-imports --incremental --install-types --show-error-codes --pretty dir/file.py - example command with pretty listing  
```

mypy.ini - file where you can store mypy arguments.  

## Black

Command arguments can be stored in pyproject.toml file under secion [tool.black]. It could be extentions of files that should be processed or excluded, also python versions, line-lenghts and few more.  

## isort

Sorts imports.  

isort file_name.py - command calling.  

## flake8

Style guide enforcement tool.  

```console
flake8 --extend-ignore=E51 dir/ - standard E51 won't be included.  
--exclude=\*.pyc - don't analyze \*.pyc files.  
```

## absolufy-imports

Convert relative imports to absolute imports.  

## Coverage

Measures code coverage, typically during test execution.

## Typer

Library to write CLI app.  

```python
def main(name: str = typer.Argument("World", help="Who to greet"):
    ...

typer.run(main)
```
