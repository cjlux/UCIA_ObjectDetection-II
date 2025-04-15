import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import os
    return Path, mo, os


@app.cell
def _():
    VER = 'v1.2'
    return (VER,)


@app.cell
def _(Path, VER):
    results_dir = Path('./Training/Results/')
    out_file = Path(results_dir, f"processed_res-{VER}.txt")
    return out_file, results_dir


@app.cell
def _(out_file):
    out_file
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
