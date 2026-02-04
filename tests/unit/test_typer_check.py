import pytest

typer = pytest.importorskip("typer")
app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def hello():
    print("Hello")

if __name__ == "__main__":
    app()
