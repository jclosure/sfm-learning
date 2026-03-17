from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .export import export_point_cloud
from .pipeline import run_pipeline

app = typer.Typer(help="Incremental SfM learning pipeline")
console = Console()


@app.command()
def reconstruct(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/sparse.ply"), "--output", "-o"),
    detector: str = typer.Option("sift", help="sift or orb"),
    no_ba: bool = typer.Option(False, help="Disable bundle adjustment"),
):
    """Run full reconstruction and export sparse point cloud."""
    result = run_pipeline(image_dir, detector=detector, run_ba=not no_ba)
    out = export_point_cloud(result.reconstruction, output)
    console.print(f"[green]Done[/green] poses={len(result.reconstruction.poses)} points={len(result.reconstruction.points)}")
    console.print(f"Exported: {out}")


if __name__ == "__main__":
    app()
