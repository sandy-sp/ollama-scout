"""
display.py - Rich terminal UI for ollama-scout.
"""
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .benchmark import BenchmarkEstimate
from .hardware import HardwareProfile
from .recommender import Recommendation

console = Console()

FIT_COLORS = {
    "Excellent": "bold green",
    "Good": "bold yellow",
    "Possible": "dim yellow",
    "Too Large": "red",
}

RUN_MODE_COLORS = {
    "GPU": "cyan",
    "CPU+GPU": "yellow",
    "CPU": "dim white",
    "N/A": "red",
}

USE_CASE_ICONS = {
    "coding": "💻",
    "reasoning": "🧠",
    "chat": "💬",
}


def print_banner():
    from scout import __version__
    banner = Text()
    banner.append("🔭 ollama", style="bold cyan")
    banner.append("-scout", style="bold white")
    banner.append(f"  v{__version__}", style="dim cyan")
    banner.append("  |  LLM Hardware Advisor", style="dim white")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def print_hardware_summary(hw: HardwareProfile):
    table = Table(
        title="[bold cyan]System Hardware[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold white",
    )
    table.add_column("Component", style="dim white")
    table.add_column("Details", style="white")

    table.add_row("OS", hw.os)
    table.add_row("CPU", hw.cpu_name)
    table.add_row("Cores / Threads", f"{hw.cpu_cores} cores / {hw.cpu_threads} threads")
    table.add_row("RAM", f"{hw.ram_gb} GB")

    if hw.is_unified_memory:
        table.add_row("Memory", "[cyan]Unified (Apple Silicon) — shared GPU/CPU[/cyan]")

    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            label = "GPU" if i == 0 else f"GPU {i+1}"
            table.add_row(label, f"{gpu.name}  [cyan]({gpu.vram_gb} GB VRAM)[/cyan]")
    else:
        table.add_row("GPU", "[dim]None detected — CPU inference only[/dim]")

    console.print(table)
    console.print()


def print_recommendations_grouped(
    grouped: dict[str, list[Recommendation]],
    pulled_models: list[str],
):
    for use_case, recs in grouped.items():
        if not recs:
            continue
        icon = USE_CASE_ICONS.get(use_case, "🤖")
        title = f"{icon}  [bold white]{use_case.capitalize()} Models[/bold white]"

        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            border_style="bright_black",
            show_header=True,
            header_style="bold dim white",
            padding=(0, 1),
        )
        table.add_column("Model", style="bold white", min_width=20)
        table.add_column("Tag / Variant", style="dim white")
        table.add_column("Quant", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Params", justify="center")
        table.add_column("Fit", justify="center")
        table.add_column("Mode", justify="center")
        table.add_column("Note", style="dim", max_width=35)
        table.add_column("Status", justify="center")

        for rec in recs:
            fit_style = FIT_COLORS.get(rec.fit_label, "white")
            mode_style = RUN_MODE_COLORS.get(rec.run_mode, "white")
            status = "[green]✔ Pulled[/green]" if rec.model.pulled else "[dim]Available[/dim]"

            table.add_row(
                rec.model.name,
                rec.variant.tag,
                f"[cyan]{rec.variant.quantization}[/cyan]",
                f"[white]{rec.variant.size_gb}GB[/white]",
                f"[dim]{rec.variant.param_size}[/dim]",
                f"[{fit_style}]{rec.fit_label}[/{fit_style}]",
                f"[{mode_style}]{rec.run_mode}[/{mode_style}]",
                rec.note,
                status,
            )

        console.print(table)
        console.print()


def print_recommendations_flat(recs: list[Recommendation]):
    table = Table(
        title="[bold cyan]Recommended Models[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        show_header=True,
        header_style="bold dim white",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="bold white", min_width=20)
    table.add_column("Tag", style="dim white")
    table.add_column("Quant", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Use Cases", max_width=22)
    table.add_column("Fit", justify="center")
    table.add_column("Mode", justify="center")
    table.add_column("Status", justify="center")

    for i, rec in enumerate(recs, 1):
        fit_style = FIT_COLORS.get(rec.fit_label, "white")
        mode_style = RUN_MODE_COLORS.get(rec.run_mode, "white")
        status = "[green]✔ Pulled[/green]" if rec.model.pulled else "[dim]Available[/dim]"
        use_cases = " ".join(
            USE_CASE_ICONS.get(uc, uc) for uc in rec.model.use_cases
        )
        table.add_row(
            str(i),
            rec.model.name,
            rec.variant.tag,
            f"[cyan]{rec.variant.quantization}[/cyan]",
            f"{rec.variant.size_gb}GB",
            use_cases,
            f"[{fit_style}]{rec.fit_label}[/{fit_style}]",
            f"[{mode_style}]{rec.run_mode}[/{mode_style}]",
            status,
        )

    console.print(table)


def print_benchmark(estimates: list[BenchmarkEstimate]):
    """Print a benchmark estimation panel for top models."""
    RATING_STYLES = {
        "Fast": ("bold green", "Fast ⚡"),
        "Moderate": ("bold yellow", "Moderate 🔄"),
        "Slow": ("dim yellow", "Slow 🐢"),
    }

    table = Table(
        title="[bold cyan]Inference Speed Estimates[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold white",
        padding=(0, 1),
    )
    table.add_column("Model", style="bold white", min_width=25)
    table.add_column("Mode", justify="center")
    table.add_column("Est. Speed", justify="right")
    table.add_column("Rating", justify="center")

    for est in estimates:
        mode_style = RUN_MODE_COLORS.get(est.run_mode, "white")
        style, label = RATING_STYLES.get(est.rating, ("white", est.rating))
        table.add_row(
            est.model_name,
            f"[{mode_style}]{est.run_mode}[/{mode_style}]",
            f"{est.tokens_per_sec} t/s",
            f"[{style}]{label}[/{style}]",
        )

    console.print(table)
    console.print(
        "[dim]Estimates only. Actual performance depends on context length, "
        "system load, and model architecture.[/dim]"
    )
    console.print()


def print_model_detail(model, variants_with_scores, pulled_models, hw):
    """Print a detailed view of a single model."""

    is_pulled = model.name in pulled_models
    uc_badges = " ".join(USE_CASE_ICONS.get(uc, uc) for uc in model.use_cases)

    # Header
    header = Text()
    header.append(f"{model.name}", style="bold cyan")
    header.append(f"  {uc_badges}")
    if is_pulled:
        header.append("  [green]✔ Pulled[/green]")

    console.print(Panel(header, border_style="cyan", padding=(0, 2)))
    console.print(f"  [dim]{model.description}[/dim]")
    console.print()

    # Variants table
    table = Table(
        title="[bold white]Available Variants[/bold white]",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        show_header=True,
        header_style="bold dim white",
        padding=(0, 1),
    )
    table.add_column("Tag", style="white")
    table.add_column("Size", justify="right")
    table.add_column("Params", justify="center")
    table.add_column("Quant", justify="center")
    table.add_column("Fit", justify="center")
    table.add_column("Mode", justify="center")
    table.add_column("Note", style="dim", max_width=40)

    best_variant = None
    best_score = -1

    for variant, score, fit_label, run_mode, note in variants_with_scores:
        fit_style = FIT_COLORS.get(fit_label, "white")
        mode_style = RUN_MODE_COLORS.get(run_mode, "white")
        table.add_row(
            variant.tag,
            f"{variant.size_gb}GB",
            variant.param_size,
            f"[cyan]{variant.quantization}[/cyan]",
            f"[{fit_style}]{fit_label}[/{fit_style}]",
            f"[{mode_style}]{run_mode}[/{mode_style}]",
            note,
        )
        if score > best_score:
            best_score = score
            best_variant = variant

    console.print(table)
    console.print()

    # Pull command for best variant
    if best_variant and best_score >= 0:
        pull_cmd = f"ollama pull {model.name}:{best_variant.tag}"
        console.print(Panel(
            f"[bold]Best fit:[/bold] {model.name}:{best_variant.tag} "
            f"({best_variant.size_gb}GB, {best_variant.quantization})\n"
            f"[bold]Pull command:[/bold] [cyan]{pull_cmd}[/cyan]",
            title="[dim]Recommendation[/dim]",
            border_style="green",
            padding=(0, 2),
        ))
    else:
        console.print("[dim]No compatible variants found for your hardware.[/dim]")

    console.print()


def print_legend():
    """Print a legend panel explaining Fit labels and Run Modes."""
    legend_text = Text()
    legend_text.append("Fit Labels:  ", style="bold white")
    legend_text.append("Excellent", style="bold green")
    legend_text.append(" = fits fully in VRAM  |  ", style="dim")
    legend_text.append("Good", style="bold yellow")
    legend_text.append(" = partial CPU offload  |  ", style="dim")
    legend_text.append("Possible", style="dim yellow")
    legend_text.append(" = CPU-only, slower\n", style="dim")
    legend_text.append("Run Modes:   ", style="bold white")
    legend_text.append("GPU", style="cyan")
    legend_text.append(" = full GPU acceleration  |  ", style="dim")
    legend_text.append("CPU+GPU", style="yellow")
    legend_text.append(" = split across GPU + RAM  |  ", style="dim")
    legend_text.append("CPU", style="dim white")
    legend_text.append(" = CPU inference only", style="dim")

    console.print(Panel(
        legend_text,
        title="[dim]Legend[/dim]",
        border_style="bright_black",
        padding=(0, 1),
    ))


def print_footer():
    """Print a footer with helpful tips."""
    console.print()
    console.print(Panel(
        "[dim]Run with [bold]--help[/bold] for all options  |  "
        "[bold]--offline[/bold] to skip API fetch  |  "
        "[bold]--export[/bold] to save report[/dim]",
        border_style="bright_black",
        padding=(0, 1),
    ))
    console.print()


def prompt_export() -> bool:
    console.print()
    answer = console.input(
        "[bold yellow]Save results as Markdown report? [y/N]:[/bold yellow] "
    ).strip().lower()
    return answer in ("y", "yes")


def prompt_pull(recs: list[Recommendation]) -> str | None:
    console.print()
    console.print("[bold yellow]Auto-pull a recommended model?[/bold yellow]")
    for i, rec in enumerate(recs[:10], 1):
        pulled_tag = " [green](already pulled)[/green]" if rec.model.pulled else ""
        label = f"{rec.model.name}:{rec.variant.tag}"
        console.print(
            f"  [dim]{i}.[/dim] [white]{label}[/white]{pulled_tag}"
        )
    console.print(f"  [dim]{0}.[/dim] Skip")
    console.print()
    choice = console.input("[bold]Enter number:[/bold] ").strip()
    if choice == "0" or not choice:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < min(10, len(recs)):
            return f"{recs[idx].model.name}:{recs[idx].variant.tag}"
    except ValueError:
        pass
    return None


def spinner(message: str) -> Progress:
    p = Progress(SpinnerColumn(), TextColumn(f"[cyan]{message}[/cyan]"), transient=True)
    return p


def print_error(msg: str):
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str):
    console.print(f"[bold green]OK:[/bold green] {msg}")


def print_info(msg: str):
    console.print(f"[dim cyan]>>[/dim cyan]  {msg}")
