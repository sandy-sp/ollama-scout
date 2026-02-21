"""
interactive.py - Interactive guided session for ollama-scout.
"""
import os
import sys

from rich.panel import Panel
from rich.text import Text

from .benchmark import benchmark_pulled_models
from .display import (
    console,
    print_benchmark,
    print_error,
    print_footer,
    print_hardware_summary,
    print_info,
    print_legend,
    print_model_comparison,
    print_recommendations_flat,
    print_recommendations_grouped,
    print_success,
    spinner,
)
from .exporter import export_markdown
from .hardware import detect_hardware
from .ollama_api import (
    check_ollama_installed,
    fetch_ollama_models,
    get_fallback_models,
    get_pulled_models,
    pull_model,
)
from .recommender import get_recommendations, group_by_use_case

USE_CASE_MENU = {
    "1": ("all", "All categories", "Show all compatible models across every use case"),
    "2": ("coding", "💻 Coding", "Code generation, completion, and debugging"),
    "3": ("reasoning", "🧠 Reasoning / Analysis", "Complex reasoning, math, and analysis"),
    "4": ("chat", "💬 Chat / General use", "Conversational AI and general-purpose tasks"),
}

TOP_N_MENU = {
    "1": 5,
    "2": 10,
    "3": 15,
    "4": 20,
}


def _sep():
    """Print a dim separator line."""
    console.rule(style="bright_black")


class InteractiveSession:
    """Guided interactive session for non-developer users."""

    def run(self):
        try:
            self._run_steps()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)

    def _run_steps(self):
        # Check Ollama before first step
        ollama_installed, ollama_version = check_ollama_installed()

        # Step 1 — Welcome
        self._step_welcome(ollama_installed, ollama_version)
        _sep()

        # Step 2 — Hardware scan
        hw = self._step_hardware_scan()
        _sep()

        # Step 3 — Connection check
        models = self._step_fetch_models()

        # Detect pulled models
        pulled = get_pulled_models()
        if pulled:
            names = ", ".join(pulled[:5])
            suffix = "..." if len(pulled) > 5 else ""
            print_info(
                f"Detected [green]{len(pulled)}[/green] "
                f"already-pulled model(s): {names}{suffix}"
            )
        _sep()

        # Step 4 — Use case selection
        use_case = self._step_use_case()
        _sep()

        # Step 5 — Results count
        top_n = self._step_results_count()

        # Step 6 — Show recommendations
        recs = get_recommendations(
            models=models, hw=hw,
            use_case_filter=use_case,
            pulled_models=pulled,
            top_n=top_n,
        )
        if not recs:
            print_error("No compatible models found for your hardware.")
            print_footer()
            return

        total = len(recs)
        label = use_case.capitalize() if use_case != "all" else "all categories"
        console.print()
        console.print(
            f"[dim]Found [bold]{total}[/bold] compatible model(s) for {label}.[/dim]"
        )
        console.print()

        if use_case != "all":
            print_recommendations_flat(recs)
        else:
            grouped = group_by_use_case(recs)
            print_recommendations_grouped(grouped, pulled)
        print_legend()
        _sep()

        # Step 6b — Compare prompt
        self._step_compare(models, hw, pulled)
        _sep()

        # Step 7 — Benchmark prompt
        self._step_benchmark(recs, hw, pulled)
        _sep()

        # Step 8 — Export prompt
        self._step_export(hw, recs)
        _sep()

        # Step 9 — Pull prompt (only if Ollama is installed)
        pulled_model_name = None
        if ollama_installed:
            pulled_model_name = self._step_pull(recs)
        else:
            console.print("[dim]Skipping pull — Ollama is not installed.[/dim]")

        # Step 10 — Exit
        console.print()
        if pulled_model_name:
            console.print(
                f"[bold green]All done![/bold green] Run your model:\n"
                f"  [cyan]ollama run {pulled_model_name}[/cyan]"
            )
        else:
            console.print(
                "[dim]Tip: Run [bold]ollama-scout --help[/bold] "
                "to see all available options.[/dim]"
            )
        print_footer()

    @staticmethod
    def _step_welcome(ollama_installed: bool, ollama_version: str):
        from scout import __version__
        banner = Text()
        banner.append("🔭 ollama", style="bold cyan")
        banner.append("-scout", style="bold white")
        banner.append(f"  v{__version__}", style="dim cyan")
        banner.append(
            "  |  LLM Hardware Advisor  ",
            style="dim white",
        )
        banner.append("(Interactive Mode)", style="dim cyan italic")
        console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
        console.print()
        console.print(
            "[bold]Welcome![/bold] Let's find the best LLMs for your hardware."
        )
        console.print(
            "[dim]We'll scan your system, filter compatible models, "
            "and guide you through the rest.[/dim]"
        )
        console.print()
        if ollama_installed:
            console.print(f"[dim]Ollama detected: {ollama_version}[/dim]")
        else:
            console.print(
                "[yellow]Ollama not found.[/yellow] "
                "Recommendations will still work, but pulling models requires Ollama."
            )
        console.print()
        console.input(
            "[dim]Press Enter to scan your system, "
            "or Ctrl+C to exit.[/dim] "
        )
        console.print()

    def _step_hardware_scan(self):
        with spinner("Detecting GPU, CPU, and RAM configuration...") as p:
            p.add_task("")
            p.start()
            try:
                hw = detect_hardware()
            except Exception as e:
                p.stop()
                print_error(f"Hardware detection failed: {e}")
                sys.exit(1)
            p.stop()

        print_hardware_summary(hw)

        if hw.is_unified_memory:
            console.print(
                "[cyan]Apple Silicon detected.[/cyan] "
                "Unified memory is used as GPU VRAM for fast local inference."
            )
            console.print()
        elif not hw.gpus:
            console.print(
                "[yellow]No dedicated GPU detected.[/yellow] "
                "We'll recommend models optimized for CPU inference."
            )
            console.print()
        else:
            gpu_names = ", ".join(g.name for g in hw.gpus)
            console.print(
                f"[green]GPU detected:[/green] {gpu_names}. "
                "Models will run with full GPU acceleration."
            )
            console.print()

        return hw

    def _step_fetch_models(self):
        answer = console.input(
            "[bold yellow]Fetch latest models from Ollama library? "
            "(requires internet) \\[Y/n]:[/bold yellow] "
        ).strip().lower()

        if answer in ("n", "no"):
            print_info("Using built-in fallback model list.")
            models = get_fallback_models()
        else:
            with spinner("Fetching latest models...") as p:
                p.start()
                try:
                    models = fetch_ollama_models(limit=100)
                except Exception:
                    p.stop()
                    print_info(
                        "Could not reach Ollama API. "
                        "Using built-in fallback list."
                    )
                    models = get_fallback_models()
                else:
                    p.stop()

        print_info(f"Loaded [bold]{len(models)}[/bold] models for analysis.")
        console.print()
        return models

    def _step_use_case(self):
        return self._ask_use_case()

    @staticmethod
    def _ask_use_case() -> str:
        console.print("[bold]What are you mainly using this for?[/bold]")
        for key, (_, label, description) in USE_CASE_MENU.items():
            console.print(f"  [dim]{key}.[/dim] {label}  [dim]— {description}[/dim]")
        console.print()

        while True:
            choice = console.input(
                "[bold]Enter 1-4 (default: 1):[/bold] "
            ).strip()
            if choice == "":
                return "all"
            if choice in USE_CASE_MENU:
                return USE_CASE_MENU[choice][0]
            console.print("[red]Invalid choice. Please enter 1-4.[/red]")

    def _step_results_count(self):
        return self._ask_results_count()

    @staticmethod
    def _ask_results_count() -> int:
        console.print("[bold yellow]How many recommendations would you like?[/bold yellow]")
        menu_labels = ["Quick overview", "Standard (default)", "More options", "Extended list"]
        for key, n in TOP_N_MENU.items():
            label = menu_labels[int(key) - 1]
            console.print(f"  [dim]{key}.[/dim] {n:>2}  [dim]— {label}[/dim]")
        console.print()

        while True:
            choice = console.input(
                "[bold]Enter 1-4 (default: 2):[/bold] "
            ).strip()
            if choice == "":
                return 10
            if choice in TOP_N_MENU:
                return TOP_N_MENU[choice]
            console.print("[red]Please enter 1, 2, 3, or 4.[/red]")

    @staticmethod
    def _step_compare(models, hw, pulled):
        console.print()
        answer = console.input(
            "[bold yellow]Would you like to compare "
            "two models? \\[y/N]:[/bold yellow] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            return

        name1 = console.input(
            "[bold]Enter first model name:[/bold] "
        ).strip()
        name2 = console.input(
            "[bold]Enter second model name:[/bold] "
        ).strip()

        if not name1 or not name2:
            print_error("Both model names are required.")
            return

        from .recommender import _score_variant

        def _build_detail(name):
            target = name.lower()
            matched = [m for m in models if m.name.lower() == target]
            if not matched:
                matched = [
                    m for m in models
                    if m.name.lower().startswith(target)
                ]
            if not matched:
                return None
            model = matched[0]
            best_score, best_v, best_fit, best_mode = -1, None, None, None
            for v in model.tags:
                sc, fl, rm, _ = _score_variant(v, hw)
                if sc > best_score:
                    best_score, best_v, best_fit, best_mode = sc, v, fl, rm
            return {
                "name": model.name,
                "description": model.description,
                "tag": best_v.tag if best_v else None,
                "size_gb": best_v.size_gb if best_v else None,
                "param_size": best_v.param_size if best_v else None,
                "quantization": best_v.quantization if best_v else None,
                "fit_label": best_fit,
                "run_mode": best_mode,
                "score": best_score,
                "est_tps": None,
                "pulled": model.name in (pulled or []),
            }

        d1 = _build_detail(name1)
        d2 = _build_detail(name2)
        if d1 is None:
            print_error(f"Model '{name1}' not found.")
        if d2 is None:
            print_error(f"Model '{name2}' not found.")
        console.print()
        print_model_comparison(d1, d2)

    @staticmethod
    def _step_benchmark(recs, hw, pulled):
        console.print()
        answer = console.input(
            "[bold yellow]Would you like to benchmark pulled models? \\[y/N]:[/bold yellow] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            return
        if not pulled:
            console.print(
                "[dim]No models are currently pulled. "
                "Pull a model first to run real benchmarks.[/dim]"
            )
            return
        print_info("Running real benchmarks on pulled models...")
        estimates = benchmark_pulled_models(pulled, hw)
        if estimates:
            print_benchmark(estimates)
        else:
            console.print("[dim]Could not measure speeds for pulled models.[/dim]")

    @staticmethod
    def _step_export(hw, recs):
        answer = console.input(
            "[bold yellow]Save these results as a Markdown "
            "report? \\[y/N]:[/bold yellow] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            return

        path_input = console.input(
            "[bold]Save to (leave blank for current folder):[/bold] "
        ).strip()

        output_path = None
        if path_input:
            output_path = os.path.expanduser(path_input)

        grouped = group_by_use_case(recs)
        try:
            path = export_markdown(hw, grouped, output_path=output_path)
            print_success(f"Report saved to: [bold]{path}[/bold]")
        except Exception as e:
            print_error(f"Export failed: {e}")

    @staticmethod
    def _step_pull(recs) -> str | None:
        """Show pull menu and pull selected model. Returns pulled model name or None."""
        console.print()
        console.print("[bold yellow]Pull a recommended model?[/bold yellow]")
        console.print(
            "[dim]You can run any pulled model with "
            "[bold]ollama run MODEL[/bold][/dim]"
        )
        console.print()
        for i, rec in enumerate(recs[:10], 1):
            pulled_tag = (
                " [green](already pulled)[/green]"
                if rec.model.pulled else ""
            )
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
                model_name = f"{recs[idx].model.name}:{recs[idx].variant.tag}"
                print_info(f"Pulling [bold]{model_name}[/bold]...")
                try:
                    pull_model(model_name)
                    print_success(f"Successfully pulled {model_name}")
                    console.print(
                        f"[dim]Run it now: [bold]ollama run {model_name}[/bold][/dim]"
                    )
                    return model_name
                except FileNotFoundError as e:
                    print_error(str(e))
                except Exception as e:
                    print_error(f"Pull failed: {e}")
        except ValueError:
            pass
        return None
