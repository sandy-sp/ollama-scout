"""
interactive.py - Interactive guided session for ollama-scout.
"""
import os
import sys

from rich.panel import Panel
from rich.text import Text

from .benchmark import benchmark_pulled_models, get_benchmark_estimates
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
    fetch_ollama_models,
    get_fallback_models,
    get_pulled_models,
    pull_model,
)
from .recommender import get_recommendations, group_by_use_case

USE_CASE_MENU = {
    "1": ("all", "All categories"),
    "2": ("coding", "💻 Coding"),
    "3": ("reasoning", "🧠 Reasoning / Analysis"),
    "4": ("chat", "💬 Chat / General use"),
}

VALID_TOP_N = {5, 10, 15, 20}


class InteractiveSession:
    """Guided interactive session for non-developer users."""

    def run(self):
        try:
            self._run_steps()
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)

    def _run_steps(self):
        # Step 1 — Welcome
        self._step_welcome()

        # Step 2 — Hardware scan
        hw = self._step_hardware_scan()

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

        # Step 4 — Use case selection
        use_case = self._step_use_case()

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

        console.print()
        if use_case != "all":
            print_recommendations_flat(recs)
        else:
            grouped = group_by_use_case(recs)
            print_recommendations_grouped(grouped, pulled)
        print_legend()

        # Step 6b — Compare prompt
        self._step_compare(models, hw, pulled)

        # Step 7 — Benchmark prompt
        self._step_benchmark(recs, hw, pulled)

        # Step 8 — Export prompt
        self._step_export(hw, recs)

        # Step 9 — Pull prompt
        self._step_pull(recs)

        # Step 10 — Exit
        console.print()
        console.print(
            "[dim]Done! Run [bold]ollama-scout --help[/bold] "
            "to see all available options.[/dim]"
        )
        print_footer()

    def _step_welcome(self):
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
            "[bold]Welcome![/bold] Let's find the best LLMs "
            "for your hardware."
        )
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

        if not hw.gpus:
            console.print(
                "[yellow]No dedicated GPU detected.[/yellow] "
                "We'll recommend models optimized for CPU inference."
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
        for key, (_, label) in USE_CASE_MENU.items():
            console.print(f"  [dim]{key}.[/dim] {label}")
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
        while True:
            choice = console.input(
                "[bold yellow]How many recommendations? "
                "\\[5/10/15/20] (default: 10):[/bold yellow] "
            ).strip()
            if choice == "":
                return 10
            try:
                n = int(choice)
                if n in VALID_TOP_N:
                    return n
            except ValueError:
                pass
            console.print("[red]Please enter 5, 10, 15, or 20.[/red]")

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

        from .benchmark import estimate_speed
        from .recommender import Recommendation, _score_variant

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
            est_tps = None
            if best_v:
                rec = Recommendation(
                    model=model, variant=best_v,
                    score=best_score, run_mode=best_mode,
                    fit_label=best_fit,
                )
                est_tps = estimate_speed(rec, hw).tokens_per_sec
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
                "est_tps": est_tps,
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
            "[bold yellow]Would you like to see estimated "
            "inference speeds? \\[y/N]:[/bold yellow] "
        ).strip().lower()
        if answer in ("y", "yes"):
            estimates = get_benchmark_estimates(recs, hw)
            if pulled:
                print_info("Running real benchmarks on pulled models...")
                real_estimates = benchmark_pulled_models(pulled, hw)
                real_names = {e.model_name.split(":")[0] for e in real_estimates}
                estimates = [
                    e for e in estimates
                    if e.model_name.split(":")[0] not in real_names
                ]
                estimates = real_estimates + estimates
            print_benchmark(estimates)

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
    def _step_pull(recs):
        console.print()
        console.print(
            "[bold yellow]Pull a recommended model?[/bold yellow]"
        )
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
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < min(10, len(recs)):
                model_name = f"{recs[idx].model.name}:{recs[idx].variant.tag}"
                print_info(f"Pulling [bold]{model_name}[/bold]...")
                try:
                    pull_model(model_name)
                    print_success(f"Successfully pulled {model_name}")
                except FileNotFoundError as e:
                    print_error(str(e))
                except Exception as e:
                    print_error(f"Pull failed: {e}")
        except ValueError:
            pass
