#!/usr/bin/env python3
"""
ollama-scout - CLI tool to scan your hardware and recommend compatible Ollama LLMs.

Usage:
    python main.py                          # Full scan, grouped by use case
    python main.py --use-case coding        # Filter by use case
    python main.py --flat                   # Show flat list instead of grouped
    python main.py --export                 # Auto-export to Markdown
    python main.py --pull llama3.2:latest   # Pull a specific model
    python main.py --top 20                 # Show top N results
    python main.py --offline                # Use built-in fallback models (no network)
    python main.py --benchmark              # Show inference speed estimates
    python main.py --model deepseek-coder   # Show detail view for a model
    python main.py --config                 # Show current config
    python main.py --config-set key=value   # Update a config value
"""
import argparse
import sys

from rich.panel import Panel

from scout import __version__
from scout.benchmark import benchmark_pulled_models
from scout.config import DEFAULT_CONFIG, load_config, print_config, save_config
from scout.display import (
    console,
    print_banner,
    print_benchmark,
    print_error,
    print_footer,
    print_hardware_summary,
    print_info,
    print_legend,
    print_model_comparison,
    print_model_detail,
    print_ollama_not_installed,
    print_recommendations_flat,
    print_recommendations_grouped,
    print_success,
    prompt_export,
    prompt_pull,
    spinner,
)
from scout.exporter import export_markdown
from scout.hardware import detect_hardware
from scout.ollama_api import (
    check_ollama_installed,
    fetch_ollama_models,
    get_fallback_models,
    get_pulled_models,
    pull_model,
)
from scout.recommender import _score_variant, get_recommendations, group_by_use_case


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ollama-scout",
        description="Scan your hardware and find compatible Ollama LLMs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive guided mode (default when no args)",
    )
    parser.add_argument(
        "--use-case",
        choices=["all", "coding", "chat", "reasoning"],
        default=None,
        help="Filter recommendations by use case (default: all)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Show a flat list instead of grouped by use case",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        default=None,
        help="Automatically export results to Markdown without prompting",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom path for Markdown export (implies --export)",
    )
    parser.add_argument(
        "--pull",
        type=str,
        default=None,
        metavar="MODEL",
        help="Pull a specific model via ollama (e.g. llama3.2:latest)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Number of top recommendations to show (default: 15)",
    )
    parser.add_argument(
        "--no-pull-prompt",
        action="store_true",
        help="Skip the interactive pull prompt at the end",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Skip live API fetch and use built-in fallback model list",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=None,
        help="Show estimated inference speed for top models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Show detailed info for a specific model (e.g. deepseek-coder)",
    )
    parser.add_argument(
        "--update-models",
        action="store_true",
        help="Force-refresh the model list from Ollama API and update local cache",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        help="Compare two models side by side (e.g. --compare llama3.2 mistral)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print current configuration and exit",
    )
    parser.add_argument(
        "--config-set",
        type=str,
        default=None,
        metavar="KEY=VALUE",
        help="Set a config value (e.g. --config-set default_top_n=20)",
    )
    return parser.parse_args()


def _apply_config(args):
    """Merge config file defaults with CLI args (CLI wins)."""
    cfg = load_config()

    if args.use_case is None:
        args.use_case = cfg.get("default_use_case", "all")
    if args.top is None:
        args.top = cfg.get("default_top_n", 15)
    if args.export is None:
        args.export = cfg.get("auto_export", False)
    if args.offline is None:
        args.offline = cfg.get("offline_mode", False)
    if args.benchmark is None:
        args.benchmark = cfg.get("show_benchmark", False)

    return cfg


def main():
    args = parse_args()

    # --- Interactive mode (no arguments or -i/--interactive) ---
    if len(sys.argv) == 1 or args.interactive:
        from scout.interactive import InteractiveSession
        InteractiveSession().run()
        return

    # --- Config commands ---
    if args.config:
        print_config()
        return

    if args.config_set:
        cfg = load_config()
        if "=" not in args.config_set:
            print_error("Format: --config-set key=value")
            sys.exit(1)
        key, value = args.config_set.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_CONFIG:
            valid = ", ".join(DEFAULT_CONFIG.keys())
            print_error(f"Unknown config key: {key}. Valid keys: {valid}")
            sys.exit(1)
        # Type coerce
        expected = type(DEFAULT_CONFIG[key])
        if expected is bool:
            value = value.strip().lower() in ("true", "1", "yes")
        elif expected is int:
            value = int(value.strip())
        else:
            value = value.strip()
        cfg[key] = value
        save_config(cfg)
        print_success(f"Config updated: {key} = {value!r}")
        return

    # --- Update models cache ---
    if args.update_models:
        print_banner()
        with spinner("Fetching latest models from Ollama API...") as p:
            p.add_task("")
            p.start()
            try:
                models = fetch_ollama_models(limit=100, force_refresh=True)
            except ConnectionError as e:
                p.stop()
                print_error(str(e))
                sys.exit(1)
            p.stop()
        print_success(f"Model list updated. {len(models)} models cached.")
        return

    cfg = _apply_config(args)

    print_banner()

    # --- Ollama installation check ---
    ollama_installed, ollama_version = check_ollama_installed()
    if ollama_installed:
        print_info(f"Ollama detected: [dim]{ollama_version}[/dim]")
    else:
        print_ollama_not_installed()

    # --- Direct pull mode ---
    if args.pull:
        print_info(f"Pulling model: [bold]{args.pull}[/bold]")
        try:
            pull_model(args.pull)
            print_success(f"Successfully pulled {args.pull}")
        except FileNotFoundError as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Pull failed: {e}")
            sys.exit(1)
        return

    # --- Hardware scan ---
    with spinner("Detecting GPU, CPU, and RAM configuration...") as p:
        p.add_task("")
        p.start()
        try:
            hw = detect_hardware()
        except Exception as e:
            print_error(f"Hardware detection failed: {e}")
            sys.exit(1)
        p.stop()

    print_hardware_summary(hw)

    # --- Fetch models ---
    if args.offline:
        print_info("Offline mode — using built-in fallback model list.")
        models = get_fallback_models()
    else:
        with spinner("Fetching latest models from Ollama library API...") as p:
            p.start()
            try:
                models = fetch_ollama_models(limit=100)
            except ConnectionError as e:
                print_error(str(e))
                print_info("Falling back to built-in model list.")
                models = get_fallback_models()
            except Exception as e:
                print_error(f"Unexpected error fetching models: {e}")
                print_info("Falling back to built-in model list.")
                models = get_fallback_models()
            p.stop()

    print_info(f"Loaded [bold]{len(models)}[/bold] models for analysis.")

    # --- Get pulled models ---
    pulled = get_pulled_models()
    if pulled:
        names = ", ".join(pulled[:5])
        suffix = "..." if len(pulled) > 5 else ""
        print_info(
            f"Detected [green]{len(pulled)}[/green] "
            f"already-pulled model(s): {names}{suffix}"
        )

    # --- Single model detail view ---
    if args.model:
        target = args.model.lower()
        matched = [m for m in models if m.name.lower() == target]
        if not matched:
            # Try prefix match
            matched = [m for m in models if m.name.lower().startswith(target)]
        if not matched:
            print_error(f"Model '{args.model}' not found in loaded models.")
            print_info("Available models: " + ", ".join(sorted(set(m.name for m in models))))
            sys.exit(1)

        model = matched[0]
        variants_with_scores = []
        for variant in model.tags:
            score, fit_label, run_mode, note = _score_variant(variant, hw)
            variants_with_scores.append((variant, score, fit_label, run_mode, note))

        console.print()
        print_model_detail(model, variants_with_scores, pulled, hw)
        print_footer()
        return

    # --- Comparison mode ---
    if args.compare:
        def _find_model(name):
            target = name.lower()
            matched = [m for m in models if m.name.lower() == target]
            if not matched:
                matched = [m for m in models if m.name.lower().startswith(target)]
            return matched[0] if matched else None

        def _model_detail(name):
            model = _find_model(name)
            if model is None:
                return None
            best_score, best_variant, best_fit, best_mode = -1, None, None, None
            for variant in model.tags:
                score, fit_label, run_mode, note = _score_variant(variant, hw)
                if score > best_score:
                    best_score = score
                    best_variant = variant
                    best_fit = fit_label
                    best_mode = run_mode
            return {
                "name": model.name,
                "description": model.description,
                "tag": best_variant.tag if best_variant else None,
                "size_gb": best_variant.size_gb if best_variant else None,
                "param_size": best_variant.param_size if best_variant else None,
                "quantization": best_variant.quantization if best_variant else None,
                "fit_label": best_fit,
                "run_mode": best_mode,
                "score": best_score,
                "est_tps": None,
                "pulled": model.name in pulled,
            }

        d1 = _model_detail(args.compare[0])
        d2 = _model_detail(args.compare[1])
        if d1 is None:
            print_error(f"Model '{args.compare[0]}' not found.")
        if d2 is None:
            print_error(f"Model '{args.compare[1]}' not found.")
        console.print()
        print_model_comparison(d1, d2)
        print_footer()
        return

    # --- Recommend ---
    recs = get_recommendations(
        models=models,
        hw=hw,
        use_case_filter=args.use_case,
        pulled_models=pulled,
        top_n=args.top,
    )

    if not recs:
        print_error("No compatible models found for your hardware profile.")
        sys.exit(0)

    console.print()

    # --- Display ---
    if args.flat or args.use_case != "all":
        print_recommendations_flat(recs)
    else:
        grouped = group_by_use_case(recs)
        print_recommendations_grouped(grouped, pulled)

    print_legend()

    # --- Benchmark ---
    if args.benchmark:
        if not pulled or not ollama_installed:
            console.print(Panel(
                "No models are currently pulled.\n"
                "Pull a model with [cyan]ollama pull MODEL[/cyan] "
                "or use the pull prompt below.",
                title="[dim]Benchmark[/dim]",
                border_style="bright_black",
                padding=(0, 2),
            ))
        else:
            with spinner("Running real benchmarks on pulled models...") as p:
                p.add_task("")
                p.start()
                estimates = benchmark_pulled_models(pulled, hw)
                p.stop()
            if estimates:
                print_benchmark(estimates)
            else:
                print_info("No benchmark results available.")

    # --- Export ---
    export_dir = cfg.get("export_dir", "")
    grouped_for_export = group_by_use_case(recs)
    should_export = args.export or args.output

    if not should_export:
        should_export = prompt_export()

    if should_export:
        output_path = args.output
        if not output_path and export_dir:
            import os
            export_dir = os.path.expanduser(export_dir)
            os.makedirs(export_dir, exist_ok=True)
            output_path = None  # let exporter generate filename
            # We'll pass the dir to exporter if set
        try:
            path = export_markdown(hw, grouped_for_export, output_path=output_path)
            print_success(f"Report saved to: [bold]{path}[/bold]")
        except Exception as e:
            print_error(f"Export failed: {e}")

    # --- Pull prompt ---
    if not args.no_pull_prompt and not args.export and ollama_installed:
        model_to_pull = prompt_pull(recs)
        if model_to_pull:
            print_info(f"Pulling [bold]{model_to_pull}[/bold]...")
            try:
                pull_model(model_to_pull)
                print_success(f"Successfully pulled {model_to_pull}")
            except FileNotFoundError as e:
                print_error(str(e))
            except Exception as e:
                print_error(f"Pull failed: {e}")

    print_footer()


if __name__ == "__main__":
    main()
