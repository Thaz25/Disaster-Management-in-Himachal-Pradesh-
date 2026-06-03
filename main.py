"""
AI-Based Disaster Prediction & Relief Optimization System
CLI Entry Point -- Himachal Pradesh

Usage:
    python main.py generate-data     Generate synthetic training data
    python main.py train             Train all models (CNN, LSTM, RF, XGBoost)
    python main.py predict --region Kullu   Predict for a specific region
    python main.py predict --all     Predict for all HP regions
"""

import argparse
import sys
import os
import io
import yaml
from pathlib import Path

# Fix Windows console encoding — force UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        sys.stdin  = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """Print application banner."""
    banner = """
+==============================================================+
|                                                              |
|  [SAT] AI-BASED DISASTER PREDICTION SYSTEM                  |
|  [HP]  Himachal Pradesh -- Flood, Landslide & Extreme Weather|
|                                                              |
|  Models: CNN - LSTM - Random Forest - XGBoost                |
|  Ensemble: Weighted Multi-Model Collaboration                |
|                                                              |
+==============================================================+
    """
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        print(f"{Fore.CYAN}{Style.BRIGHT}{banner}{Style.RESET_ALL}")
    except (ImportError, Exception):
        print(banner)


def cmd_generate_data(args):
    """Generate synthetic training data."""
    from src.data_collection.synthetic_generator import SyntheticGenerator
    generator = SyntheticGenerator()
    generator.generate_all()
    print("\n✅ Synthetic data generation complete!")
    print("   You can now run: python main.py train")


def cmd_train(args):
    """Train all models."""
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    results = pipeline.run(generate_data=not args.skip_datagen)
    print("\n✅ Training complete! You can now run predictions:")
    print("   python main.py predict --region Kullu")
    print("   python main.py predict --all")


def cmd_predict(args):
    """Run prediction pipeline."""
    from src.pipeline.prediction_pipeline import PredictionPipeline
    pipeline = PredictionPipeline()

    # Load trained models
    pipeline.load_models()

    if args.all:
        results = pipeline.predict_all_regions()
    elif args.region:
        # Validate region name
        valid_regions = list(pipeline.config["regions"].keys())
        if args.region not in valid_regions:
            print(f"\n[ERROR] Unknown region: '{args.region}'")
            print(f"   Valid regions: {', '.join(valid_regions)}")
            sys.exit(1)

        result = pipeline.predict_region(args.region)

        # Generate map for single region
        from src.visualization.map_visualizer import MapVisualizer
        from src.visualization.report_generator import ReportGenerator

        map_viz = MapVisualizer(pipeline.config)
        map_viz.create_map(
            {args.region: result["ensemble"]},
            {args.region: result["relief"]},
            {args.region: result["route"]}
        )

        report_gen = ReportGenerator(pipeline.config)
        report_gen.generate_html_report({args.region: result})
    else:
        print("\n[ERROR] Specify --region <name> or --all")
        print("   Example: python main.py predict --region Kullu")
        sys.exit(1)

    print("\n[OK] Prediction complete! Check the outputs/ directory.")


def cmd_demo(args):
    """Interactive demo mode — enter region names and see relief predictions."""
    from src.pipeline.prediction_pipeline import PredictionPipeline
    from src.visualization.map_visualizer import MapVisualizer
    from src.visualization.report_generator import ReportGenerator

    pipeline = PredictionPipeline()
    valid_regions = list(pipeline.config["regions"].keys())

    print("\n" + "=" * 65)
    print("  INTERACTIVE DISASTER PREDICTION DEMO")
    print("  Type a region name to predict disaster risk & relief strategy")
    print("=" * 65)
    print(f"\n  Available regions: {', '.join(valid_regions)}")
    print("  Type 'quit' or 'exit' to stop.\n")

    # Load trained models once
    print("  Loading all 4 trained models (CNN, LSTM, RF, XGBoost)...")
    pipeline.load_models()
    print("  Models loaded! Ready for predictions.\n")

    while True:
        try:
            sys.stdout.write("  Enter region name >> ")
            sys.stdout.flush()
            region = sys.stdin.readline().strip()
            if not region and region is not None:
                # empty read can mean EOF on some terminals
                continue
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting demo. Goodbye!")
            break

        if not region:
            continue

        if region.lower() in ("quit", "exit", "q"):
            print("\n  Exiting demo. Goodbye!")
            break

        # Try to match case-insensitively
        matched = None
        for r in valid_regions:
            if r.lower() == region.lower():
                matched = r
                break

        if matched is None:
            print(f"\n  [ERROR] Unknown region: '{region}'")
            print(f"  Available: {', '.join(valid_regions)}")
            print()
            continue

        print(f"\n  Analyzing satellite data & running models for {matched}...\n")

        # Run full prediction
        result = pipeline.predict_region(matched)

        # Generate map and report for this region
        map_viz = MapVisualizer(pipeline.config)
        map_viz.create_map(
            {matched: result["ensemble"]},
            {matched: result["relief"]},
            {matched: result["route"]}
        )

        report_gen = ReportGenerator(pipeline.config)
        report_gen.generate_html_report({matched: result})

        print(f"\n  [OK] Map saved to outputs/disaster_map.html")
        print(f"  [OK] Report saved to outputs/report.html")
        print(f"\n  {'=' * 55}")
        print(f"  Enter another region or type 'quit' to exit.\n")


def main():
    """Main CLI entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="AI Disaster Prediction System -- Himachal Pradesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate-data              Create synthetic training data
  python main.py train                      Train all 4 models
  python main.py predict --region Kullu     Predict for Kullu
  python main.py predict --all              Predict for all 8 HP regions
  python main.py demo                       Interactive mode (type region names)
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-data
    gen_parser = subparsers.add_parser("generate-data",
                                        help="Generate synthetic training data")

    # train
    train_parser = subparsers.add_parser("train", help="Train all models")
    train_parser.add_argument("--skip-datagen", action="store_true",
                              dest="skip_datagen",
                              help="Skip synthetic data generation (use existing)")

    # predict
    predict_parser = subparsers.add_parser("predict",
                                            help="Run prediction pipeline")
    predict_parser.add_argument("--region", type=str,
                                help="Region name (e.g., Kullu, Shimla)")
    predict_parser.add_argument("--all", action="store_true",
                                help="Predict for all regions")

    # demo (interactive mode)
    demo_parser = subparsers.add_parser("demo",
                                         help="Interactive demo: type region names, see relief predictions")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n  Quick start:")
        print("   1. python main.py generate-data")
        print("   2. python main.py train")
        print("   3. python main.py demo        <-- interactive mode for demo")
        sys.exit(0)

    commands = {
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "predict": cmd_predict,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
