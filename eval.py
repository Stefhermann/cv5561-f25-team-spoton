# evaluate_pipeline.py - Comprehensive Pipeline Evaluation

import cv2
import numpy as np
from ultralytics import YOLO
from spatial_association import associate_die_card
from tracking import DiceTracker
from scoring import Scoring
import json
from datetime import datetime

def deduplicate_by_class(detections, names):
    """Keep only ONE detection per class (highest confidence)."""
    class_best = {}
    for det in detections:
        cls_name = names[det["cls"]]
        if cls_name not in class_best or det["conf"] > class_best[cls_name]["conf"]:
            class_best[cls_name] = det
    return list(class_best.values())


def process_single_video(video_path, enable_class_dedup=True, verbose=False):
    """
    Process a single video and return final scores.
    
    Returns:
        dict: {"blue": int, "red": int, "yellow": int}
    """
    model = YOLO("model/rdg_obb/weights/best.pt")
    tracker = DiceTracker()
    scorer = Scoring()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    final_scores = {"blue": 0, "red": 0, "yellow": 0}
    
    if verbose:
        print(f"Processing {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        frame_count += 1
        
        # YOLO detection with aggressive NMS
        results = model(
            frame,
            task="obb",
            conf=0.35,
            iou=0.25,
            imgsz=512,
            agnostic_nms=False,
            verbose=False  # Suppress YOLO output
        )
        r = results[0]
        
        detections = []
        for b in r.obb:
            x1, y1, x2, y2 = b.xyxy[0]
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cls": int(b.cls.item()),
                "conf": float(b.conf.item())
            })
        
        # Class-based deduplication
        if enable_class_dedup:
            detections = deduplicate_by_class(detections, r.names)
        
        # Spatial association
        associations = associate_die_card(detections, r.names)
        
        # Extract dice
        dice_inputs = []
        for det in detections:
            cls_name = r.names[det["cls"]]
            if "_" in cls_name:
                dice_inputs.append({
                    "class": cls_name,
                    "bbox": det["bbox"]
                })
        
        # Tracking and scoring
        tracked_dice = tracker.update(dice_inputs)
        final_scores = scorer.update_scores(tracked_dice, associations)
    
    cap.release()
    
    if verbose:
        print(f"  Processed {frame_count} frames")
        print(f"  Final scores: {final_scores}")
    
    return final_scores


def calculate_metrics(actual_scores, predicted_scores):
    """
    Calculate comprehensive accuracy metrics.
    
    Args:
        actual_scores: List of [blue, red, yellow] for each game
        predicted_scores: List of [blue, red, yellow] for each game
    
    Returns:
        dict: Comprehensive metrics
    """
    n_games = len(actual_scores)
    
    metrics = {
        "n_games": n_games,
        "exact_matches": 0,
        "per_player_accuracy": {"blue": 0, "red": 0, "yellow": 0},
        "per_player_mae": {"blue": 0.0, "red": 0.0, "yellow": 0.0},
        "per_player_mse": {"blue": 0.0, "red": 0.0, "yellow": 0.0},
        "total_mae": 0.0,
        "total_mse": 0.0,
        "game_results": [],
        "error_distribution": []
    }
    
    colors = ["blue", "red", "yellow"]
    color_errors = {"blue": [], "red": [], "yellow": []}
    
    for i, (actual, predicted) in enumerate(zip(actual_scores, predicted_scores)):
        game_result = {
            "game": i + 1,
            "actual": dict(zip(colors, actual)),
            "predicted": dict(zip(colors, predicted)),
            "exact_match": False,
            "errors": {}
        }
        
        # Check exact match (all three players correct)
        if actual == predicted:
            metrics["exact_matches"] += 1
            game_result["exact_match"] = True
        
        # Per-player analysis
        for j, color in enumerate(colors):
            error = abs(actual[j] - predicted[j])
            color_errors[color].append(error)
            game_result["errors"][color] = predicted[j] - actual[j]  # Signed error
            
            # Perfect player score
            if actual[j] == predicted[j]:
                metrics["per_player_accuracy"][color] += 1
        
        metrics["game_results"].append(game_result)
        
        # Total error for this game
        total_error = sum(abs(actual[j] - predicted[j]) for j in range(3))
        metrics["error_distribution"].append(total_error)
    
    # Calculate per-player metrics
    for color in colors:
        errors = color_errors[color]
        metrics["per_player_accuracy"][color] = (metrics["per_player_accuracy"][color] / n_games) * 100
        metrics["per_player_mae"][color] = np.mean(errors)
        metrics["per_player_mse"][color] = np.mean([e**2 for e in errors])
    
    # Calculate overall metrics
    all_errors = [e for errors in color_errors.values() for e in errors]
    metrics["total_mae"] = np.mean(all_errors)
    metrics["total_mse"] = np.mean([e**2 for e in all_errors])
    metrics["total_rmse"] = np.sqrt(metrics["total_mse"])
    
    # Overall accuracy (all predictions correct)
    metrics["exact_match_accuracy"] = (metrics["exact_matches"] / n_games) * 100
    
    # Calculate per-score-level accuracy (within N points)
    metrics["within_tolerance"] = {
        "0_points": metrics["exact_matches"],
        "2_points": sum(1 for e in metrics["error_distribution"] if e <= 2),
        "5_points": sum(1 for e in metrics["error_distribution"] if e <= 5),
        "10_points": sum(1 for e in metrics["error_distribution"] if e <= 10)
    }
    
    return metrics


def print_evaluation_report(metrics):
    """Print a comprehensive evaluation report."""
    
    print("\n" + "="*70)
    print("PIPELINE EVALUATION REPORT")
    print("="*70)
    
    # Overall Accuracy
    print(f"\n📊 OVERALL ACCURACY")
    print(f"{'─'*70}")
    print(f"Games Evaluated: {metrics['n_games']}")
    print(f"Exact Matches (all 3 players correct): {metrics['exact_matches']}/{metrics['n_games']} ({metrics['exact_match_accuracy']:.1f}%)")
    print(f"Total MAE (Mean Absolute Error): {metrics['total_mae']:.2f} points")
    print(f"Total RMSE (Root Mean Square Error): {metrics['total_rmse']:.2f} points")
    
    # Tolerance Levels
    print(f"\n🎯 ACCURACY BY TOLERANCE")
    print(f"{'─'*70}")
    for tolerance, count in metrics["within_tolerance"].items():
        pct = (count / metrics["n_games"]) * 100
        tolerance_label = tolerance.replace("_", " ").title()
        print(f"Within {tolerance_label}: {count}/{metrics['n_games']} ({pct:.1f}%)")
    
    # Per-Player Accuracy
    print(f"\n🎲 PER-PLAYER ACCURACY")
    print(f"{'─'*70}")
    print(f"{'Player':<10} {'Exact Match':<15} {'MAE':<10} {'RMSE':<10}")
    print(f"{'─'*50}")
    for color in ["blue", "red", "yellow"]:
        acc = metrics["per_player_accuracy"][color]
        mae = metrics["per_player_mae"][color]
        rmse = np.sqrt(metrics["per_player_mse"][color])
        print(f"{color.capitalize():<10} {acc:>6.1f}%{'':<8} {mae:>6.2f}{'':<4} {rmse:>6.2f}")
    
    # Game-by-Game Results
    print(f"\n📋 GAME-BY-GAME RESULTS")
    print(f"{'─'*70}")
    print(f"{'Game':<6} {'Actual (B/R/Y)':<20} {'Predicted (B/R/Y)':<20} {'Status':<10}")
    print(f"{'─'*70}")
    
    for result in metrics["game_results"]:
        game_num = result["game"]
        actual = result["actual"]
        predicted = result["predicted"]
        
        actual_str = f"{actual['blue']}/{actual['red']}/{actual['yellow']}"
        predicted_str = f"{predicted['blue']}/{predicted['red']}/{predicted['yellow']}"
        status = "✅ PASS" if result["exact_match"] else "❌ FAIL"
        
        print(f"{game_num:<6} {actual_str:<20} {predicted_str:<20} {status}")
        
        # Show individual player errors if not exact match
        if not result["exact_match"]:
            errors = result["errors"]
            error_parts = []
            for color in ["blue", "red", "yellow"]:
                err = errors[color]
                if err != 0:
                    sign = "+" if err > 0 else ""
                    error_parts.append(f"{color[0].upper()}{sign}{err}")
            if error_parts:
                print(f"{'':6} Errors: {', '.join(error_parts)}")
    
    # Error Analysis
    print(f"\n📈 ERROR ANALYSIS")
    print(f"{'─'*70}")
    
    errors = metrics["error_distribution"]
    print(f"Min Total Error: {min(errors)} points")
    print(f"Max Total Error: {max(errors)} points")
    print(f"Mean Total Error: {np.mean(errors):.2f} points")
    print(f"Median Total Error: {np.median(errors):.2f} points")
    print(f"Std Dev: {np.std(errors):.2f} points")
    
    # Grade Assignment
    print(f"\n🎓 OVERALL GRADE")
    print(f"{'─'*70}")
    
    acc = metrics["exact_match_accuracy"]
    mae = metrics["total_mae"]
    
    if acc >= 90 and mae <= 2:
        grade = "A+ (Excellent)"
    elif acc >= 80 and mae <= 3:
        grade = "A (Very Good)"
    elif acc >= 70 and mae <= 5:
        grade = "B (Good)"
    elif acc >= 60 and mae <= 7:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"
    
    print(f"Grade: {grade}")
    print(f"  - Exact Match Accuracy: {acc:.1f}%")
    print(f"  - Mean Absolute Error: {mae:.2f} points")
    
    print(f"\n{'='*70}\n")


def save_results(metrics, output_file="evaluation_results.json"):
    """Save evaluation results to JSON file."""
    
    # Prepare serializable version
    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "n_games": metrics["n_games"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "total_mae": metrics["total_mae"],
            "total_rmse": metrics["total_rmse"]
        },
        "per_player": {
            color: {
                "accuracy": metrics["per_player_accuracy"][color],
                "mae": metrics["per_player_mae"][color],
                "rmse": np.sqrt(metrics["per_player_mse"][color])
            }
            for color in ["blue", "red", "yellow"]
        },
        "game_results": metrics["game_results"],
        "within_tolerance": metrics["within_tolerance"]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    
    # Ground truth scores: [blue, red, yellow] for each game
    actual_scores = [
        [44, 0, 0],    # test_video_4
        # [20, 12, 6],    # test_video_2
        # [0, 24, 10],    # test_video_3
        # [18, 0, 18],    # test_video_4
        # [6, 15, 12],    # test_video_5
        # [24, 20, 8],    # test_video_6
        # [10, 6, 20],    # test_video_7
        # [15, 18, 0],    # test_video_8
    ]
    
    # Process all videos
    print("\n🎬 Starting Pipeline Evaluation...")
    print("="*70)
    
    predicted_scores = []
    
    # for i in range(1, 2):
    video_path = "./test_data/test_video_4.mp4"
        # print(f"\nProcessing test_video_{i}.mp4...")
        
    try:
        scores = process_single_video(video_path, enable_class_dedup=True, verbose=True)
        predicted_scores.append([scores["blue"], scores["red"], scores["yellow"]])
    except Exception as e:
        print(f"  ❌ Error processing video {i}: {e}")
        predicted_scores.append([0, 0, 0])  # Default to zeros on error
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(actual_scores, predicted_scores)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save results
    save_results(metrics)
    
    # Return for further analysis
    return metrics, predicted_scores


if __name__ == "__main__":
    metrics, predicted_scores = main()