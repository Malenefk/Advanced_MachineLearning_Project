import os
import traceback
from losses import LOSS_REGISTRY
from train import train
from test import evaluate

SWEEP_MODELS = ["unet", "fno"]
SWEEP_LOSSES = list(LOSS_REGISTRY.keys())
SWEEP_ROOT = "sweep_results"


# Main sweep
def main():
    """Run training nad evaluation sweep for FNO, UNET with all loss function combinations
    
    Builds all lossfunction combinations for FNO and UNET model, then train and evalaute each of these combinations. 
    Saves metrices and logs.
    If one combination is failed, the failed combinations are reported into the end and next traning continiues.

    Output:
        Each model and loss function combination gets own subdirectory sweep_results/model_lossfunction:
    best_model.pt: model from epoch with lowest validation error
    last_model.pt: model from the last epoch
    history.csv: train and validation loss, loss function componenets, lr, time per epoch
    config.json: configuration, hyperparamters, total time and memory usage of GPU/CPU
    metrics.json: evluation metrices both normalised and physical unit.
    figures/ qualitative predicted fields predicted timesteps, RMSE and std_ratio

    
    """
    combos = [(model_name, loss_name) for model_name in SWEEP_MODELS for loss_name in SWEEP_LOSSES] #Define all combinations
    failed = [] #store filed entries
 
    print(f"\n{'=' * 70}")
    print(f"  SWEEP PLAN — {len(combos)} combinations")
    for i, (model_name, loss_name) in enumerate(combos, 1):
        print(f"  {i:>2}. {model_name:<6} x {loss_name}")
    print(f"{'=' * 70}\n")

    os.makedirs(SWEEP_ROOT, exist_ok=True)

#The main loop for sweep
    for model_name, loss_name in combos:
        tag = f"{model_name}_{loss_name}"
        run_dir = os.path.join(SWEEP_ROOT, tag) #Save

        print(f"\n{'=' * 70}")
        print(f"  START {tag}")
        print(f"{'=' * 70}")

        try:
            train(
                model_name=model_name,
                loss_name=loss_name,
                run_dir=run_dir,
                resume=False,
            )

            ckpt_path = os.path.join(run_dir, "best_model.pt")
            evaluate(checkpoint_path=ckpt_path)
            print(f"\n  {tag} DONE")

        except Exception:
            print(f"\n  ERROR in {tag}:")
            traceback.print_exc()
            failed.append((model_name, loss_name))
            continue

    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE: {len(combos) - len(failed)} / {len(combos)} runs succeeded")
    if failed:
        print(f"  FAILED: {failed}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
