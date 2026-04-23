import numpy as np
import os, sys, argparse, time, subprocess
from concurrent.futures import ProcessPoolExecutor as Executor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from functools import partial
from subprocess import Popen, CalledProcessError
import shlex

def _build_child_cmd(args, fold_id):
    cmd = [
        sys.executable, "-u", "src/train_one_fold_dynamic.py",
        "--fold_id", str(fold_id),
        "-amp_tr", args.amp_tr,
        "-non_amp_tr", args.non_amp_tr,
        "-sample_ratio", args.sample_ratio,
        "-out_dir", args.out_dir,
        "-model_name", args.model_name,
    ]
    if args.amp_te is not None:
        cmd += ["-amp_te", args.amp_te]
    if args.non_amp_te is not None:
        cmd += ["-non_amp_te", args.non_amp_te]
    return cmd

def parent_visible_ids():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [s.strip() for s in cvd.split(",") if s.strip() != ""]

def launch_child_gpu_async(args, fold_id, logical_slot):
    vis = parent_visible_ids()
    if logical_slot >= len(vis):
        raise RuntimeError(f"Slot {logical_slot} requested but only {len(vis)} GPUs available")

    phys_id = vis[logical_slot]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = phys_id
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = _build_child_cmd(args, fold_id)
    print(f"[LAUNCH] fold={fold_id} -> SLOT{logical_slot} (phys GPU {phys_id})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    proc = Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, fold_id, logical_slot

def run_gpu_fold(args):
    fold_count = 5
    pending = list(range(fold_count))
    active  = []

    vis = parent_visible_ids()
    n_gpus = len(vis)
    if n_gpus < 1:
        raise RuntimeError(f"[FATAL] No GPUs found; got CUDA_VISIBLE_DEVICES='{','.join(vis)}'")

    print(f"[INFO] Running with {n_gpus} GPU(s)", flush=True)

    for slot in range(min(n_gpus, fold_count)):
        fid = pending.pop(0)
        proc, _, _ = launch_child_gpu_async(args, fid, slot)
        active.append((proc, fid, slot))

    while active:
        time.sleep(1)
        next_active = []
        for proc, fid, slot in active:
            rc = proc.poll()
            if rc is None:
                next_active.append((proc, fid, slot))
                continue
            if rc != 0:
                print(f"[FOLD {fid}] FAILED on SLOT{slot}", flush=True)
                raise CalledProcessError(rc, ['child_fold'])
            print(f"[FOLD {fid}] OK on SLOT{slot}", flush=True)
            if pending:
                new_fid = pending.pop(0)
                new_proc, _, _ = launch_child_gpu_async(args, new_fid, slot)
                next_active.append((new_proc, new_fid, slot))
        active = next_active

def run_cpu_fold(args, fold_id):
    cmd = [
        sys.executable, "src/train_one_fold_dynamic.py",
        "--fold_id", str(fold_id),
        "-amp_tr", args.amp_tr,
        "-non_amp_tr", args.non_amp_tr,
        "-sample_ratio", args.sample_ratio,
        "-out_dir", args.out_dir,
        "-model_name", args.model_name
    ]
    if args.amp_te is not None:
        cmd += ["-amp_te", args.amp_te]
    if args.non_amp_te is not None:
        cmd += ["-non_amp_te", args.non_amp_te]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_ensemble_threshold(out_dir, sample_ratio):
    if sample_ratio == 'balanced':
        print("Balanced model: using fixed threshold 0.5")
        return 0.5
    thresholds = []
    for f in range(5):
        t = float(np.load(f"{out_dir}/optimal_threshold_{f}.npy"))
        print(f"Fold {f} threshold: {t:.4f}")
        thresholds.append(t)
    threshold = np.median(thresholds)
    print(f"Ensemble threshold (median): {threshold:.4f}")
    return threshold

def average_predictions(out_dir, sample_ratio):
    indv_pred_train, indv_pred_test = [], []
    for f in range(5):
        indv_pred_train.append(np.load(f"{out_dir}/temp_pred_train{f}.npy"))
        indv_pred_test.append(np.load(f"{out_dir}/temp_pred_test{f}.npy"))
    y_pred_prob_test  = np.mean(np.array(indv_pred_test), axis=0)
    y_pred_prob_train = np.mean(np.array(indv_pred_train), axis=0)
    np.save(f"{out_dir}/ensemble_pred_train.npy", y_pred_prob_train)
    np.save(f"{out_dir}/ensemble_pred_test.npy",  y_pred_prob_test)

    threshold = get_ensemble_threshold(out_dir, sample_ratio)
    np.save(f"{out_dir}/ensemble_threshold.npy", np.array(threshold))
    final_metrics(out_dir, y_pred_prob_test, y_pred_prob_train, threshold)

def final_metrics(out_dir, y_pred_prob_test, y_pred_prob_train, threshold):
    y_test  = np.load(f"{out_dir}/y_test_labels.npy")
    y_train = np.load(f"{out_dir}/y_train_labels.npy")
    y_pred_class_test  = (y_pred_prob_test > threshold).astype(int)
    y_pred_class_train = (y_pred_prob_train > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class_test).ravel()
    print('**************************** final model ****************************')
    print(f'threshold used: {threshold:.4f}')
    print('overall train acc: ', accuracy_score(y_train, y_pred_class_train))
    print('overall test  acc: ', accuracy_score(y_test,  y_pred_class_test))
    print(confusion_matrix(y_test, y_pred_class_test))
    print('overall test sens: ', tp/(tp+fn))
    print('overall test spec: ', tn/(tn+fp))
    print('overall test f1:   ', f1_score(y_test, y_pred_class_test))
    print('overall test roc_auc:', roc_auc_score(y_test, y_pred_prob_test))
    print('*********************************************************************')

def main():
    parser = argparse.ArgumentParser(description="AMP 5-Fold Training Pipeline")
    parser.add_argument('--amp_tr', required=True)
    parser.add_argument('--non_amp_tr', required=True)
    parser.add_argument('--amp_te', default=None)
    parser.add_argument('--non_amp_te', default=None)
    parser.add_argument('--sample_ratio', choices=['balanced', 'imbalanced'], default='balanced')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--model_num', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with Executor(max_workers=args.model_num) as executor:
            executor.map(partial(run_cpu_fold, args), range(5))
    else:
        run_gpu_fold(args)

    if args.amp_te is not None and args.non_amp_te is not None:
        average_predictions(args.out_dir, args.sample_ratio)

if __name__ == "__main__":
    main()