#!/usr/bin/env python3
"""
使用 CONTRAfold 计算候选二级结构的对数配分函数（能量），
与真实结构计算 MCC（归一化到 [0,1]），
分析二者之间的相关性。

数据目录结构:
  data_512/TestA_512/
    bpseq/          ← 真实结构文件 (*.bpseq)
    res_10/         ← 子文件夹（与 bpseq/*.bpseq 一一对应），内含候选结构 (*.bpseq)

优化：同一子文件夹内的所有候选结构共享同一序列（bpseq 文件中 index/base 相同），
因此无约束配分函数只需计算一次，大幅减少 CONTRAfold 调用次数。
"""

import math
import os
import re
import subprocess
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 路径配置
# ============================================================
DATA_ROOT = "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data_code/data_512/TestB_512"
REAL_BPSEQ_DIR = os.path.join(DATA_ROOT, "bpseq")
CAND_ROOT = os.path.join(DATA_ROOT, "res_10")
CONTRAFOLD_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "contrafold"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MCC 计算（从 dataset.py 的 evaluate_similarity + calculate_average 移植）
# ============================================================
def calculate_mcc(tp, tn, fp, fn):
    """计算 MCC 并归一化到 [0, 1]"""
    mc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mc == 0:
        mcc = 0
    else:
        mcc = ((tp * tn) - (fp * fn)) / mc
    return (mcc + 1) / 2  # 归一化 [-1, 1] → [0, 1]


def evaluate_mcc(real_path, cand_path):
    """读取 BPSEQ，计算归一化 MCC"""
    nodes_real, nodes_cand = [], []

    with open(real_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                nodes_real.append(parts[2])

    with open(cand_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                nodes_cand.append(parts[2])

    if len(nodes_real) != len(nodes_cand):
        return None

    tp = tn = fp = fn = 0
    for r, c in zip(nodes_real, nodes_cand):
        r, c = int(r), int(c)
        if r == c:
            if r != 0:
                tp += 1
            else:
                tn += 1
        else:
            if r != 0 and c == 0:
                fp += 1
            else:
                fn += 1

    return calculate_mcc(tp, tn, fp, fn)


# ============================================================
# BPSEQ 工具函数
# ============================================================
def _parse_bpseq_sequence(bpseq_path):
    """
    从 BPSEQ 文件中提取序列字符串（跳过 pair_index 列），
    用于判断同一子文件夹内的候选是否共享相同序列。
    """
    seq = []
    with open(bpseq_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                seq.append(parts[1].upper())
    return "".join(seq)


def extract_sequence_for_fasta(bpseq_path):
    """
    从 BPSEQ 文件提取序列并创建临时 FASTA 文件，
    用于计算无约束配分函数（更准确）。
    但 contrafold 可以直接从 BPSEQ 读序列，所以返回序列字符串供校验。
    """
    return _parse_bpseq_sequence(bpseq_path)


# ============================================================
# CONTRAfold 配分函数计算
# ============================================================
def call_contrafold_partition(bpseq_path, use_constraints=False):
    """
    调用 contrafold predict <bpseq> --partition [--constraints]。
    返回配分函数值 float，或 None。
    """
    cmd = [CONTRAFOLD_BIN, "predict", bpseq_path, "--partition"]
    if use_constraints:
        cmd.append("--constraints")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        match = re.search(
            r"Log partition coefficient for\s+\S+\s*:\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            output,
        )
        if match:
            return float(match.group(1))
        else:
            print(f"  [WARN] 无法解析配分函数: {bpseq_path}")
            print(f"  输出: {output[:150]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [WARN] 超时: {bpseq_path}")
        return None
    except Exception as e:
        print(f"  [ERROR] contrafold 异常: {e}")
        return None


# ============================================================
# 缓存管理（同一子文件夹内候选结构共享序列，复用无约束配分）
# ============================================================
_unconstrained_cache = {}  # seq_string -> float


def get_unconstrained(cand_path):
    """获取候选结构的无约束配分（读序列→查缓存→计算→存入缓存）"""
    seq = _parse_bpseq_sequence(cand_path)
    if seq not in _unconstrained_cache:
        val = call_contrafold_partition(cand_path, use_constraints=False)
        _unconstrained_cache[seq] = val
        return val
    return _unconstrained_cache[seq]


def compute_energy(cand_path):
    """energy = constrained(cand) - unconstrained(seq)"""
    constrained = call_contrafold_partition(cand_path, use_constraints=True)
    if constrained is None:
        return None
    unconstrained = get_unconstrained(cand_path)
    if unconstrained is None:
        return None
    return constrained - unconstrained


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("CONTRAfold 配分函数 vs MCC 相关性分析")
    print("=" * 60)
    print(f"真实结构: {REAL_BPSEQ_DIR}")
    print(f"候选结构: {CAND_ROOT}")
    print(f"CONTRAfold: {CONTRAFOLD_BIN}")
    print()

    # 校验路径
    for p, name in [(REAL_BPSEQ_DIR, "真实结构目录"),
                    (CAND_ROOT, "候选结构目录"),
                    (CONTRAFOLD_BIN, "CONTRAfold")]:
        if not os.path.exists(p):
            print(f"[ERROR] {name} 不存在: {p}")
            sys.exit(1)

    # 收集子文件夹
    all_subdirs = sorted(
        [d for d in os.listdir(CAND_ROOT) if os.path.isdir(os.path.join(CAND_ROOT, d))],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"发现 {len(all_subdirs)} 个子文件夹")

    all_mcc = []
    all_energy = []
    all_records = []

    total = 0
    skip_no_real = 0
    skip_mcc = 0
    skip_energy = 0

    t_start = time.time()

    for idx, subfolder in enumerate(all_subdirs):
        real_path = os.path.join(REAL_BPSEQ_DIR, f"{subfolder}.bpseq")
        if not os.path.exists(real_path):
            skip_no_real += 1
            continue

        cand_dir = os.path.join(CAND_ROOT, subfolder)
        cand_files = sorted(
            [os.path.join(cand_dir, f) for f in os.listdir(cand_dir) if f.endswith(".bpseq")]
        )
        if not cand_files:
            continue

        elapsed = time.time() - t_start
        print(f"\r[{idx + 1}/{len(all_subdirs)}] subfolder={subfolder}  "
              f"candidates={len(cand_files)}  cache_size={len(_unconstrained_cache)}  "
              f"elapsed={elapsed:.0f}s", end="", flush=True)

        for cand_path in cand_files:
            # MCC
            mcc_norm = evaluate_mcc(real_path, cand_path)
            if mcc_norm is None:
                skip_mcc += 1
                continue

            # 能量
            energy = compute_energy(cand_path)
            if energy is None:
                skip_energy += 1
                continue

            all_mcc.append(mcc_norm)
            all_energy.append(energy)
            all_records.append((subfolder, os.path.basename(cand_path), mcc_norm, energy))
            total += 1

    print()
    t_elapsed = time.time() - t_start
    print(f"\n耗时: {t_elapsed:.0f}s ({t_elapsed / 60:.1f}min)")
    print(f"成功: {total}, 跳过(无真实结构): {skip_no_real}, "
          f"跳过(MCC): {skip_mcc}, 跳过(能量): {skip_energy}")
    print(f"无约束配分缓存命中: {len(_unconstrained_cache)} 条唯一序列")

    if len(all_mcc) < 2:
        print("[ERROR] 数据不足，退出")
        sys.exit(1)

    all_mcc = np.array(all_mcc)
    all_energy = np.array(all_energy)

    # ============================================================
    # 相关性分析
    # ============================================================
    print("\n" + "=" * 60)
    print("相关性分析结果")
    print("=" * 60)
    print(f"数据点数: {len(all_mcc)}")
    print(f"MCC(归一化)  min={all_mcc.min():.4f}  max={all_mcc.max():.4f}  "
          f"mean={all_mcc.mean():.4f}  std={all_mcc.std():.4f}")
    print(f"Energy      min={all_energy.min():.4f}  max={all_energy.max():.4f}  "
          f"mean={all_energy.mean():.4f}  std={all_energy.std():.4f}")

    pearson_r, pearson_p = pearsonr(all_energy, all_mcc)
    spearman_r, spearman_p = spearmanr(all_energy, all_mcc)

    print(f"\nPearson  r = {pearson_r:.6f}  p = {pearson_p:.6e}")
    print(f"Spearman ρ = {spearman_r:.6f}  p = {spearman_p:.6e}")

    # ============================================================
    # 保存结果
    # ============================================================
    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "mcc_energy_records.csv")
    with open(csv_path, "w") as f:
        f.write("subfolder,candidate_file,mcc_norm,energy\n")
        for sf, cn, m, e in all_records:
            f.write(f"{sf},{cn},{m:.6f},{e:.6f}\n")
    print(f"\nCSV 已保存: {csv_path}")

    # 报告
    summary_path = os.path.join(OUTPUT_DIR, "mcc_energy_summary.txt")
    with open(summary_path, "w") as f:
        f.write("===== CONTRAfold 配分函数 vs MCC 相关性分析报告 =====\n")
        f.write(f"数据目录:     {DATA_ROOT}\n")
        f.write(f"分析时间:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"运行耗时:     {t_elapsed:.0f}s\n\n")
        f.write(f"数据点数:     {len(all_mcc)}\n")
        f.write(f"候选结构总数: {total}\n")
        f.write(f"唯一序列数:   {len(_unconstrained_cache)}\n\n")
        f.write(f"MCC(归一化)  min={all_mcc.min():.4f}  max={all_mcc.max():.4f}  "
                f"mean={all_mcc.mean():.4f}  std={all_mcc.std():.4f}\n")
        f.write(f"Energy      min={all_energy.min():.4f}  max={all_energy.max():.4f}  "
                f"mean={all_energy.mean():.4f}  std={all_energy.std():.4f}\n\n")
        f.write(f"Pearson  r = {pearson_r:.6f}  p = {pearson_p:.6e}\n")
        f.write(f"Spearman ρ = {spearman_r:.6f}  p = {spearman_p:.6e}\n")
    print(f"报告已保存: {summary_path}")

    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    r_abs = abs(pearson_r)
    if r_abs > 0.7:
        strength = "强"
    elif r_abs > 0.5:
        strength = "中等"
    elif r_abs > 0.3:
        strength = "弱"
    else:
        strength = "极弱或无"

    print(f"Energy 与归一化 MCC 之间存在{strength}相关性 "
          f"(Pearson r={pearson_r:.4f}, p={pearson_p:.2e})")
    if pearson_r > 0.3:
        print("→ 正向：CONTRAfold 能量越高（越接近0），结构越准确")
    elif pearson_r < -0.3:
        print("→ 负向：CONTRAfold 能量越低，结构越准确")
    else:
        print("→ CONTRAfold 的配分函数评分与结构准确性无明显线性关系")


if __name__ == "__main__":
    main()
