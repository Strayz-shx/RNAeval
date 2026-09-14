#!/usr/bin/env python3
"""
读取 mcc_energy_records.csv，对每个 subfolder 组内：
  1. 将 energy 组内归一化（min-max 或 z-score）
  2. 计算归一化后的 energy 与 mcc_norm 的相关性
  3. 按 energy 排名，统计 energy 排第 1 的结构其 mcc 排第几（忽略只有一个候选的 subfolder）
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcc_energy_records.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FILTER_FILE = os.path.join(OUTPUT_DIR, "A_434.txt")


def load_folder_filter():
    """读取 A_434.txt，返回需要统计的文件夹名集合"""
    if not os.path.exists(FILTER_FILE):
        print(f"[WARN] 过滤文件不存在: {FILTER_FILE}，将处理所有文件夹")
        return None
    folders = set()
    with open(FILTER_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("Folder"):
                continue
            folder_name = line.split()[0].strip()
            if folder_name.isdigit():
                folders.add(folder_name)
    print(f"已加载 {len(folders)} 个需统计的文件夹")
    return folders


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] 找不到 CSV 文件: {CSV_PATH}")
        sys.exit(1)

    allowed_folders = load_folder_filter()

    # ---- 读取数据 ----
    records = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 如果有关键文件夹过滤，跳过不在白名单中的 subfolder
            if allowed_folders is not None and row["subfolder"].strip() not in allowed_folders:
                continue
            row["mcc_norm"] = float(row["mcc_norm"])
            row["energy"] = float(row["energy"])
            records.append(row)

    print(f"总记录数: {len(records)}")

    # ---- 按 subfolder 分组 ----
    groups = defaultdict(list)
    for r in records:
        groups[r["subfolder"]].append(r)

    # 过滤掉只有一个候选的 subfolder
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"子文件夹总数: {len(groups)}")
    print(f"候选数 >1 的子文件夹数: {len(multi_groups)}")

    # ---- 1. 组内 energy 归一化 + 计算相关性 ----
    all_norm_energy = []
    all_mcc = []

    for sf, cands in multi_groups.items():
        energies = np.array([c["energy"] for c in cands])
        mccs = np.array([c["mcc_norm"] for c in cands])

        # min-max 归一化到 [0, 1]
        e_min, e_max = energies.min(), energies.max()
        if e_max > e_min:
            normed = (energies - e_min) / (e_max - e_min)
        else:
            normed = energies - energies  # 全 0

        all_norm_energy.extend(normed.tolist())
        all_mcc.extend(mccs.tolist())

    all_norm_energy = np.array(all_norm_energy)
    all_mcc = np.array(all_mcc)

    # Pearson / Spearman
    from scipy.stats import pearsonr, spearmanr

    pr, pp = pearsonr(all_norm_energy, all_mcc)
    sr, sp = spearmanr(all_norm_energy, all_mcc)

    print("\n" + "=" * 60)
    print("组内归一化 Energy vs MCC 相关性（Min-Max 归一化）")
    print("=" * 60)
    print(f"数据点数: {len(all_norm_energy)}")
    print(f"Pearson  r = {pr:.6f}  p = {pp:.6e}")
    print(f"Spearman ρ = {sr:.6f}  p = {sp:.6e}")

    # ---- 2. 组内排名统计 ----
    # 对每个 subfolder，按 energy 降序排名、按 mcc 降序排名
    # 统计 energy 排第 1 的候选，其 mcc 排在第几位
    rank_counts = defaultdict(int)  # mcc_rank -> 频次
    total_groups = 0

    for sf, cands in multi_groups.items():
        total_groups += 1

        # energy 降序
        sorted_by_energy = sorted(cands, key=lambda x: x["energy"], reverse=True)
        # mcc 降序
        sorted_by_mcc = sorted(cands, key=lambda x: x["mcc_norm"], reverse=True)

        # energy 排名第 1 的候选文件名
        top_energy_file = sorted_by_energy[0]["candidate_file"]

        # 它在 mcc 排名中的位置（第几名）
        mcc_rank = next(
            i + 1 for i, c in enumerate(sorted_by_mcc) if c["candidate_file"] == top_energy_file
        )

        rank_counts[mcc_rank] += 1

    print("\n" + "=" * 60)
    print("Energy 排第 1 的候选 → MCC 排名分布")
    print(f"（统计基数: {total_groups} 个子文件夹）")
    print("=" * 60)

    sorted_ranks = sorted(rank_counts.keys())
    for rank in sorted_ranks:
        count = rank_counts[rank]
        pct = count / total_groups * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"MCC 第 {rank:3d} 名: {count:4d} 次 ({pct:5.1f}%) {bar}")

    # ---- 3. 累积统计 ----
    print()
    cum = 0
    for rank in sorted_ranks:
        cum += rank_counts[rank]
        pct = cum / total_groups * 100
        print(f"Energy 第1 → MCC 前 {rank:3d} 名: 累计 {cum:4d} 次 ({pct:5.1f}%)")

    # ---- 4. 输出 MCC 各排名段的命中率汇总 ----
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    top1 = rank_counts.get(1, 0)
    top3 = sum(rank_counts.get(r, 0) for r in [1, 2, 3])
    top5 = sum(rank_counts.get(r, 0) for r in range(1, 6))
    bottom = sum(
        rank_counts.get(r, 0)
        for r in sorted_ranks
        if r > len(next(iter(multi_groups.values()))) - 2
    )
    avg_candidates = np.mean([len(v) for v in multi_groups.values()])
    random_baseline_top1 = 1.0 / avg_candidates * 100

    print(f"平均每个 subfolder 候选数: {avg_candidates:.1f}")
    print(f"随机基线（Energy 第1=MCC第1）: {random_baseline_top1:.1f}%")
    print()
    print(f"Energy 第1 恰好也是 MCC 第1:  {top1:4d} 次 ({top1 / total_groups * 100:.1f}%)")
    print(f"Energy 第1 落在 MCC 前3:      {top3:4d} 次 ({top3 / total_groups * 100:.1f}%)")
    print(f"Energy 第1 落在 MCC 前5:      {top5:4d} 次 ({top5 / total_groups * 100:.1f}%)")
    print(
        f"Energy 第1 落在 MCC 末尾:      {bottom:4d} 次 ({bottom / total_groups * 100:.1f}%)"
    )

    # ---- 保存结果 ----
    summary_lines = [
        "===== Energy vs MCC 组内排名分析报告 =====",
        f"总记录数: {len(records)}",
        f"子文件夹总数: {len(groups)}",
        f"候选数>1的子文件夹数: {len(multi_groups)}",
        "",
        "--- 组内归一化 Energy vs MCC 相关性 (Min-Max) ---",
        f"Pearson  r = {pr:.6f}  p = {pp:.6e}",
        f"Spearman ρ = {sr:.6f}  p = {sp:.6e}",
        "",
        "--- Energy 第1 → MCC 排名分布 ---",
    ]
    for rank in sorted_ranks:
        count = rank_counts[rank]
        pct = count / total_groups * 100
        summary_lines.append(f"MCC 第 {rank:3d} 名: {count:4d} 次 ({pct:5.1f}%)")
    summary_lines += [
        "",
        "--- 累积 ---",
    ]
    cum = 0
    for rank in sorted_ranks:
        cum += rank_counts[rank]
        pct = cum / total_groups * 100
        summary_lines.append(f"Energy 第1 → MCC 前 {rank:3d} 名: 累计 {cum:4d} 次 ({pct:5.1f}%)")
    summary_lines += [
        "",
        "--- 汇总 ---",
        f"平均候选数: {avg_candidates:.1f}",
        f"随机基线 Top1: {random_baseline_top1:.1f}%",
        f"Energy 第1 = MCC 第1: {top1} 次 ({top1 / total_groups * 100:.1f}%)",
        f"Energy 第1 ∈ MCC 前3: {top3} 次 ({top3 / total_groups * 100:.1f}%)",
        f"Energy 第1 ∈ MCC 前5: {top5} 次 ({top5 / total_groups * 100:.1f}%)",
        f"Energy 第1 ∈ MCC 末尾: {bottom} 次 ({bottom / total_groups * 100:.1f}%)",
    ]

    out_path = os.path.join(OUTPUT_DIR, "energy_mcc_ranking_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
