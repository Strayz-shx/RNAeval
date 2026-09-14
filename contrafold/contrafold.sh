#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shihaoxuan0519@qq.com

# 加载环境
source ~/.bashrc
module load miniforge3/24.1 compilers/cuda/12.4 cudnn/8.9.5.29_cuda12.x compilers/gcc/12.2.0 cmake/3.26.3
export CONDA_PREFIX=/home/bingxing2/home/scx7auf/.conda
conda activate rnallm

# ==================== 配置区域 ====================
INPUT_FASTA="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/trainseta.fasta"   # ← 修改：多序列 FASTA 文件路径
OUTPUT_DIR="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/contrafold_train"       # ← 修改：BPSEQ 结果保存文件夹
CONTRAFOLD_BIN="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/contrafold/src/contrafold"                 # ← 修改：contrafold 可执行文件路径
# ================================================

# 创建输出文件夹
mkdir -p "$OUTPUT_DIR"

# 创建临时拆分目录
TMP_DIR="$OUTPUT_DIR/.tmp_split_$$"
mkdir -p "$TMP_DIR"
trap "rm -rf $TMP_DIR" EXIT

echo "正在拆分多序列 FASTA 文件: $INPUT_FASTA"

# 使用 Python 标准库拆分多序列 FASTA
python3 - "$INPUT_FASTA" "$TMP_DIR" << 'PYEOF'
import sys, os

input_fasta = sys.argv[1]
tmp_dir = sys.argv[2]

current_id = None
current_lines = []

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in '-_.' else '_' for c in name)

def write_seq(seq_id, lines):
    if not seq_id or not lines:
        return
    safe_id = sanitize_filename(seq_id)
    out_path = os.path.join(tmp_dir, f"{safe_id}.fasta")
    counter = 1
    original_safe_id = safe_id
    while os.path.exists(out_path):
        safe_id = f"{original_safe_id}_{counter}"
        out_path = os.path.join(tmp_dir, f"{safe_id}.fasta")
        counter += 1
    with open(out_path, 'w') as f:
        f.write(f">{seq_id}\n")
        f.write("".join(lines) + "\n")
    print(f"SPLIT:{safe_id}:{seq_id}")

with open(input_fasta, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current_id is not None:
                write_seq(current_id, current_lines)
            current_id = line[1:].split()[0]
            current_lines = []
        else:
            current_lines.append(line)
    if current_id is not None:
        write_seq(current_id, current_lines)
PYEOF

# 统计拆分出的文件数量
total=$(find "$TMP_DIR" -name "*.fasta" | wc -l)
echo "共拆分出 $total 条序列，开始批量预测..."

# 遍历拆分后的单序列文件
count=0
for fasta_file in "$TMP_DIR"/*.fasta; do
    [ -e "$fasta_file" ] || continue
    
    # 获取序列 ID（文件名去掉 .fasta）
    seq_id=$(basename "$fasta_file" .fasta)
    output_bpseq="$OUTPUT_DIR/${seq_id}.bpseq"
    
    count=$((count + 1))
    echo "[$count/$total] 正在预测序列: $seq_id"
    
    # 调用 contrafold 预测，输出 bpseq 格式
    $CONTRAFOLD_BIN predict "$fasta_file" --bpseq "$output_bpseq"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 完成: $output_bpseq"
    else
        echo "  ✗ 失败: $seq_id" >&2
    fi
done

echo "批量预测完成！共处理 $count 条序列，结果保存在: $OUTPUT_DIR"