import base64
import csv
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rqvae_embed.rqvae_clip import RQVAE_EMBED_CLIP

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局常量和配置 ---
EXPECTED_EMBEDDING_DIM = 512
CHUNK_SIZE = 5000  # pandas每次读取的行数
BATCH_SIZE = 128  # 模型批次推理的样本数

INPUT_EMBEDDING_COL = 'TODO'  # 输入文件中embedding列的名称，例如"feature"
INPUT_ID_COL = 'TODO'  # 输入文件中ID列的名称，例如"base62_string"
CKPT_PATH = 'output_model/checkpoint-7.pth'  # todo: 需要推理的ckpt
INPUT_FILE_PATH = './item_feature/final/part_01.csv'  # todo: 输入的emb，可从https://huggingface.co/datasets/AL-GR/Item-EMB获取
OUTPUT_FILE_PATH = 'inference_results_batch.csv'  # todo: 输出结果


def build_model(ckpt_path: str) -> torch.nn.Module:
    """
    构建并加载预训练的RQ-VAE模型。
    """
    logging.info("开始构建模型...")
    codebook_num = 3
    codebook_size = 8192
    codebook_dim = 64
    input_dim = 512

    hps = {
        "bottleneck_type": "rq", "embed_dim": codebook_dim, "n_embed": codebook_size,
        "latent_shape": [8, 8, codebook_dim], "code_shape": [8, 8, codebook_num],
        "shared_codebook": False, "decay": 0.99, "restart_unused_codes": True,
        "loss_type": "cosine", "latent_loss_weight": 0.15, "masked_dropout": 0.0,
        "use_padding_idx": False, "VQ_ema": False, "do_bn": True, 'rotation_trick': False
    }
    ddconfig = {
        "double_z": False, "z_channels": codebook_dim, "resolution": 256, "in_channels": 3,
        "out_ch": 3, "ch": 128, "ch_mult": [1, 1, 2, 2, 4, 4], "num_res_blocks": 2,
        "attn_resolutions": [8], "dropout": 0.00, "input_dim": input_dim
    }

    try:
        model = RQVAE_EMBED_CLIP(hps, ddconfig=ddconfig, checkpointing=True)
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        logging.info(f"正在从 '{ckpt_path}' 加载模型权重...")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
        logging.info("模型加载成功！")
        return model
    except FileNotFoundError:
        logging.error(f"模型检查点文件未找到: {ckpt_path}")
        raise
    except Exception as e:
        logging.error(f"构建模型时发生未知错误: {e}")
        raise


def predict_batch(
        model: torch.nn.Module,
        item_ids_batch: List[str],
        embeddings_batch: List[str]
) -> List[Tuple[str, str]]:
    """
    对一个批次的数据进行解码和模型推理。

    Args:
        model: 已加载的 PyTorch 模型。
        item_ids_batch: 批次的 item_id 列表。
        embeddings_batch: 批次的 base64 编码的 embedding 字符串列表。

    Returns:
        一个包含 (item_id, SID) 元组的列表。
    """
    valid_item_ids = []
    embedding_list = []

    # 1. 解码Base64并过滤无效数据
    for item_id, embedding_str in zip(item_ids_batch, embeddings_batch):
        try:
            embedding_np = np.frombuffer(base64.b64decode(embedding_str), dtype=np.float32)
            if embedding_np.shape[0] != EXPECTED_EMBEDDING_DIM:
                logging.warning(
                    f"Item ID '{item_id}' 的 embedding 维度不正确。 "
                    f"期望维度: {EXPECTED_EMBEDDING_DIM}, 实际维度: {embedding_np.shape[0]}。已跳过此样本。"
                )
                continue  # 跳过这个不符合规范的样本
            embedding_list.append(embedding_np)
            valid_item_ids.append(item_id)
        except Exception as e:
            logging.warning(f"Item ID '{item_id}' 在解码时发生错误: {e}。已跳过。")
            continue

    if not valid_item_ids:
        return []

    # 2. 转换为Tensor并进行推理 (在GPU上执行)
    device = next(model.parameters()).device
    embedding_tensor = torch.from_numpy(np.array(embedding_list)).to(device)

    with torch.no_grad():
        index_batch = model.rq_model.get_codes(embedding_tensor)

    # 3. 将结果转换回CPU并格式化
    cpu_indices = index_batch.cpu().numpy()

    results = []
    for item_id, index_row in zip(valid_item_ids, cpu_indices):
        sid_str = ','.join(index_row.astype(str))
        results.append((item_id, sid_str))

    return results


def process_file(
        model: torch.nn.Module,
        input_path: str,
        output_path: str,
        chunk_size: int,
        batch_size: int
):
    """
    主处理函数，读取CSV，分批推理，并写入结果。
    """
    try:
        # 预先计算总行数以提供准确的进度条
        logging.info("正在计算文件总行数...")
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1
        logging.info(f"文件 '{input_path}' 共有 {total_lines} 行数据。")
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {input_path}")
        return
    except Exception as e:
        logging.error(f"获取文件总行数时发生错误: {e}")
        total_lines = None  # 即使失败也继续，只是进度条不显示百分比

    # 使用 with 语句和 csv.writer 确保文件正确关闭和写入
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['item_id', 'SID'])

        # 缓冲区，用于存储跨pandas chunk的数据
        item_ids_buffer = []
        embeddings_buffer = []

        # 使用tqdm显示进度
        with tqdm(total=total_lines, desc='Processing data') as pbar:
            try:
                # 迭代读取大文件块
                for chunk in pd.read_csv(input_path, chunksize=chunk_size, encoding='utf-8'):
                    # 过滤掉header行（如果存在于数据中）
                    chunk = chunk[chunk[INPUT_EMBEDDING_COL] != INPUT_EMBEDDING_COL]
                    if chunk.empty:
                        continue

                    item_ids_buffer.extend(chunk[INPUT_ID_COL].tolist())
                    embeddings_buffer.extend(chunk[INPUT_EMBEDDING_COL].tolist())

                    # 当缓冲区数据足够时，按批次处理
                    while len(item_ids_buffer) >= batch_size:
                        # 从缓冲区头部取出
                        ids_to_process = item_ids_buffer[:batch_size]
                        embs_to_process = embeddings_buffer[:batch_size]

                        # 从缓冲区移除已取出的数据
                        item_ids_buffer = item_ids_buffer[batch_size:]
                        embeddings_buffer = embeddings_buffer[batch_size:]

                        # 模型推理
                        sid_results = predict_batch(model, ids_to_process, embs_to_process)

                        # 写入结果
                        if sid_results:
                            writer.writerows(sid_results)

                        pbar.update(len(ids_to_process))

                # 处理剩余不足一个批次的数据
                if item_ids_buffer:
                    sid_results = predict_batch(model, item_ids_buffer, embeddings_buffer)
                    if sid_results:
                        writer.writerows(sid_results)
                    pbar.update(len(item_ids_buffer))

            except FileNotFoundError:
                logging.error(f"输入文件未找到: {input_path}")
            except KeyError as e:
                logging.error(
                    f"CSV文件中缺少预期的列: {e}。请检查是否存在 '{INPUT_ID_COL}' 和 '{INPUT_EMBEDDING_COL}' 列。")
            except Exception as e:
                logging.error(f"处理文件时发生未知错误: {e}", exc_info=True)

    logging.info(f"推理完成，结果已保存到: {output_path}")


def main():
    """程序主入口"""
    model = build_model(CKPT_PATH)
    process_file(
        model=model,
        input_path=INPUT_FILE_PATH,
        output_path=OUTPUT_FILE_PATH,
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()
