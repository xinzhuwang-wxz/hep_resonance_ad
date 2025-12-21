#!/usr/bin/env python
"""
数据下载脚本

从 Zenodo 数据集下载所有数据文件到本地。
数据集: https://zenodo.org/records/14618719
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# 添加项目路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resonance_ad.core.logging import setup_logging, get_logger

logger = get_logger(__name__)

# Zenodo record ID
ZENODO_RECORD_ID = "14618719"
ZENODO_API_BASE = "https://zenodo.org/api"


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """计算文件的MD5哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_file(
    url: str,
    output_path: Path,
    expected_md5: Optional[str] = None,
    chunk_size: int = 8192,
    retries: int = 3,
) -> bool:
    """
    下载文件，支持断点续传和MD5校验
    
    Args:
        url: 下载URL
        output_path: 输出文件路径
        expected_md5: 期望的MD5值（如果提供，下载后会校验）
        chunk_size: 下载块大小
        retries: 重试次数
        
    Returns:
        bool: 下载是否成功
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否已存在
    if output_path.exists():
        file_size = output_path.stat().st_size
        
        # 尝试获取远程文件大小
        try:
            head_response = requests.head(url, allow_redirects=True, timeout=30)
            remote_size = int(head_response.headers.get("Content-Length", 0))
            
            if file_size == remote_size:
                # 文件大小匹配，验证MD5（如果提供）
                if expected_md5:
                    logger.info(f"验证已存在文件的MD5: {output_path.name}")
                    actual_md5 = calculate_md5(output_path)
                    if actual_md5 == expected_md5:
                        logger.info(f"✓ 文件已存在且MD5校验通过: {output_path.name}")
                        return True
                    else:
                        logger.warning(
                            f"MD5不匹配，将重新下载: {output_path.name} "
                            f"(期望: {expected_md5}, 实际: {actual_md5})"
                        )
                        output_path.unlink()  # 删除损坏的文件
                else:
                    logger.info(f"✓ 文件已存在: {output_path.name}")
                    return True
        except Exception as e:
            logger.warning(f"无法验证已存在文件，将重新下载: {e}")
    
    # 下载文件
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("Content-Length", 0))
            
            # 断点续传：如果文件部分存在，从断点继续
            mode = "ab" if output_path.exists() else "wb"
            initial_pos = output_path.stat().st_size if output_path.exists() else 0
            
            if initial_pos > 0:
                # 设置Range头进行断点续传
                headers = {"Range": f"bytes={initial_pos}-"}
                response = requests.get(
                    url, headers=headers, stream=True, timeout=60, allow_redirects=True
                )
            
            with open(output_path, mode) as f:
                if initial_pos > 0:
                    total_size = total_size - initial_pos
                    pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=output_path.name,
                        initial=initial_pos,
                    )
                else:
                    pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=output_path.name,
                    )
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                
                pbar.close()
            
            # MD5校验
            if expected_md5:
                logger.info(f"验证MD5: {output_path.name}")
                actual_md5 = calculate_md5(output_path)
                if actual_md5 != expected_md5:
                    logger.error(
                        f"MD5校验失败: {output_path.name} "
                        f"(期望: {expected_md5}, 实际: {actual_md5})"
                    )
                    output_path.unlink()
                    if attempt < retries - 1:
                        logger.info(f"重试下载 ({attempt + 1}/{retries})...")
                        continue
                    return False
                else:
                    logger.info(f"✓ MD5校验通过: {output_path.name}")
            
            logger.info(f"✓ 下载完成: {output_path.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                logger.info("等待5秒后重试...")
                import time
                time.sleep(5)
            else:
                logger.error(f"下载失败，已达到最大重试次数: {output_path.name}")
                if output_path.exists():
                    output_path.unlink()  # 删除不完整的文件
                return False
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    return False


def get_zenodo_files(record_id: str) -> List[Dict]:
    """
    从Zenodo API获取文件列表
    
    Args:
        record_id: Zenodo记录ID
        
    Returns:
        文件信息列表，每个元素包含 'filename', 'links', 'checksum' 等
    """
    api_url = f"{ZENODO_API_BASE}/records/{record_id}"
    
    logger.info(f"从Zenodo获取文件列表: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        files = data.get("files", [])
        logger.info(f"找到 {len(files)} 个文件")
        
        return files
        
    except requests.exceptions.RequestException as e:
        logger.error(f"无法从Zenodo获取文件列表: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从Zenodo下载数据集文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载到默认目录 (data/precompiled_data/skimmed_data_2016H_30555/)
  python scripts/download_data.py

  # 指定数据ID
  python scripts/download_data.py --data-id skimmed_data_2016H_30555

  # 指定输出目录
  python scripts/download_data.py --output-dir data/precompiled_data/my_data

  # 只下载muon文件
  python scripts/download_data.py --filter mu

  # 只下载jet文件
  python scripts/download_data.py --filter jet
        """,
    )
    parser.add_argument(
        "--data-id",
        type=str,
        default="skimmed_data_2016H_30555",
        help="数据ID（用于创建子目录，默认: skimmed_data_2016H_30555）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: data/precompiled_data/{data_id}）",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["mu", "jet"],
        default=None,
        help="只下载包含指定关键词的文件（mu 或 jet）",
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=ZENODO_RECORD_ID,
        help=f"Zenodo记录ID（默认: {ZENODO_RECORD_ID}）",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="跳过MD5校验（加快下载速度）",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(log_dir=None)  # 不使用文件日志，只在控制台输出
    logger.info("=" * 60)
    logger.info("Zenodo数据下载工具")
    logger.info(f"数据集: https://zenodo.org/records/{args.record_id}")
    logger.info("=" * 60)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认使用项目根目录下的data/precompiled_data/{data_id}
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "data" / "precompiled_data" / args.data_id
    
    logger.info(f"输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取文件列表
    try:
        files = get_zenodo_files(args.record_id)
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        return 1
    
    # 过滤文件
    if args.filter:
        files = [f for f in files if args.filter in f["filename"]]
        logger.info(f"过滤后剩余 {len(files)} 个文件")
    
    # 统计信息
    total_size = sum(f.get("size", 0) for f in files)
    logger.info(f"总文件数: {len(files)}")
    logger.info(f"总大小: {total_size / 1024**3:.2f} GB")
    logger.info("")
    
    # 下载文件
    successful = 0
    failed = 0
    
    for file_info in files:
        filename = file_info["filename"]
        download_url = file_info["links"]["self"]
        file_size = file_info.get("size", 0)
        checksum = file_info.get("checksum", "")
        
        # 提取MD5（格式: "md5:xxxxxxxx"）
        md5 = checksum.split(":")[-1] if checksum.startswith("md5:") else None
        
        output_path = output_dir / filename
        
        logger.info(f"\n下载文件: {filename}")
        logger.info(f"大小: {file_size / 1024**2:.2f} MB")
        if md5 and not args.skip_md5:
            logger.info(f"MD5: {md5}")
        
        if download_file(
            download_url,
            output_path,
            expected_md5=md5 if not args.skip_md5 else None,
        ):
            successful += 1
        else:
            failed += 1
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("下载完成")
    logger.info(f"成功: {successful}/{len(files)}")
    logger.info(f"失败: {failed}/{len(files)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

