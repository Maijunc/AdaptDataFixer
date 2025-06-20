#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载和保存模块
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def load_data(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    加载并预处理数据

    Args:
        file_path (str): CSV文件路径
        nrows (Optional[int]): 要读取的行数，None表示读取全部

    Returns:
        pd.DataFrame: 预处理后的数据框

    Raises:
        ValueError: 如果数据不符合要求
    """
    try:
        # 读取CSV文件
        logger.info(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # 检查必要的列
        required_cols = ['代码', '日期']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {', '.join(missing_cols)}")

        # 转换日期列
        logger.debug("正在转换日期格式...")
        df['日期'] = pd.to_datetime(df['日期'])

        # 删除不需要的列
        columns_to_drop = ['涨速%', '短换手%', '开盘竞价']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # 数值列处理
        logger.debug("正在处理数值列...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 按股票代码和日期排序
        logger.debug("正在排序数据...")
        df = df.sort_values(['代码', '日期'])

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"数据加载完成，共 {len(df)} 条记录，{len(df['代码'].unique())} 只股票")
        return df

    except Exception as e:
        logger.error(f"加载数据时发生错误: {str(e)}")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    保存数据到CSV文件

    Args:
        df (pd.DataFrame): 要保存的数据框
        file_path (str): 保存路径

    Raises:
        IOError: 如果保存失败
    """
    try:
        logger.info(f"正在保存数据到: {file_path}")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info("数据保存成功")

    except Exception as e:
        logger.error(f"保存数据时发生错误: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> Optional[str]:
    """
    验证数据的有效性

    Args:
        df (pd.DataFrame): 要验证的数据框

    Returns:
        Optional[str]: 如果数据无效，返回错误信息；如果数据有效，返回None
    """
    # 检查是否为空
    if df.empty:
        return "数据为空"

    # 检查必要的列
    required_cols = ['代码', '日期']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return f"缺少必要的列: {', '.join(missing_cols)}"

    # 检查日期格式
    if not pd.api.types.is_datetime64_any_dtype(df['日期']):
        return "'日期'列不是日期时间格式"

    # 检查代码列是否有空值
    if df['代码'].isnull().any():
        return "'代码'列存在空值"

    # 检查数据是否按代码和日期排序
    if not df.equals(df.sort_values(['代码', '日期']).reset_index(drop=True)):
        return "数据未按'代码'和'日期'排序"

    return None


def get_data_info(df: pd.DataFrame) -> dict:
    """
    获取数据的基本信息

    Args:
        df (pd.DataFrame): 数据框

    Returns:
        dict: 包含数据基本信息的字典
    """
    return {
        'total_records': len(df),
        'stock_count': len(df['代码'].unique()),
        'date_range': (df['日期'].min(), df['日期'].max()),
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'zero_values': (df == 0).sum().to_dict()
    }
