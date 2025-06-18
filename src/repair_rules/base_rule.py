#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础修复规则
"""
from typing import Tuple, List
import pandas as pd
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseRepairRule:
    """所有修复规则的基类"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化基础修复规则

        Args:
            name (str): 规则名称
            description (str): 规则描述
        """
        self.name = name
        self.description = description

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用修复规则到数据框

        Args:
            df (pd.DataFrame): 要修复的数据框

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        logger.debug(f"应用规则: {self.name}")

        # 如果数据框为空，直接返回
        if df.empty:
            logger.warning("数据为空，跳过修复")
            return df, 0

        # 调用子类实现的具体应用方法
        df_repaired, repair_count = self._apply_to_group(df)

        logger.debug(f"规则 '{self.name}' 修复了 {repair_count} 处数据")
        return df_repaired, repair_count

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用修复规则到单个股票组，子类必须实现此方法

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        raise NotImplementedError("子类必须实现_apply_to_group方法")

    def check_columns(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """
        检查数据框是否包含所需的列

        Args:
            df (pd.DataFrame): 要检查的数据框
            columns (List[str]): 所需的列名列表

        Returns:
            bool: 如果所有列都存在则返回True，否则返回False
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"缺少必要的列: {', '.join(missing_cols)}")
            return False
        return True

    def __str__(self) -> str:
        """返回规则的字符串表示"""
        return f"{self.name}: {self.description}"
