#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
封单额修复规则
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)


class FengdanRepairRule(BaseRepairRule):
    """封单额修复规则，处理封单额、昨封单额和前封单额的修复"""

    def __init__(self):
        """初始化封单额修复规则"""
        super().__init__(
            name="封单额修复",
            description="修复封单额、昨封单额和前封单额数据"
        )

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用封单额修复规则到单个股票组

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        logger.debug("正在修复封单额数据...")

        # 检查必要的列是否存在
        required_cols = ['封单额', '昨封单额', '前封单额']
        if not self.check_columns(group, required_cols):
            logger.warning(f"缺少必要的列，跳过封单额修复")
            return group, 0

        repair_count = 0
        indices = group.index.tolist()

        for i in range(len(indices)):
            current_idx = indices[i]
            current_fengdan = group.at[current_idx, '封单额']

            # 获取下一日和再下一日的索引
            next_idx = indices[i + 1] if i + 1 < len(indices) else None
            next_next_idx = indices[i + 2] if i + 2 < len(indices) else None

            # 获取相关封单额值（避免索引越界）
            next_zuo_fengdan = group.at[next_idx, '昨封单额'] if next_idx else 0
            next_next_qian_fengdan = group.at[next_next_idx, '前封单额'] if next_next_idx else 0

            # 规则1：当前日封单额覆盖下一日的昨封单额（如果为0）
            if next_idx and next_zuo_fengdan == 0 and current_fengdan != 0:
                group.at[next_idx, '昨封单额'] = current_fengdan
                repair_count += 1

            # 规则2：当前日封单额覆盖再下一日的前封单额（如果为0）
            if next_next_idx and next_next_qian_fengdan == 0 and current_fengdan != 0:
                group.at[next_next_idx, '前封单额'] = current_fengdan
                repair_count += 1

            # 规则3：下一日的昨封单额覆盖当前日的封单额（如果当前日封单额=0）
            if next_idx and current_fengdan == 0 and next_zuo_fengdan != 0:
                group.at[current_idx, '封单额'] = next_zuo_fengdan
                repair_count += 1

            # 规则4：下一日的昨封单额覆盖再下一日的前封单额（如果前封单额=0）
            if next_idx and next_next_idx and next_next_qian_fengdan == 0 and next_zuo_fengdan != 0:
                group.at[next_next_idx, '前封单额'] = next_zuo_fengdan
                repair_count += 1

            # 规则5：再下一日的前封单额覆盖当前日的封单额（如果封单额=0）
            if next_next_idx and current_fengdan == 0 and next_next_qian_fengdan != 0:
                group.at[current_idx, '封单额'] = next_next_qian_fengdan
                repair_count += 1

            # 规则6：再下一日的前封单额覆盖下一日的昨封单额（如果昨封单额=0）
            if next_idx and next_next_idx and next_zuo_fengdan == 0 and next_next_qian_fengdan != 0:
                group.at[next_idx, '昨封单额'] = next_next_qian_fengdan
                repair_count += 1

        logger.info(f"封单额修复完成，共修复 {repair_count} 处数据")
        return group, repair_count