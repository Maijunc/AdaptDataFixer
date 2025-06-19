#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价格修复规则
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)


class PriceRepairRule(BaseRepairRule):
    """价格修复规则，处理与价格相关的数据修复"""

    def __init__(self):
        """初始化价格修复规则"""
        super().__init__(
            name="价格修复",
            description="修复开盘价、收盘价、最高价、最低价等价格相关指标"
        )

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用价格修复规则到单个股票组

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        repair_count = 0

        # 1. 基本价格修复（开盘价、收盘价、最高价、最低价）
        logger.debug("正在修复基本价格...")
        price_cols = ['今开', '现价', '最高', '最低']
        missing_cols = [col for col in price_cols if col not in group.columns]

        if missing_cols:
            logger.warning(f"缺少基本价格列 {missing_cols}，跳过基本价格修复")
        else:
            # 使用前值填充0值
            for col in price_cols:
                mask = (group[col] == 0)
                if mask.any():
                    group[col] = group[col].replace(0, np.nan)
                    group[col] = group[col].ffill().bfill()
                    repair_count += mask.sum()

            # 确保价格逻辑关系正确
            mask = (group['最高'] < group['最低'])
            if mask.any():
                temp = group.loc[mask, '最高'].copy()
                group.loc[mask, '最高'] = group.loc[mask, '最低']
                group.loc[mask, '最低'] = temp
                repair_count += mask.sum()

            # 确保开盘价和收盘价在最高价和最低价之间
            for col in ['今开', '现价']:
                mask_high = (group[col] > group['最高'])
                mask_low = (group[col] < group['最低'])

                if mask_high.any():
                    group.loc[mask_high, col] = group.loc[mask_high, '最高']
                    repair_count += mask_high.sum()

                if mask_low.any():
                    group.loc[mask_low, col] = group.loc[mask_low, '最低']
                    repair_count += mask_low.sum()

        # 2. 均价修复
        logger.debug("正在修复均价...")
        if '均价' in group.columns and '现价' in group.columns and '今开' in group.columns:
            mask = (group['均价'] == 0)
            if mask.any():
                # 使用开盘价和收盘价的平均值作为均价
                group.loc[mask, '均价'] = (group.loc[mask, '今开'] + group.loc[mask, '现价']) / 2
                repair_count += mask.sum()

        # 3. 昨收价修复
        logger.debug("正在修复昨收价...")
        if '昨收' in group.columns and '现价' in group.columns and '今开' in group.columns:
            mask = (group['昨收'] == 0)
            if mask.any():
                # 使用前一天的收盘价作为昨收
                group['昨收'] = group['现价'].shift(1)
                # 对于第一天的数据，使用当天开盘价
                first_day_mask = (group['昨收'].isna())
                group.loc[first_day_mask, '昨收'] = group.loc[first_day_mask, '今开']
                repair_count += mask.sum()

        # 4. 52周最高价和最低价修复
        logger.debug("正在修复52周最高价和最低价...")
        if '52周最高' in group.columns and '52周最低' in group.columns and '最高' in group.columns and '最低' in group.columns:
            # 计算52周（约252个交易日）滚动窗口的最高价和最低价
            rolling_high = group['最高'].rolling(window=252, min_periods=1).max()
            rolling_low = group['最低'].rolling(window=252, min_periods=1).min()

            # 修复52周最高价
            mask_high = (group['52周最高'] == 0)
            if mask_high.any():
                group.loc[mask_high, '52周最高'] = rolling_high[mask_high]
                repair_count += mask_high.sum()

            # 修复52周最低价
            mask_low = (group['52周最低'] == 0)
            if mask_low.any():
                group.loc[mask_low, '52周最低'] = rolling_low[mask_low]
                repair_count += mask_low.sum()

            # 确保52周最高价大于52周最低价
            mask = (group['52周最高'] < group['52周最低'])
            if mask.any():
                temp = group.loc[mask, '52周最高'].copy()
                group.loc[mask, '52周最高'] = group.loc[mask, '52周最低']
                group.loc[mask, '52周最低'] = temp
                repair_count += mask.sum()

        # 5. 涨跌停价修复
        logger.debug("正在修复涨跌停价...")
        if '涨停价' in group.columns and '跌停价' in group.columns and '昨收' in group.columns:
            # 获取涨跌停比例（默认为10%，ST股票为5%）
            limit_rate = 0.1  # 可以根据实际情况调整

            # 修复涨停价
            mask_up = (group['涨停价'] == 0)
            if mask_up.any():
                group.loc[mask_up, '涨停价'] = group.loc[mask_up, '昨收'] * (1 + limit_rate)
                repair_count += mask_up.sum()

            # 修复跌停价
            mask_down = (group['跌停价'] == 0)
            if mask_down.any():
                group.loc[mask_down, '跌停价'] = group.loc[mask_down, '昨收'] * (1 - limit_rate)
                repair_count += mask_down.sum()

        logger.info(f"价格修复完成，共修复 {repair_count} 处数据")
        return group, repair_count
