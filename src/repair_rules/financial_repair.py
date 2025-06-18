#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务指标修复规则
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)


class FinancialRepairRule(BaseRepairRule):
    """财务指标修复规则，处理与财务相关的数据修复"""

    def __init__(self):
        """初始化财务指标修复规则"""
        super().__init__(
            name="财务指标修复",
            description="修复利润、市值、估值等财务相关指标"
        )

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用财务指标修复规则到单个股票组

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        repair_count = 0

        # 1. 利润同比%、净利润率%的修复
        logger.info("正在修复利润同比%、净利润率%...")
        for col in ['利润同比%', '净利润率%']:
            if col in group.columns:
                # 使用前值覆盖
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].ffill().bfill().fillna(0)
                repair_count += (group[col] == 0).sum()

        # 2. 市值相关指标的修复
        logger.info("正在修复市值相关指标...")
        # 修复流通市值
        if {'流通股(亿)', '现价', '流通市值'}.issubset(group.columns):
            mask = (group['流通市值'] == 0) & (group['流通股(亿)'] > 0) & (group['现价'] > 0)
            group.loc[mask, '流通市值'] = group.loc[mask, '流通股(亿)'] * group.loc[mask, '现价'] * 1e4
            repair_count += mask.sum()

            # 反向修复流通股
            mask = (group['流通股(亿)'] == 0) & (group['流通市值'] > 0) & (group['现价'] > 0)
            group.loc[mask, '流通股(亿)'] = group.loc[mask, '流通市值'] / (group.loc[mask, '现价'] * 1e4)
            repair_count += mask.sum()

        # 修复AB股总市值
        if {'总股本(亿)', '现价', 'AB股总市值'}.issubset(group.columns):
            mask = (group['AB股总市值'] == 0) & (group['总股本(亿)'] > 0) & (group['现价'] > 0)
            group.loc[mask, 'AB股总市值'] = group.loc[mask, '总股本(亿)'] * group.loc[mask, '现价'] * 1e4
            repair_count += mask.sum()

        # 3. 市盈率、市净率等估值指标的修复
        logger.info("正在修复估值指标...")
        # 修复市盈(TTM)
        if {'AB股总市值', '净利润(亿)', '市盈(TTM)'}.issubset(group.columns):
            mask = (group['市盈(TTM)'] == 0) & (group['AB股总市值'] > 0) & (group['净利润(亿)'] > 0)
            group.loc[mask, '市盈(TTM)'] = group.loc[mask, 'AB股总市值'] / group.loc[mask, '净利润(亿)']
            repair_count += mask.sum()

        # 修复市净率
        if {'AB股总市值', '净资产(亿)', '市净率'}.issubset(group.columns):
            mask = (group['市净率'] == 0) & (group['AB股总市值'] > 0) & (group['净资产(亿)'] > 0)
            group.loc[mask, '市净率'] = group.loc[mask, 'AB股总市值'] / group.loc[mask, '净资产(亿)']
            repair_count += mask.sum()

        # 4. 资产负债率、毛利率等财务比率的修复
        logger.info("正在修复财务比率...")
        # 修复资产负债率%
        if {'净资产(亿)', '少数股权(亿)', '总资产(亿)', '资产负债率%'}.issubset(group.columns):
            mask = (group['资产负债率%'] == 0) & (group['总资产(亿)'] > 0)
            group.loc[mask, '资产负债率%'] = (1 - (group.loc[mask, '净资产(亿)'] + group.loc[mask, '少数股权(亿)']) / \
                                            group.loc[mask, '总资产(亿)']) * 100
            repair_count += mask.sum()

        # 修复毛利率%
        if {'营业收入(亿)', '营业成本(亿)', '毛利率%'}.issubset(group.columns):
            mask = (group['毛利率%'] == 0) & (group['营业收入(亿)'] > 0)
            group.loc[mask, '毛利率%'] = (group.loc[mask, '营业收入(亿)'] - group.loc[mask, '营业成本(亿)']) / \
                                       group.loc[mask, '营业收入(亿)'] * 100
            repair_count += mask.sum()

        # 修复营业利润率%
        if {'营业利润(亿)', '营业收入(亿)', '营业利润率%'}.issubset(group.columns):
            mask = (group['营业利润率%'] == 0) & (group['营业收入(亿)'] > 0)
            group.loc[mask, '营业利润率%'] = group.loc[mask, '营业利润(亿)'] / group.loc[mask, '营业收入(亿)'] * 100
            repair_count += mask.sum()

        # 修复净益率%
        if {'净利润(亿)', '净资产(亿)', '净益率%'}.issubset(group.columns):
            mask = (group['净益率%'] == 0) & (group['净资产(亿)'] > 0)
            group.loc[mask, '净益率%'] = group.loc[mask, '净利润(亿)'] / group.loc[mask, '净资产(亿)'] * 100
            repair_count += mask.sum()

        # 5. 每股指标的修复
        logger.info("正在修复每股指标...")
        # 修复每股收益
        if {'净利润(亿)', '总股本(亿)', '每股收益'}.issubset(group.columns):
            mask = (group['每股收益'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '每股收益'] = group.loc[mask, '净利润(亿)'] / group.loc[mask, '总股本(亿)']
            repair_count += mask.sum()

        # 修复每股净资
        if {'净资产(亿)', '总股本(亿)', '每股净资'}.issubset(group.columns):
            mask = (group['每股净资'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '每股净资'] = group.loc[mask, '净资产(亿)'] / group.loc[mask, '总股本(亿)']
            repair_count += mask.sum()

        # 6. 流通比例、股本结构等指标的修复
        logger.info("正在修复流通比例和股本结构...")
        # 修复流通比例Z%
        if {'流通股(亿)', '总股本(亿)', '流通比例Z%'}.issubset(group.columns):
            mask = (group['流通比例Z%'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '流通比例Z%'] = group.loc[mask, '流通股(亿)'] / group.loc[mask, '总股本(亿)'] * 100
            repair_count += mask.sum()

        # 使用前值填充的指标列表
        invariant_cols = [
            '总股本(亿)', 'B/A股(亿)', 'H股(亿)', '流通股(亿)', '流通股本Z',
            '总资产(亿)', '净资产(亿)', '少数股权(亿)', '流动资产(亿)', '固定资产(亿)',
            '无形资产(亿)', '流动负债(亿)', '货币资金(亿)', '存货(亿)', '应收账款(亿)',
            '合同负债(亿)', '资本公积金(亿)', '营业收入(亿)', '营业成本(亿)', '营业利润(亿)',
            '投资收益(亿)', '净利润(亿)', '扣非净利润(亿)', '未分利润(亿)', '经营现金流(亿)',
            '总现金流(亿)', '股东人数', '员工人数'
        ]

        logger.info("正在修复基本不变的财务指标...")
        for col in invariant_cols:
            if col in group.columns:
                # 使用前值填充0值
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].ffill().bfill().fillna(0)
                repair_count += (group[col] == 0).sum()

        logger.info(f"财务指标修复完成，共修复 {repair_count} 处数据")
        return group, repair_count
