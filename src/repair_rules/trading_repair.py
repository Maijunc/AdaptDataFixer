#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易数据修复规则
"""
from typing import Tuple
import pandas as pd
import numpy as np
from .base_rule import BaseRepairRule
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TradingRepairRule(BaseRepairRule):
    """交易数据修复规则，处理与交易相关的数据修复"""

    def __init__(self):
        """初始化交易数据修复规则"""
        super().__init__(
            name="交易数据修复",
            description="修复金额、换手率、委托数据等交易相关指标"
        )

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用交易数据修复规则到单个股票组

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        repair_count = 0

        # 1. 金额修复
        logger.debug("正在修复金额数据...")
        # 检查列是否存在
        amount_cols = ['总金额', '2分钟金额', '开盘金额', '昨成交额', '3日成交额']
        price_cols = ['均价', '现价']
        
        # 找到可用的列
        amount_col = next((col for col in amount_cols if col in group.columns), None)
        price_col = next((col for col in price_cols if col in group.columns), None)
        
        if amount_col and price_col:
            # 使用前值填充0值
            mask_amount = (group[amount_col] == 0)
            if mask_amount.any():
                group[amount_col] = group[amount_col].replace(0, np.nan)
                group[amount_col] = group[amount_col].ffill().bfill().fillna(0)
                repair_count += mask_amount.sum()

        # 2. 换手率修复
        logger.debug("正在修复换手率...")
        turnover_cols = ['换手%', '换手Z', '换手']
        capital_cols = ['流通股本Z', '流通股(亿)', '流通股本']
        amount_cols = ['总金额', '总量']
        
        # 找到可用的列
        turnover_col = next((col for col in turnover_cols if col in group.columns), None)
        capital_col = next((col for col in capital_cols if col in group.columns), None)
        amount_col = next((col for col in amount_cols if col in group.columns), None)
        
        if turnover_col and capital_col and amount_col:
            mask = (group[turnover_col] == 0) & (group[amount_col] > 0) & (group[capital_col] > 0)
            if mask.any():
                # 使用前值填充0值
                group[turnover_col] = group[turnover_col].replace(0, np.nan)
                group[turnover_col] = group[turnover_col].ffill().bfill().fillna(0)
                repair_count += mask.sum()

        # 3. 委托买卖量修复
        logger.debug("正在修复委托买卖量...")
        buy_sell_cols = {
            '委买一': '委卖一',
            '委买二': '委卖二',
            '委买三': '委卖三',
            '委买四': '委卖四',
            '委买五': '委卖五'
        }

        for buy_col, sell_col in buy_sell_cols.items():
            if {buy_col, sell_col}.issubset(group.columns):
                # 使用前值填充0值
                for col in [buy_col, sell_col]:
                    mask = (group[col] == 0)
                    if mask.any():
                        group[col] = group[col].replace(0, np.nan)
                        group[col] = group[col].ffill().bfill()
                        repair_count += mask.sum()

        # 4. 大单、中单、小单数据修复
        logger.debug("正在修复大单、中单、小单数据...")
        order_cols = ['当日___超大单', '当日___大单', '当日___中单', '当日___小单']
        available_order_cols = [col for col in order_cols if col in group.columns]
        
        if available_order_cols:
            # 检查是否有0值需要修复
            for col in available_order_cols:
                mask = (group[col] == 0)
                if mask.any():
                    # 使用前后值填充
                    group[col] = group[col].replace(0, np.nan)
                    group[col] = group[col].ffill().bfill().fillna(0)
                    repair_count += mask.sum()
            
            # 确保各类单子的比例合理
            total_orders = group[available_order_cols].sum(axis=1)
            non_zero_total = total_orders > 0
            
            if non_zero_total.any():
                # 计算每种单子的平均占比
                avg_ratios = group.loc[non_zero_total, available_order_cols].div(total_orders[non_zero_total], axis=0).mean()
                
                # 对于总和为0的行，使用平均比例填充
                zero_total = total_orders == 0
                if zero_total.any():
                    # 使用总金额或总量作为参考
                    reference_cols = ['总金额', '总量']
                    reference_col = next((col for col in reference_cols if col in group.columns), None)
                    
                    if reference_col:
                        reference_values = group.loc[zero_total, reference_col]
                        non_zero_ref = reference_values > 0
                        
                        if non_zero_ref.any():
                            # 计算参考值的平均值
                            avg_ref = reference_values[non_zero_ref].mean()
                            
                            for col in available_order_cols:
                                group.loc[zero_total & non_zero_ref, col] = reference_values[non_zero_ref] * avg_ratios[col]
                                repair_count += (zero_total & non_zero_ref).sum()

        # 5. 主力资金流向数据修复
        logger.debug("正在修复主力资金流向数据...")
        flow_cols = ['主力净额', '主力净比%', '主力占比%', '主力净流入', '主力净流入占比']
        available_flow_cols = [col for col in flow_cols if col in group.columns]
        
        if available_flow_cols:
            # 修复主力净额
            if '主力净额' in available_flow_cols:
                mask = (group['主力净额'] == 0)
                if mask.any():
                    # 使用前后值填充
                    group['主力净额'] = group['主力净额'].replace(0, np.nan)
                    group['主力净额'] = group['主力净额'].ffill().bfill().fillna(0)
                    repair_count += mask.sum()
            
            # 修复主力净比%
            if '主力净比%' in available_flow_cols and '主力净额' in available_flow_cols and '总金额' in group.columns:
                mask = (group['主力净比%'] == 0) & (group['主力净额'] != 0) & (group['总金额'] > 0)
                if mask.any():
                    group.loc[mask, '主力净比%'] = group.loc[mask, '主力净额'] / group.loc[mask, '总金额'] * 100
                    repair_count += mask.sum()
            
            # 修复主力占比%
            if '主力占比%' in available_flow_cols and '主力净流入' in available_flow_cols and '总金额' in group.columns:
                mask = (group['主力占比%'] == 0) & (group['主力净流入'] != 0) & (group['总金额'] > 0)
                if mask.any():
                    group.loc[mask, '主力占比%'] = abs(group.loc[mask, '主力净流入']) / group.loc[mask, '总金额'] * 100
                    repair_count += mask.sum()

        # 6. 委托笔数修复
        logger.debug("正在修复委托笔数...")
        order_count_cols = ['总委托笔数', '委托笔数']
        trade_count_cols = ['总成交笔数', '成交笔数']
        
        order_count_col = next((col for col in order_count_cols if col in group.columns), None)
        trade_count_col = next((col for col in trade_count_cols if col in group.columns), None)
        
        if order_count_col and trade_count_col:
            # 确保委托笔数大于等于成交笔数
            mask = (group[order_count_col] < group[trade_count_col])
            if mask.any():
                group.loc[mask, order_count_col] = group.loc[mask, trade_count_col]
                repair_count += mask.sum()

            # 修复0值
            mask = (group[order_count_col] == 0) & (group[trade_count_col] > 0)
            if mask.any():
                # 使用历史平均比例估算
                valid_rows = (group[order_count_col] > 0) & (group[trade_count_col] > 0)
                if valid_rows.any():
                    avg_ratio = group.loc[valid_rows, order_count_col].div(group.loc[valid_rows, trade_count_col]).mean()
                    group.loc[mask, order_count_col] = group.loc[mask, trade_count_col] * avg_ratio
                else:
                    # 如果没有有效的历史数据，使用默认比例
                    group.loc[mask, order_count_col] = group.loc[mask, trade_count_col] * 1.5
                repair_count += mask.sum()

        logger.debug(f"交易数据修复完成，共修复 {repair_count} 处数据")
        return group, repair_count
