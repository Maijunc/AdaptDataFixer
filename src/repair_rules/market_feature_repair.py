#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场特征修复规则
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # 需要显式启用
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)

class MarketFeatureRepairRule(BaseRepairRule):
    """市场特征修复规则，处理与市场特征相关的数据修复"""

    def __init__(self):
        """初始化市场特征修复规则"""
        super().__init__(
            name="市场特征修复",
            description="修复涨跌幅、强弱度等市场特征相关指标"
        )

    def _apply_to_group(self, group: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        应用市场特征修复规则到单个股票组

        Args:
            group (pd.DataFrame): 单个股票的数据，已按日期排序

        Returns:
            Tuple[pd.DataFrame, int]: 修复后的数据框和修复数量
        """
        repair_count = 0

        # 0. 涨幅%和涨跌幅的相互修复
        if '涨幅%' in group.columns and '涨跌幅' in group.columns:
            # 1. 处理涨幅%=0且涨跌幅≠0的情况（直接用涨跌幅覆盖涨幅%）
            mask1 = (group['涨幅%'] == 0) & (group['涨跌幅'] != 0)
            group.loc[mask1, '涨幅%'] = group.loc[mask1, '涨跌幅']
            repair_count += mask1.sum()

            # 2. 处理涨跌幅=0且涨幅%≠0的情况（直接用涨幅%覆盖涨跌幅）
            mask2 = (group['涨跌幅'] == 0) & (group['涨幅%'] != 0)
            group.loc[mask2, '涨跌幅'] = group.loc[mask2, '涨幅%']
            repair_count += mask2.sum()

            # 3. 处理涨跌幅≠涨幅%的情况（根据现价和昨收重新计算）
            if '现价' in group.columns and '昨收' in group.columns:
                # 计算理论涨幅%：(现价-昨收)/昨收*100
                calculated_pct = ((group['现价'] - group['昨收']) / group['昨收'] * 100).fillna(0)
                # 找出涨跌幅与涨幅%不匹配的行（考虑浮点数精度误差）
                mask3 = ~np.isclose(group['涨跌幅'], group['涨幅%'], atol=1e-6)
                # 覆盖涨幅%和涨跌幅
                group.loc[mask3, '涨幅%'] = calculated_pct[mask3]
                group.loc[mask3, '涨跌幅'] = calculated_pct[mask3]
                repair_count += mask3.sum()

        # 1. 涨跌幅相关指标修复
        logger.info("正在修复涨跌幅相关指标...")
        # 检查涨跌幅列是否存在
        change_cols = ['涨跌幅', '涨跌', '涨跌幅%', '日涨跌幅']
        change_col = next((col for col in change_cols if col in group.columns), None)

        if change_col:
            period_cols = {
                '3日涨幅%': 3,
                '5日涨幅%': 5,
                '10日涨幅%': 10,
                '20日涨幅%': 20,
                '60日涨幅%': 60
            }

            for col, period in period_cols.items():
                if col in group.columns:
                    # 只修复值为0的部分
                    mask = (group[col] == 0)
                    if mask.any():
                        # 使用min_periods=1确保即使数据不足也能计算
                        rolling_sum = group[change_col].rolling(window=period, min_periods=1).sum()
                        group.loc[mask, col] = rolling_sum.loc[mask]
                        # 处理可能的NaN值
                        group[col] = group[col].fillna(0)
                        repair_count += mask.sum()

        # 2. 年初至今%和一年涨幅%的修复
        logger.info("正在修复年初至今%和一年涨幅%...")
        # 检查涨跌幅列是否存在
        change_cols = ['涨跌幅', '涨跌', '涨跌幅%', '日涨跌幅']
        change_col = next((col for col in change_cols if col in group.columns), None)

        if '年初至今%' in group.columns and '日期' in group.columns and change_col:
            try:
                # 按年分组计算累计涨跌幅
                group['year'] = pd.to_datetime(group['日期']).dt.year
                for year in group['year'].unique():
                    year_data = group[group['year'] == year]

                    mask = (group['year'] == year) & (group['年初至今%'] == 0)
                    if mask.any():
                        group.loc[mask, '年初至今%'] = group.loc[mask, change_col].cumsum()
                        repair_count += mask.sum()

                group.drop('year', axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"年初至今%修复失败: {str(e)}")

        # 检查52周最高/最低列
        high_cols = ['52周最高', '52周高', '年最高']
        low_cols = ['52周最低', '52周低', '年最低']
        year_change_cols = ['一年涨幅%', '年涨幅%', '52周涨幅%']

        high_col = next((col for col in high_cols if col in group.columns), None)
        low_col = next((col for col in low_cols if col in group.columns), None)
        year_change_col = next((col for col in year_change_cols if col in group.columns), None)

        if high_col and low_col and year_change_col:
            mask = (group[year_change_col] == 0) & (group[high_col] > 0) & (group[low_col] > 0)
            group.loc[mask, year_change_col] = (group.loc[mask, high_col] - group.loc[mask, low_col]) / \
                                             group.loc[mask, low_col] * 100
            repair_count += mask.sum()

        # 3. 强弱度%的修复
        logger.info("正在修复强弱度%...")
        strength_cols = ['强弱度%', '强弱度', '相对强弱']
        strength_col = next((col for col in strength_cols if col in group.columns), None)

        if strength_col and change_col:
            # 按照公式：(100+上一个交易的强弱度%)/(100+上一个交易的涨跌幅)=(100+强弱度%)/(100+涨跌幅)
            prev_strength = group[strength_col].shift(1)
            prev_change = group[change_col].shift(1)
            current_change = group[change_col]

            mask = (group[strength_col] == 0)
            if mask.any():
                try:
                    group.loc[mask, strength_col] = ((100 + prev_strength) * (100 + current_change) / \
                                                  (100 + prev_change) - 100).fillna(0)
                    repair_count += mask.sum()
                except Exception as e:
                    logger.warning(f"强弱度%修复失败: {str(e)}")

        # 4. 开盘%、最高%、最低%、均涨幅%的修复
        logger.info("正在修复开盘%、最高%、最低%、均涨幅%...")

        # 检查昨收价列
        prev_close_cols = ['昨收', '昨收价', '前收盘']
        prev_close_col = next((col for col in prev_close_cols if col in group.columns), None)

        if prev_close_col:
            # 开盘%修复
            open_pct_cols = ['开盘%', '开盘涨幅%']
            open_cols = ['今开', '开盘价', '开盘']

            open_pct_col = next((col for col in open_pct_cols if col in group.columns), None)
            open_col = next((col for col in open_cols if col in group.columns), None)

            if open_pct_col and open_col:
                mask = (group[open_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, open_pct_col] = (group.loc[mask, open_col] - group.loc[mask, prev_close_col]) / \
                                                  group.loc[mask, prev_close_col] * 100
                    repair_count += mask.sum()

            # 最高%修复
            high_pct_cols = ['最高%', '最高涨幅%']
            high_cols = ['最高', '最高价', '高']

            high_pct_col = next((col for col in high_pct_cols if col in group.columns), None)
            high_col = next((col for col in high_cols if col in group.columns), None)

            if high_pct_col and high_col:
                mask = (group[high_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, high_pct_col] = (group.loc[mask, high_col] - group.loc[mask, prev_close_col]) / \
                                                  group.loc[mask, prev_close_col] * 100
                    repair_count += mask.sum()

            # 最低%修复
            low_pct_cols = ['最低%', '最低涨幅%']
            low_cols = ['最低', '最低价', '低']

            low_pct_col = next((col for col in low_pct_cols if col in group.columns), None)
            low_col = next((col for col in low_cols if col in group.columns), None)

            if low_pct_col and low_col:
                mask = (group[low_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, low_pct_col] = (group.loc[mask, low_col] - group.loc[mask, prev_close_col]) / \
                                                 group.loc[mask, prev_close_col] * 100
                    repair_count += mask.sum()

            # 均涨幅%修复
            avg_pct_cols = ['均涨幅%', '均价涨幅%']
            avg_cols = ['均价', '平均价', 'avg_price']

            avg_pct_col = next((col for col in avg_pct_cols if col in group.columns), None)
            avg_col = next((col for col in avg_cols if col in group.columns), None)

            if avg_pct_col and avg_col:
                mask = (group[avg_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, avg_pct_col] = (group.loc[mask, avg_col] - group.loc[mask, prev_close_col]) / \
                                                 group.loc[mask, prev_close_col] * 100
                    repair_count += mask.sum()

        # 5. 回头波%、攻击波%的修复
        logger.info("正在修复回头波%、攻击波%...")
        change_cols = ['涨跌幅', '涨跌', '涨跌幅%', '日涨跌幅']
        change_col = next((col for col in change_cols if col in group.columns), None)

        for wave_col in ['回头波%', '攻击波%']:
            if wave_col in group.columns and change_col:
                # 尝试使用涨跌幅和主力净比%计算
                net_ratio_cols = ['主力净比%', '净比%', '主力净额比%']
                net_ratio_col = next((col for col in net_ratio_cols if col in group.columns), None)

                if net_ratio_col:
                    mask = (group[wave_col] == 0) & (group[change_col] != 0) & (group[net_ratio_col] != 0)
                    if mask.any():
                        group.loc[mask, wave_col] = group.loc[mask, change_col] * group.loc[mask, net_ratio_col] / 100
                        repair_count += mask.sum()

                # 对于仍然为0的值，使用前后均值修复
                mask = (group[wave_col] == 0)
                if mask.any():
                    prev_val = group[wave_col].shift(1).fillna(0)
                    next_val = group[wave_col].shift(-1).fillna(0)
                    group.loc[mask, wave_col] = prev_val[mask] * 0.6 + next_val[mask] * 0.4
                    repair_count += mask.sum()

        # 6. 距5日线%的修复
        logger.info("正在修复距5日线%...")
        ma_dist_cols = ['距5日线%', '距MA5%', '5日线距离%']
        price_cols = ['现价', '收盘价', '收盘', '价格']

        ma_dist_col = next((col for col in ma_dist_cols if col in group.columns), None)
        price_col = next((col for col in price_cols if col in group.columns), None)

        if ma_dist_col and price_col:
            try:
                # 计算5日均线
                ma5 = group[price_col].rolling(window=5, min_periods=1).mean()
                mask = (group[ma_dist_col] == 0) & (ma5 > 0)
                if mask.any():
                    group.loc[mask, ma_dist_col] = (group.loc[mask, price_col] - ma5[mask]) / ma5[mask] * 100
                    repair_count += mask.sum()
            except Exception as e:
                logger.warning(f"距5日线%修复失败: {str(e)}")

        # 7. MICE多元插补修复
        logger.info("正在进行MICE多元插补修复...")
        mice_cols = [
            '开盘抢筹%', '现均差%', '总委比%', '总委量差', '主力净额',
            '主力净比%', '主力占比%', '净买率%', '总卖占比%', '总撤委比%',
            '总委托笔数', '总成交笔数', '散户单增比', '当日___超大单',
            '当日___大单', '当日___中单', '当日___小单'
        ]

        # 筛选需要插补的列
        cols_to_impute = [col for col in mice_cols if col in group.columns]

        if cols_to_impute:
            # 检查是否有0值需要插补
            data_to_impute = group[cols_to_impute].copy()
            mask_zero = (data_to_impute == 0)

            if mask_zero.any().any():
                # 检查每列是否有足够的非零值用于插补
                valid_cols = []
                for col in cols_to_impute:
                    non_zero_count = (data_to_impute[col] != 0).sum()
                    if non_zero_count >= 2:  # 至少需要2个非零值才能进行插补
                        valid_cols.append(col)

                if valid_cols:
                    try:
                        # 只使用有效的列进行插补
                        data_to_impute = data_to_impute[valid_cols]
                        data_to_impute[data_to_impute == 0] = np.nan

                        # 确保数据类型为float
                        data_to_impute = data_to_impute.astype(float)

                        # 创建MICE插补器
                        imputer = IterativeImputer(
                            max_iter=10,           # 最大迭代次数
                            random_state=42,       # 随机种子，确保结果可重复
                            min_value=0,           # 插补值的最小值
                            verbose=0,             # 不显示迭代过程
                            n_nearest_features=5   # 用于每个特征的最近邻特征数
                        )

                        # 执行插补
                        imputed_array = imputer.fit_transform(data_to_impute)
                        imputed_data = pd.DataFrame(
                            imputed_array,
                            columns=data_to_impute.columns,
                            index=data_to_impute.index
                        )

                        # 只替换原始为0的值，并确保插补值非负
                        for col in valid_cols:
                            mask = (group[col] == 0)
                            if mask.any():
                                imputed_values = imputed_data[col]
                                # 确保插补值非负
                                imputed_values = np.maximum(imputed_values, 0)
                                group.loc[mask, col] = imputed_values[mask]
                                repair_count += mask.sum()

                    except Exception as e:
                        logger.warning(f"MICE插补失败: {str(e)}")
                        # 使用前后均值作为备选方案，并确保非负
                        for col in valid_cols:
                            mask = (group[col] == 0)
                            if mask.any():
                                prev_val = group[col].shift(1).fillna(method='ffill')
                                next_val = group[col].shift(-1).fillna(method='bfill')
                                avg_val = (prev_val + next_val) / 2
                                # 确保插补值非负
                                avg_val = np.maximum(avg_val, 0)
                                group.loc[mask, col] = avg_val[mask]
                                repair_count += mask.sum()

        logger.info(f"市场特征修复完成，共修复 {repair_count} 处数据")
        return group, repair_count
