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
                group.loc[mask3, '涨幅%'] = (calculated_pct[mask3]).__round__(2)
                group.loc[mask3, '涨跌幅'] = (calculated_pct[mask3]).__round__(2)
                repair_count += mask3.sum()

        if '涨跌' in group.columns:
            mask = (group['涨跌'] == 0)
            if mask.any():
                if '现价' in group.columns and '昨收' in group.columns:
                    group.loc[mask, '涨跌'] = group.loc[mask, '现价'] - group.loc[mask, '昨收']
                    repair_count += mask.sum()

        if '振幅%' in group.columns:
            mask = (group['振幅%'] == 0)
            if mask.any():
                group.loc[mask, '振幅%'] = abs(group.loc[mask, '最高'] - group.loc[mask, '最低'])
                repair_count += mask.sum()

        if '内盘' in group.columns:
            # 检查是否存在总量和内盘列（用于准确计算）
            if '总量' in group.columns and '外盘' in group.columns:
                # 先尝试用总量-内盘计算外盘（准确方法）
                valid_mask = (group['总量'] != 0) & (group['外盘'] != 0)
                if valid_mask.any():
                    group.loc[valid_mask, '内盘'] = group.loc[valid_mask, '总量'] - group.loc[valid_mask, '外盘']
                    repair_count += valid_mask.sum()

            # 无准确方法时的替补策略（总量或内盘数据无效）
            invalid_mask = (group['内盘'] == 0) | (~(group['总量'] != 0) | ~(group['外盘'] != 0))
            if invalid_mask.any():
                # 计算上一交易日和下一交易日的均值
                group.loc[invalid_mask, '内盘'] = (group['内盘'].shift() + group['内盘'].shift(-1)) / 2
                # 处理不存在下一交易日的情况（用前一交易日的值覆盖）
                group.loc[invalid_mask & (group['内盘'].isna()), '内盘'] = group['内盘'].shift()
                # 处理所有数据都缺失的情况（向前填充）
                group['内盘'] = group['内盘'].ffill()
                repair_count += invalid_mask.sum()

        if '外盘' in group.columns:
            # 检查是否存在总量和内盘列（用于准确计算）
            if '总量' in group.columns and '内盘' in group.columns:
                # 先尝试用总量-内盘计算外盘（准确方法）
                valid_mask = (group['总量'] != 0) & (group['内盘'] != 0)
                if valid_mask.any():
                    group.loc[valid_mask, '外盘'] = group.loc[valid_mask, '总量'] - group.loc[valid_mask, '内盘']
                    repair_count += valid_mask.sum()

            # 无准确方法时的替补策略（总量或内盘数据无效）
            invalid_mask = (group['外盘'] == 0) | (~(group['总量'] != 0) | ~(group['内盘'] != 0))
            if invalid_mask.any():
                # 计算上一交易日和下一交易日的均值
                group.loc[invalid_mask, '外盘'] = (group['外盘'].shift() + group['外盘'].shift(-1)) / 2
                # 处理不存在下一交易日的情况（用前一交易日的值覆盖）
                group.loc[invalid_mask & (group['外盘'].isna()), '外盘'] = group['外盘'].shift()
                # 处理所有数据都缺失的情况（向前填充）
                group['外盘'] = group['外盘'].ffill()
                repair_count += invalid_mask.sum()

        if '内外比' in group.columns:
            # 检查是否存在内盘和外盘列（用于准确计算）
            if '内盘' in group.columns and '外盘' in group.columns:
                # 先尝试用内盘/外盘计算内外比（准确方法）
                mask = (group['内盘'] != 0) & (group['外盘'] != 0) & (group['外盘'] != 0)
                if mask.any():
                    group.loc[mask, '内外比'] = group.loc[mask, '内盘'] / group.loc[mask, '外盘']
                    repair_count += mask.sum()

            # 无准确方法时的替补策略（内盘/外盘数据无效或计算结果异常）
            invalid_mask = (group['内外比'] == 0) | ~mask | (group['内外比'] < 0)
            if invalid_mask.any():
                # 方法1：用前后交易日的均值填充
                group.loc[invalid_mask, '内外比'] = (group['内外比'].shift() + group['内外比'].shift(-1)) / 2
                # 方法2：处理不存在下一交易日的情况（用前一交易日的值）
                group.loc[invalid_mask & (group['内外比'].isna()), '内外比'] = group['内外比'].shift()
                # 方法3：处理极端异常值（如负数，设为1.0作为中性值）
                group.loc[group['内外比'] < 0, '内外比'] = 1.0
                # 最终向前填充处理全量缺失
                group['内外比'] = group['内外比'].ffill()
                repair_count += invalid_mask.sum()

        if '买量' in group.columns:
            # 买量修复：若为0则设为收盘价（根据文档规则，可能存在业务逻辑简化）
            mask = (group['买量'] == 0)
            if mask.any():
                group.loc[mask, '买量'] = group.loc[mask, '收盘价']  # 注：实际业务中可能需用委托数据，此处按文档规则处理
                repair_count += mask.sum()

        if '卖量' in group.columns:
            # 卖量修复：若为0则设为收盘价（根据文档规则）
            mask = (group['卖量'] == 0)
            if mask.any():
                group.loc[mask, '卖量'] = group.loc[mask, '收盘价']  # 注：同上
                repair_count += mask.sum()

            # 进阶替补策略：若收盘价也为0，用前后交易日的买量/卖量均值填充
            if '买量' in group.columns and '卖量' in group.columns:
                invalid_buy = (group['买量'] == 0) & (group['收盘价'] == 0)
                invalid_sell = (group['卖量'] == 0) & (group['收盘价'] == 0)

                if invalid_buy.any():
                    group.loc[invalid_buy, '买量'] = (group['买量'].shift() + group['买量'].shift(-1)) / 2
                    group.loc[invalid_buy & group['买量'].isna(), '买量'] = group['买量'].shift()
                    repair_count += invalid_buy.sum()

                if invalid_sell.any():
                    group.loc[invalid_sell, '卖量'] = (group['卖量'].shift() + group['卖量'].shift(-1)) / 2
                    group.loc[invalid_sell & group['卖量'].isna(), '卖量'] = group['卖量'].shift()
                    repair_count += invalid_sell.sum()


        if '委比%' in group.columns:
            # 检查买量和卖量是否存在
            if '买量' in group.columns and '卖量' in group.columns:
                valid_mask = (group['买量'] != 0) & (group['卖量'] != 0)
                if valid_mask.any():
                    # 计算委比，处理分母为0的情况
                    group.loc[valid_mask, '委比%'] = ((group.loc[valid_mask, '买量'] - group.loc[valid_mask, '卖量']) /
                                                      (group.loc[valid_mask, '买量'] + group.loc[
                                                          valid_mask, '卖量'])) * 100
                    # 避免除零错误，当买量+卖量=0时设为0
                    zero_denominator = (group['买量'] + group['卖量'] == 0)
                    group.loc[zero_denominator, '委比%'] = 0
                    repair_count += valid_mask.sum()

            # 替补策略：买量/卖量无效时用历史数据填充
            invalid_mask = (group['委比%'] == 0) | ~valid_mask
            if invalid_mask.any():
                group.loc[invalid_mask, '委比%'] = (group['委比%'].shift() + group['委比%'].shift(-1)) / 2
                group.loc[invalid_mask & group['委比%'].isna(), '委比%'] = group['委比%'].shift()
                group['委比%'] = group['委比%'].ffill()
                repair_count += invalid_mask.sum()

        if '量涨速%' in group.columns:
            invalid_mask = (group['量涨速%'] == 0) | (group['量涨速%'].isna()) | (group['量涨速%'] > 1000)  # 过滤异常大值
            if invalid_mask.any():
                group.loc[invalid_mask, '量涨速%'] = group['量涨速%'].ffill()
                group.loc[invalid_mask & group['量涨速%'].isna(), '量涨速%'] = group['量涨速%'].ffill()
                repair_count += invalid_mask.sum()

        if '活跃度' in group.columns:
            # 检查相关列是否存在
            required_cols = ['换手%', '量比', '振幅%']
            if all(col in group.columns for col in required_cols):
                valid_mask = (group['换手%'] != 0) & (group['量比'] != 0) & (group['振幅%'] != 0)
                if valid_mask.any():
                    group.loc[valid_mask, '活跃度'] = (group.loc[valid_mask, '换手%'] +
                                                       group.loc[valid_mask, '量比'] +
                                                       group.loc[valid_mask, '振幅%'])
                    repair_count += valid_mask.sum()

            # 替补策略：相关列无效时用历史活跃度填充
            invalid_mask = (group['活跃度'] == 0) | ~valid_mask
            if invalid_mask.any():
                group.loc[invalid_mask, '活跃度'] = group['活跃度'].ffill()
                group.loc[invalid_mask & group['活跃度'].isna(), '活跃度'] = group['活跃度'].fillna(
                    (group['活跃度'].shift() + group['活跃度'].shift(-1)) / 2
                )
                repair_count += invalid_mask.sum()



        # 1. 涨跌幅相关指标修复
        logger.info("正在修复涨跌幅相关指标...")
        # 检查涨跌幅列是否存在
        change_col = '涨跌幅' if '涨跌幅' in group.columns else None

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
                    mask = (group[col] == 0)
                    if mask.any():
                        # 计算前一日的X日涨幅%
                        prev_day_col = group[col].shift(1)
                        # 计算前第X日的单日涨跌幅（修正此处，原为group[col].shift(period)）
                        prev_period_change = group[change_col].shift(period)
                        # 计算本日的涨跌幅
                        current_change = group[change_col]

                        # 修正公式：今日X日涨幅 = 昨日X日涨幅 - X天前单日涨跌幅 + 今日单日涨跌幅
                        calculated_col = prev_day_col - prev_period_change + current_change

                        # 覆盖原值为0的位置
                        group.loc[mask, col] = (calculated_col.loc[mask]).__round__(2)
                        group[col] = group[col].fillna(0)
                        repair_count += mask.sum()
                        logger.info(f"修复了 {mask.sum()} 处 {col} 的值")

        # 2. 年初至今%和一年涨幅%的修复
        logger.info("正在修复年初至今%和一年涨幅%...")
        # 检查涨跌幅列是否存在
        change_col = '涨跌幅' if '涨跌幅' in group.columns else None

        # 1. 修复年初至今%
        if '年初至今%' in group.columns and '日期' in group.columns and change_col:
            try:
                # 按年分组计算累计涨跌幅
                group['year'] = pd.to_datetime(group['日期']).dt.year
                for year in group['year'].unique():
                    year_data = group[group['year'] == year]

                    mask = (group['year'] == year) & (group['年初至今%'] == 0)
                    if mask.any():
                        # 计算上一个交易日的年初至今%
                        prev_day_col = year_data['年初至今%'].shift(1)
                        # 计算本日的涨跌幅
                        current_change = year_data[change_col]

                        # 根据公式计算新的年初至今%
                        calculated_col = prev_day_col + current_change

                        # 覆盖原值为0的位置
                        group.loc[mask, '年初至今%'] = (calculated_col.loc[mask]).__round__(2)
                        repair_count += mask.sum()
                group.drop('year', axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"年初至今%修复失败: {str(e)}")

        # 检查52周最高/最低列
        high_col = '52周最高' if '52周最高' in group.columns else None
        low_col = '52周最低' if '52周最低' in group.columns else None
        year_change_col = '一年涨幅%' if '一年涨幅%' in group.columns else None

        # 2. 修复一年涨幅%
        if high_col and low_col and year_change_col:
            mask = (group[year_change_col] == 0) & (group[high_col] > 0) & (group[low_col] > 0)
            if mask.any():
                # 计算上一个交易日的一年涨幅%
                prev_day_col = group[year_change_col].shift(1)
                # 计算本日的涨跌幅
                current_change = group[change_col]

                # 根据公式计算新的一年涨幅%
                calculated_col = prev_day_col + current_change

                # 覆盖原值为0的位置
                group.loc[mask, year_change_col] = (calculated_col.loc[mask]).__round__(2)
                repair_count += mask.sum()

        # 3. 强弱度%的修复
        logger.info("正在修复强弱度%...")
        strength_col = '强弱度%' if '强弱度%' in group.columns else None
        change_col = '涨跌幅' if '涨跌幅' in group.columns else None

        if strength_col and change_col:
            # 按照公式：(100+上一个交易的强弱度%)/(100+上一个交易的涨跌幅)=(100+强弱度%)/(100+涨跌幅)
            prev_strength = group[strength_col].shift(1)
            prev_change = group[change_col].shift(1)
            current_change = group[change_col]

            mask = (group[strength_col] == 0)
            if mask.any():
                try:
                    # 计算新的强弱度%
                    calculated_strength = ((100 + prev_strength) / (100 + prev_change)) * (100 + current_change) - 100
                    group.loc[mask, strength_col] = (calculated_strength[mask]).__round__(2)
                    repair_count += mask.sum()
                    logger.info(f"修复了 {mask.sum()} 处 {strength_col} 的值")
                except Exception as e:
                    logger.warning(f"强弱度%修复失败: {str(e)}", exc_info=True)



        # 4. 开盘%、最高%、最低%、均涨幅%的修复
        logger.info("正在修复开盘%、最高%、最低%、均涨幅%...")

        # 检查昨收列
        prev_close_col = '昨收' if '昨收' in group.columns else None

        if prev_close_col:
            # 开盘%修复
            open_pct_col = '开盘%' if '开盘%' in group.columns else None
            open_col = '今开' if '今开' in group.columns else None

            if open_pct_col and open_col:
                mask = (group[open_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, open_pct_col] = ((group.loc[mask, open_col] - group.loc[mask, prev_close_col]) / \
                                                  group.loc[mask, prev_close_col] * 100).__round__(2)
                    repair_count += mask.sum()

            # 最高%修复
            high_pct_col = '最高%' if '最高%' in group.columns else None
            high_col = '最高' if '最高' in group.columns else None

            if high_pct_col and high_col:
                mask = (group[high_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, high_pct_col] = ((group.loc[mask, high_col] - group.loc[mask, prev_close_col]) / \
                                                  group.loc[mask, prev_close_col] * 100).__round__(2)
                    repair_count += mask.sum()

            # 最低%修复
            low_pct_col = '最低%' if '最低%' in group.columns else None
            low_col = '最低' if '最低' in group.columns else None

            if low_pct_col and low_col:
                mask = (group[low_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, low_pct_col] = ((group.loc[mask, low_col] - group.loc[mask, prev_close_col]) / \
                                                 group.loc[mask, prev_close_col] * 100).__round__(2)
                    repair_count += mask.sum()

            # 均涨幅%修复
            avg_pct_cols = ['均涨幅%', '均价涨幅%']
            avg_cols = ['均价', '平均价', 'avg_price']

            avg_pct_col = next((col for col in avg_pct_cols if col in group.columns), None)
            avg_col = next((col for col in avg_cols if col in group.columns), None)

            if avg_pct_col and avg_col:
                mask = (group[avg_pct_col] == 0) & (group[prev_close_col] > 0)
                if mask.any():
                    group.loc[mask, avg_pct_col] = ((group.loc[mask, avg_col] - group.loc[mask, prev_close_col]) / \
                                                 group.loc[mask, prev_close_col] * 100).__round__(2)
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
                        group.loc[mask, wave_col] = (group.loc[mask, change_col] * group.loc[mask, net_ratio_col] / 100
                                                     ).__round__(2)
                        repair_count += mask.sum()

                # 对于仍然为0的值，使用前后均值修复
                mask = (group[wave_col] == 0)
                if mask.any():
                    prev_val = group[wave_col].shift(1).fillna(0)
                    next_val = group[wave_col].shift(-1).fillna(0)
                    group.loc[mask, wave_col] = (prev_val[mask] * 0.6 + next_val[mask] * 0.4).__round__(2)
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
                    group.loc[mask, ma_dist_col] = ((group.loc[mask, price_col] - ma5[mask]) / ma5[mask] * 100)\
                        .__round__(2)
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
