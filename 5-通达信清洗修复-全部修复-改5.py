import pandas as pd
import numpy as np
import warnings
from scipy.stats import gmean

# 忽略警告信息
warnings.filterwarnings('ignore')


def preprocess_data(df):
    """数据预处理"""
    # 确保日期列存在并转换为datetime
    if '日期' not in df.columns:
        raise ValueError("数据中缺少'日期'列")
    df['日期'] = pd.to_datetime(df['日期'])

    # 确保代码列存在
    if '代码' not in df.columns:
        raise ValueError("数据中缺少'代码'列")

    # 检查关键列是否存在
    required_cols = ['昨收', '现价', '收盘价', '今开', '涨幅%', '涨跌幅']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 处理空值
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def process_fengdan_data(df):
    """封单额数据修复，增强容错性和覆盖范围"""
    repair_count = 0

    # 按股票代码分组处理
    for code, group in df.groupby('代码'):
        group = group.sort_values('日期')
        indices = group.index.tolist()

        for i in range(len(indices)):
            current_idx = indices[i]
            current_fengdan = df.at[current_idx, '封单额']

            # 规则1：当前日封单额覆盖下一日的"昨封单额"
            if i + 1 < len(indices) and current_fengdan != 0:
                next_idx = indices[i + 1]
                if df.at[next_idx, '昨封单额'] == 0:
                    df.at[next_idx, '昨封单额'] = current_fengdan
                    repair_count += 1

            # 规则2：当前日封单额覆盖再下一日的"前封单额"
            if i + 2 < len(indices) and current_fengdan != 0:
                next_next_idx = indices[i + 2]
                if df.at[next_next_idx, '前封单额'] == 0:
                    df.at[next_next_idx, '前封单额'] = current_fengdan
                    repair_count += 1

            # 规则3：下一日的"昨封单额"可以覆盖当前日的"封单额"
            if i + 1 < len(indices):
                next_idx = indices[i + 1]
                next_zuofengdan = df.at[next_idx, '昨封单额']
                if next_zuofengdan != 0 and df.at[current_idx, '封单额'] == 0:
                    df.at[current_idx, '封单额'] = next_zuofengdan
                    repair_count += 1

            # 规则4：下一日的"昨封单额"可以覆盖再下一日的"前封单额"
            if i + 1 < len(indices):
                next_idx = indices[i + 1]
                next_zuofengdan = df.at[next_idx, '昨封单额']
                if i + 2 < len(indices) and next_zuofengdan != 0:
                    next_next_idx = indices[i + 2]
                    if df.at[next_next_idx, '前封单额'] == 0:
                        df.at[next_next_idx, '前封单额'] = next_zuofengdan
                        repair_count += 1

            # 规则5：再下一日的"前封单额"可以覆盖当前日的"封单额"
            if i + 2 < len(indices):
                next_next_idx = indices[i + 2]
                next_next_qianfengdan = df.at[next_next_idx, '前封单额']
                if next_next_qianfengdan != 0 and df.at[current_idx, '封单额'] == 0:
                    df.at[current_idx, '封单额'] = next_next_qianfengdan
                    repair_count += 1

            # 规则6：再下一日的"前封单额"可以覆盖下一日的"昨封单额"
            if i + 2 < len(indices):
                next_next_idx = indices[i + 2]
                next_next_qianfengdan = df.at[next_next_idx, '前封单额']
                if i + 1 < len(indices) and next_next_qianfengdan != 0:
                    next_idx = indices[i + 1]
                    if df.at[next_idx, '昨封单额'] == 0:
                        df.at[next_idx, '昨封单额'] = next_next_qianfengdan
                        repair_count += 1

    print(f"封单额数据修复完成，共修复 {repair_count} 处数据")
    return df


def process_price_data(df):
    """价格数据修复，增强鲁棒性"""
    repair_count = 0

    for code, group in df.groupby('代码'):
        group = group.sort_values('日期')
        indices = group.index.tolist()

        for i in range(len(indices)):
            current_idx = indices[i]
            original_close = df.at[current_idx, '昨收']
            original_price = df.at[current_idx, '现价']

            # 规则1：当前现价 → 下一日昨收（仅当下一日昨收为0时）
            if i + 1 < len(indices) and original_price != 0:
                next_idx = indices[i + 1]
                if df.at[next_idx, '昨收'] == 0:
                    df.at[next_idx, '昨收'] = original_price
                    repair_count += 1

            # 规则2：下一日昨收 → 当前现价（仅当当前现价为0时）
            if i + 1 < len(indices) and df.at[current_idx, '现价'] == 0:
                next_idx = indices[i + 1]
                if df.at[next_idx, '昨收'] != 0:
                    df.at[current_idx, '现价'] = df.at[next_idx, '昨收']
                    repair_count += 1

    print(f"价格数据修复完成，共修复 {repair_count} 处数据")
    return df


def process_amount_data(df):
    """成交额数据修复，增强逻辑完整性"""
    repair_count = 0

    # 按股票代码分组处理
    for code, group in df.groupby('代码'):
        group = group.sort_values('日期')
        indices = group.index.tolist()

        for i in range(len(indices)):
            current_idx = indices[i]
            current_total_amount = df.at[current_idx, '总金额']
            current_prev_amount = df.at[current_idx, '昨成交额']

            # 规则1：当前日总金额覆盖下一日的"昨成交额"
            if i + 1 < len(indices) and current_total_amount != 0:
                next_idx = indices[i + 1]
                if df.at[next_idx, '昨成交额'] == 0:
                    df.at[next_idx, '昨成交额'] = current_total_amount
                    repair_count += 1

            # 规则2：下一日的"昨成交额"可以覆盖当前日的"总金额"
            if i + 1 < len(indices):
                next_idx = indices[i + 1]
                next_prev_amount = df.at[next_idx, '昨成交额']
                if next_prev_amount != 0 and df.at[current_idx, '总金额'] == 0:
                    df.at[current_idx, '总金额'] = next_prev_amount
                    repair_count += 1

            # 规则3：开盘金额 → 昨开盘金额（下一日）
            current_open_amount = df.at[current_idx, '开盘金额']
            if i + 1 < len(indices) and current_open_amount != 0:
                next_idx = indices[i + 1]
                if df.at[next_idx, '昨开盘金额'] == 0:
                    df.at[next_idx, '昨开盘金额'] = current_open_amount
                    repair_count += 1

            # 规则4：昨开盘金额 → 开盘金额（当前日）
            if i > 0:
                prev_idx = indices[i - 1]
                prev_zuo_open_amount = df.at[prev_idx, '昨开盘金额']
                if prev_zuo_open_amount != 0 and df.at[current_idx, '开盘金额'] == 0:
                    df.at[current_idx, '开盘金额'] = prev_zuo_open_amount
                    repair_count += 1

    print(f"成交额数据修复完成，共修复 {repair_count} 处数据")
    return df


def repair_stock_data(df, Fix_all=True):
    """
    修复股票数据的主函数（优化版）
    :param df: 输入的DataFrame
    :param Fix_all: 是否全量修复，True为修复所有数据，False只修复最后一天
    :return: 修复后的DataFrame
    """
    # 保存原始列顺序
    original_columns = df.columns.tolist()

    # 预处理数据
    df = preprocess_data(df)

    # 如果只修复最后一天，则只保留最近10天的数据
    if not Fix_all:
        recent_dates = df['日期'].unique()
        recent_dates.sort()
        recent_dates = recent_dates[-10:] if len(recent_dates) >= 10 else recent_dates
        df = df[df['日期'].isin(recent_dates)]

    # 删除不需要的列
    for col in ['涨速%', '短换手%', '开盘竞价']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # 更新原始列顺序
    original_columns = [col for col in original_columns if col in df.columns]

    # 记录原始空值数量
    original_null_counts = df.isnull().sum().sum()
    print(f"\n开始数据修复，原始数据空值数量: {original_null_counts}")

    # 第一步：基础数据修复（在整个DataFrame上执行）
    print("\n开始基础数据修复...")
    df = process_fengdan_data(df)
    df = process_price_data(df)
    df = process_amount_data(df)

    # 按股票代码分组处理
    grouped = df.groupby('代码')
    repaired_groups = []
    total_repairs = 0

    for name, group in grouped:
        # 按日期排序
        group = group.sort_values('日期')
        group_repairs = 0

        # ========== 1. 基础价格链修复 ==========
        # 确保关键列存在
        for col in ['昨收', '现价', '收盘价', '今开']:
            if col not in group.columns:
                group[col] = np.nan

        # 新股首日处理
        new_stock_mask = (group['昨收'] == 0) & (group['今开'] > 0)
        group.loc[new_stock_mask, '昨收'] = group.loc[new_stock_mask, '今开']
        group_repairs += new_stock_mask.sum()

        # ========== 相互覆盖组修复 ==========
        # 涨幅%组修复
        if '涨幅%' in group.columns and '涨跌幅' in group.columns:
            # 涨幅%覆盖涨跌幅
            mask = (group['涨跌幅'] == 0) & (group['涨幅%'] != 0)
            group.loc[mask, '涨跌幅'] = group.loc[mask, '涨幅%']
            group_repairs += mask.sum()

            # 涨跌幅覆盖涨幅%
            mask = (group['涨幅%'] == 0) & (group['涨跌幅'] != 0)
            group.loc[mask, '涨幅%'] = group.loc[mask, '涨跌幅']
            group_repairs += mask.sum()

        # 主力净额组修复 (移除主力净比组相互覆盖)
        if '主力净额' in group.columns and '主力净流入' in group.columns:
            # 主力净流入覆盖主力净额
            mask = (group['主力净额'] == 0) & (group['主力净流入'] != 0)
            group.loc[mask, '主力净额'] = group.loc[mask, '主力净流入']
            group_repairs += mask.sum()

        if '主买净额' in group.columns and '主力净流入' in group.columns:
            # 主力净流入覆盖主买净额
            mask = (group['主买净额'] == 0) & (group['主力净流入'] != 0)
            group.loc[mask, '主买净额'] = group.loc[mask, '主力净流入']
            group_repairs += mask.sum()

        # 换手率组修复
        if '换手%' in group.columns and '换手Z' in group.columns:
            # 相互覆盖
            mask = (group['换手%'] == 0) & (group['换手Z'] != 0)
            group.loc[mask, '换手%'] = group.loc[mask, '换手Z']
            group_repairs += mask.sum()

            mask = (group['换手Z'] == 0) & (group['换手%'] != 0)
            group.loc[mask, '换手Z'] = group.loc[mask, '换手%']
            group_repairs += mask.sum()

        # 流通股组修复
        if '流通股(亿)' in group.columns and '流通股本Z' in group.columns:
            # 相互覆盖
            mask = (group['流通股(亿)'] == 0) & (group['流通股本Z'] != 0)
            group.loc[mask, '流通股(亿)'] = group.loc[mask, '流通股本Z']
            group_repairs += mask.sum()

            mask = (group['流通股本Z'] == 0) & (group['流通股(亿)'] != 0)
            group.loc[mask, '流通股本Z'] = group.loc[mask, '流通股(亿)']
            group_repairs += mask.sum()

        # 修复涨幅%
        if '涨幅%' in group.columns:
            # 方法1：使用涨跌和昨收计算
            mask = (group['涨幅%'] == 0) & (group['涨跌'] != 0) & (group['昨收'] > 0)
            group.loc[mask, '涨幅%'] = (group.loc[mask, '涨跌'] / group.loc[mask, '昨收']) * 100
            group_repairs += mask.sum()

            # 方法2：使用现价和昨收计算
            mask = (group['涨幅%'] == 0) & (group['现价'] > 0) & (group['昨收'] > 0)
            group.loc[mask, '涨幅%'] = ((group.loc[mask, '现价'] - group.loc[mask, '昨收']) / group.loc[
                mask, '昨收']) * 100
            group_repairs += mask.sum()

        # 修复现价和收盘价
        if '现价' in group.columns:
            mask = (group['现价'] == 0) & (group['收盘价'] > 0)
            group.loc[mask, '现价'] = group.loc[mask, '收盘价']
            group_repairs += mask.sum()

            mask = (group['现价'] == 0) & (group['今开'] > 0)
            group.loc[mask, '现价'] = group.loc[mask, '今开']
            group_repairs += mask.sum()

            mask = (group['现价'] == 0) & (group['昨收'] > 0) & (group['涨幅%'] != 0)
            group.loc[mask, '现价'] = group.loc[mask, '昨收'] * (1 + group.loc[mask, '涨幅%'] / 100)
            group_repairs += mask.sum()

        if '收盘价' in group.columns:
            mask = (group['收盘价'] == 0) & (group['现价'] > 0)
            group.loc[mask, '收盘价'] = group.loc[mask, '现价']
            group_repairs += mask.sum()

        # 修复今开
        if '今开' in group.columns:
            mask = (group['今开'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '今开'] = group.loc[mask, '昨收']
            group_repairs += mask.sum()

        # 修复涨跌
        if '涨跌' in group.columns:
            mask = (group['涨跌'] == 0) & (group['现价'] > 0) & (group['昨收'] > 0)
            group.loc[mask, '涨跌'] = group.loc[mask, '现价'] - group.loc[mask, '昨收']
            group_repairs += mask.sum()

        # ========== 2. 交易数据修复 ==========
        # 修复总金额、总量、均价
        if {'总金额', '总量', '均价'}.issubset(group.columns):
            # 修复均价
            mask = (group['均价'] == 0) & (group['总金额'] > 0) & (group['总量'] > 0)
            group.loc[mask, '均价'] = (group.loc[mask, '总金额'] * 1e4) / (group.loc[mask, '总量'] * 100)
            group_repairs += mask.sum()

            # 修复总金额
            mask = (group['总金额'] == 0) & (group['均价'] > 0) & (group['总量'] > 0)
            group.loc[mask, '总金额'] = group.loc[mask, '均价'] * group.loc[mask, '总量'] / 1e4
            group_repairs += mask.sum()

            # 修复总量
            mask = (group['总量'] == 0) & (group['均价'] > 0) & (group['总金额'] > 0)
            group.loc[mask, '总量'] = (group.loc[mask, '总金额'] * 1e4) / group.loc[mask, '均价']
            group_repairs += mask.sum()

        # 修复内外比
        if {'内盘', '外盘', '内外比'}.issubset(group.columns):
            mask = (group['内外比'] == 0) & (group['外盘'] > 0) & (group['内盘'] > 0)
            group.loc[mask, '内外比'] = group.loc[mask, '外盘'] / group.loc[mask, '内盘']
            group_repairs += mask.sum()

        # 修复'买量', '卖量', '委比%'
        if {'买量', '卖量', '委比%'}.issubset(group.columns):
            # 情况1：买量=卖量=委比%=0 → 先修复买量卖量，再计算委比%
            mask_all_zero = (group['买量'] == 0) & (group['卖量'] == 0) & (group['委比%'] == 0)
            if mask_all_zero.any():
                # 线性插值修复买量卖量
                group.loc[mask_all_zero, '买量'] = group.loc[mask_all_zero, '买量'].replace(0, np.nan).interpolate(
                    method='linear', limit_direction='both').fillna(0)
                group.loc[mask_all_zero, '卖量'] = group.loc[mask_all_zero, '卖量'].replace(0, np.nan).interpolate(
                    method='linear', limit_direction='both').fillna(0)
                group_repairs += mask_all_zero.sum()

                # 计算委比%
                mask_calc = mask_all_zero & (group['买量'] > 0) & (group['卖量'] > 0)
                group.loc[mask_calc, '委比%'] = (group.loc[mask_calc, '买量'] - group.loc[mask_calc, '卖量']) / \
                                                (group.loc[mask_calc, '买量'] + group.loc[mask_calc, '卖量']) * 100
                group_repairs += mask_calc.sum()

            # 情况2：买量=卖量=0但委比%不为0 → 只修复买量卖量
            mask_vol_zero_ratio_nonzero = (group['买量'] == 0) & (group['卖量'] == 0) & (group['委比%'] != 0)
            if mask_vol_zero_ratio_nonzero.any():
                group.loc[mask_vol_zero_ratio_nonzero, '买量'] = group.loc[mask_vol_zero_ratio_nonzero, '买量'].replace(
                    0, np.nan).interpolate(method='linear', limit_direction='both').fillna(0)
                group.loc[mask_vol_zero_ratio_nonzero, '卖量'] = group.loc[mask_vol_zero_ratio_nonzero, '卖量'].replace(
                    0, np.nan).interpolate(method='linear', limit_direction='both').fillna(0)
                group_repairs += mask_vol_zero_ratio_nonzero.sum()

            # 情况3：买量或卖量不为0 → 直接计算委比%
            mask_calc_ratio = (group['委比%'] == 0) & (group['买量'] > 0) & (group['卖量'] > 0)
            group.loc[mask_calc_ratio, '委比%'] = (group.loc[mask_calc_ratio, '买量'] - group.loc[
                mask_calc_ratio, '卖量']) / \
                                                  (group.loc[mask_calc_ratio, '买量'] + group.loc[
                                                      mask_calc_ratio, '卖量']) * 100
            group_repairs += mask_calc_ratio.sum()

        # ========== 3. 流通市值与流通股修复 ==========
        if {'流通股(亿)', '现价'}.issubset(group.columns) and '流通市值' in group.columns:
            # 修复流通市值
            mask = (group['流通市值'] == 0) & (group['流通股(亿)'] > 0) & (group['现价'] > 0)
            group.loc[mask, '流通市值'] = group.loc[mask, '流通股(亿)'] * group.loc[mask, '现价'] * 1e4
            group_repairs += mask.sum()

            # 修复流通股
            mask = (group['流通股(亿)'] == 0) & (group['流通市值'] > 0) & (group['现价'] > 0)
            group.loc[mask, '流通股(亿)'] = group.loc[mask, '流通市值'] / (group.loc[mask, '现价'] * 1e4)
            group_repairs += mask.sum()

        # 修复AB股总市值
        if {'总股本(亿)', '现价'}.issubset(group.columns) and 'AB股总市值' in group.columns:
            mask = (group['AB股总市值'] == 0) & (group['总股本(亿)'] > 0) & (group['现价'] > 0)
            group.loc[mask, 'AB股总市值'] = group.loc[mask, '总股本(亿)'] * group.loc[mask, '现价'] * 1e4
            group_repairs += mask.sum()

        # 修复市盈(TTM)
        if {'AB股总市值', '净利润(亿)'}.issubset(group.columns) and '市盈(TTM)' in group.columns:
            mask = (group['市盈(TTM)'] == 0) & (group['AB股总市值'] > 0) & (group['净利润(亿)'] > 0)
            group.loc[mask, '市盈(TTM)'] = group.loc[mask, 'AB股总市值'] / group.loc[mask, '净利润(亿)']
            group_repairs += mask.sum()

        # 修复市净率
        if {'AB股总市值', '净资产(亿)'}.issubset(group.columns) and '市净率' in group.columns:
            mask = (group['市净率'] == 0) & (group['AB股总市值'] > 0) & (group['净资产(亿)'] > 0)
            group.loc[mask, '市净率'] = group.loc[mask, 'AB股总市值'] / group.loc[mask, '净资产(亿)']
            group_repairs += mask.sum()

        # 修复资产负债率%
        if {'净资产(亿)', '少数股权(亿)', '总资产(亿)'}.issubset(group.columns) and '资产负债率%' in group.columns:
            mask = (group['资产负债率%'] == 0) & (group['总资产(亿)'] > 0)
            group.loc[mask, '资产负债率%'] = (1 - (group.loc[mask, '净资产(亿)'] + group.loc[mask, '少数股权(亿)']) /
                                              group.loc[mask, '总资产(亿)']) * 100
            group_repairs += mask.sum()

        # 修复毛利率%
        if {'营业收入(亿)', '营业成本(亿)'}.issubset(group.columns) and '毛利率%' in group.columns:
            mask = (group['毛利率%'] == 0) & (group['营业收入(亿)'] > 0)
            group.loc[mask, '毛利率%'] = (group.loc[mask, '营业收入(亿)'] - group.loc[mask, '营业成本(亿)']) / \
                                         group.loc[mask, '营业收入(亿)'] * 100
            group_repairs += mask.sum()

        # 修复营业利润率%
        if {'营业利润(亿)', '营业收入(亿)'}.issubset(group.columns) and '营业利润率%' in group.columns:
            mask = (group['营业利润率%'] == 0) & (group['营业收入(亿)'] > 0)
            group.loc[mask, '营业利润率%'] = group.loc[mask, '营业利润(亿)'] / group.loc[mask, '营业收入(亿)'] * 100
            group_repairs += mask.sum()

        # 修复净利润率% (改为前值覆盖)
        if '净利润率%' in group.columns:
            # 使用前值覆盖
            group['净利润率%'] = group['净利润率%'].replace(0, np.nan)
            group['净利润率%'] = group['净利润率%'].ffill().bfill().fillna(0)
            mask = group['净利润率%'] == 0
            group_repairs += mask.sum()

        # 修复净益率%
        if {'净利润(亿)', '净资产(亿)'}.issubset(group.columns) and '净益率%' in group.columns:
            mask = (group['净益率%'] == 0) & (group['净资产(亿)'] > 0)
            group.loc[mask, '净益率%'] = group.loc[mask, '净利润(亿)'] / group.loc[mask, '净资产(亿)'] * 100
            group_repairs += mask.sum()

        # 修复每股收益
        if {'净利润(亿)', '总股本(亿)'}.issubset(group.columns) and '每股收益' in group.columns:
            mask = (group['每股收益'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '每股收益'] = group.loc[mask, '净利润(亿)'] / group.loc[mask, '总股本(亿)']
            group_repairs += mask.sum()

        # 修复每股净资
        if {'净资产(亿)', '总股本(亿)'}.issubset(group.columns) and '每股净资' in group.columns:
            mask = (group['每股净资'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '每股净资'] = group.loc[mask, '净资产(亿)'] / group.loc[mask, '总股本(亿)']
            group_repairs += mask.sum()

        # 修复流通比例Z%
        if {'流通股(亿)', '总股本(亿)'}.issubset(group.columns) and '流通比例Z%' in group.columns:
            mask = (group['流通比例Z%'] == 0) & (group['总股本(亿)'] > 0)
            group.loc[mask, '流通比例Z%'] = group.loc[mask, '流通股(亿)'] / group.loc[mask, '总股本(亿)'] * 100
            group_repairs += mask.sum()

        # 修复年振幅%
        if {'52周最高', '52周最低'}.issubset(group.columns) and '年振幅%' in group.columns:
            mask = (group['年振幅%'] == 0) & (group['52周最低'] > 0)
            group.loc[mask, '年振幅%'] = (group.loc[mask, '52周最高'] - group.loc[mask, '52周最低']) / \
                                         group.loc[mask, '52周最低'] * 100
            group_repairs += mask.sum()

        # 修复现均差%
        if {'现价', '均价'}.issubset(group.columns) and '现均差%' in group.columns:
            mask = (group['现均差%'] == 0) & (group['均价'] > 0)
            group.loc[mask, '现均差%'] = (group.loc[mask, '现价'] - group.loc[mask, '均价']) / \
                                         group.loc[mask, '均价'] * 100
            group_repairs += mask.sum()

        # 修复开盘%
        if {'今开', '昨收'}.issubset(group.columns) and '开盘%' in group.columns:
            mask = (group['开盘%'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '开盘%'] = (group.loc[mask, '今开'] - group.loc[mask, '昨收']) / \
                                       group.loc[mask, '昨收'] * 100
            group_repairs += mask.sum()

        # 修复主力净比%
        #if {'主力净额', '总金额'}.issubset(group.columns) and '主力净比%' in group.columns:
        #    mask = (group['主力净比%'] == 0) & (group['总金额'] > 0)
        #    group.loc[mask, '主力净比%'] = group.loc[mask, '主力净额'] / group.loc[mask, '总金额'] * 100
        #    group_repairs += mask.sum()

        # 修复开盘昨比%
        if {'开盘金额', '昨成交额'}.issubset(group.columns) and '开盘昨比%' in group.columns:
            mask = (group['开盘昨比%'] == 0) & (group['昨成交额'] > 0)
            group.loc[mask, '开盘昨比%'] = group.loc[mask, '开盘金额'] / group.loc[mask, '昨成交额'] * 100
            group_repairs += mask.sum()

        # 修复封流比%
        if {'封单额', '流通市值'}.issubset(group.columns) and '封流比%' in group.columns:
            mask = (group['封流比%'] == 0) & (group['流通市值'] > 0)
            group.loc[mask, '封流比%'] = group.loc[mask, '封单额'] / group.loc[mask, '流通市值'] * 100
            group_repairs += mask.sum()

        # 封单额合理性检查
        if {'封单额', '流通市值'}.issubset(group.columns):
            mask = (group['封单额'] > group['流通市值'] * 0.3)
            group.loc[mask, '封单额'] = group.loc[mask, '流通市值'] * 0.3
            group_repairs += mask.sum()

        # ========== 新增修复规则 ==========
        # 开盘%修复
        if '开盘%' in group.columns and '昨收' in group.columns and '今开' in group.columns:
            mask = (group['开盘%'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '开盘%'] = ((group.loc[mask, '今开'] - group.loc[mask, '昨收']) / group.loc[
                mask, '昨收']) * 100
            group_repairs += mask.sum()

        # 开盘昨比%修复
        if '开盘昨比%' in group.columns and '开盘金额' in group.columns and '昨成交额' in group.columns:
            mask = (group['开盘昨比%'] == 0) & (group['昨成交额'] > 0)
            group.loc[mask, '开盘昨比%'] = (group.loc[mask, '开盘金额'] / group.loc[mask, '昨成交额']) * 100
            group_repairs += mask.sum()

        # 开盘换手Z修复
        if {'开盘换手Z', '开盘金额', '今开', '流通股(亿)'}.issubset(group.columns):
            mask = (group['开盘换手Z'] == 0) & (group['今开'] > 0) & (group['流通股(亿)'] > 0)
            group.loc[mask, '开盘换手Z'] = (group.loc[mask, '开盘金额'] / (
                        group.loc[mask, '今开'] * group.loc[mask, '流通股(亿)'] )) * 100
            group_repairs += mask.sum()

        # 连涨天修复
        if '连涨天' in group.columns and '涨跌幅' in group.columns:
            group = group.sort_values('日期')
            prev_change = group['涨跌幅'].shift(1)
            prev_days = group['连涨天'].shift(1)

            mask = (group['涨跌幅'] > 0) & (prev_change < 0)
            group.loc[mask, '连涨天'] = 1

            mask = (group['涨跌幅'] > 0) & (prev_change > 0)
            group.loc[mask, '连涨天'] = prev_days[mask] + 1

            mask = (group['涨跌幅'] < 0) & (prev_change < 0)
            group.loc[mask, '连涨天'] = prev_days[mask] - 1

            mask = (group['涨跌幅'] < 0) & (prev_change > 0)
            group.loc[mask, '连涨天'] = -1

            mask = (group['涨跌幅'] == 0)
            group.loc[mask, '连涨天'] = 0

            group_repairs += group['连涨天'].isna().sum()


        # ========== 修改：连板天修复 ==========
        if '连板天' in group.columns and '涨跌幅' in group.columns and '封单额' in group.columns:
            group = group.sort_values('日期')
            prev_board_days = group['连板天'].shift(1)

            # 定义涨停区间
            limit_up_ranges = [
                (4.7, 5.3),  # 5%涨停
                (9.7, 10.3),  # 10%涨停
                (19.7, 20.3),  # 20%涨停
                (29.7, 30.3)  # 30%涨停
            ]

            # 初始化连板天为0
            group['连板天'] = 0

            # 检查是否涨停
            for low, high in limit_up_ranges:
                mask = (group['涨跌幅'] >= low) & (group['涨跌幅'] <= high) & (group['封单额'] != 0)
                if mask.any():
                    # 处理首日情况
                    mask_first_day = mask & prev_board_days.isna()
                    group.loc[mask_first_day, '连板天'] = 1

                    # 处理非首日情况
                    mask_non_first = mask & ~prev_board_days.isna()
                    group.loc[mask_non_first, '连板天'] = prev_board_days[mask_non_first] + 1

                    # 修复封成比
                    if '封成比' in group.columns:
                        mask2 = mask & (group['封成比'] == 0)
                        if mask2.any():
                            avg_ratio = group.loc[group['封成比'] > 0, '封成比'].mean()
                            if not np.isnan(avg_ratio):
                                group.loc[mask2, '封成比'] = avg_ratio
                                group_repairs += mask2.sum()

            # 非涨停情况
            mask = (group['涨跌幅'] < 0) & (group['封单额'] != 0)
            group.loc[mask, '连板天'] = -1

            mask = (group['封单额'] == 0)
            group.loc[mask, '连板天'] = 0

            group_repairs += group['连板天'].isna().sum()

        # 实体涨幅%修复
        if '实体涨幅%' in group.columns and '收盘价' in group.columns:
            # 使用前5日收盘价的均值
            group['prev_5_close'] = group['收盘价'].shift(1).rolling(window=5, min_periods=1).mean()
            mask = (group['实体涨幅%'] == 0) & (group['prev_5_close'] > 0)
            group.loc[mask, '实体涨幅%'] = (group.loc[mask, '收盘价'] - group.loc[mask, 'prev_5_close']) / group.loc[
                mask, 'prev_5_close'] * 100
            group_repairs += mask.sum()
            group.drop('prev_5_close', axis=1, inplace=True, errors='ignore')

        # ========== 新增：最高%、最低%、均涨幅%修复 ==========
        if '最高%' in group.columns and '最高' in group.columns and '昨收' in group.columns:
            mask = (group['最高%'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '最高%'] = ((group.loc[mask, '最高'] - group.loc[mask, '昨收']) /
                                        group.loc[mask, '昨收']) * 100
            group_repairs += mask.sum()

        if '最低%' in group.columns and '最低' in group.columns and '昨收' in group.columns:
            mask = (group['最低%'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '最低%'] = ((group.loc[mask, '最低'] - group.loc[mask, '昨收']) /
                                        group.loc[mask, '昨收']) * 100
            group_repairs += mask.sum()

        if '均涨幅%' in group.columns and '均价' in group.columns and '昨收' in group.columns:
            mask = (group['均涨幅%'] == 0) & (group['昨收'] > 0)
            group.loc[mask, '均涨幅%'] = ((group.loc[mask, '均价'] - group.loc[mask, '昨收']) /
                                          group.loc[mask, '昨收']) * 100
            group_repairs += mask.sum()

        # ========== 新增：形态指标修复 ==========
        for col in ['近日指标提示', '短期形态', '中期形态', '长期形态']:
            if col in group.columns:
                mask = group[col] == 0
                group.loc[mask, col] = 14
                group_repairs += mask.sum()



        # ========== 4. 基本不变列值的前值覆盖 ==========
        # 股本结构
        invariant_cols = [
            '总股本(亿)', 'B/A股(亿)', 'H股(亿)', '流通股(亿)', '流通股本Z', '流通比例Z%'
        ]

        # 财务数据（仅财报季更新）
        financial_cols = [
            '总资产(亿)', '净资产(亿)', '少数股权(亿)', '流动资产(亿)', '固定资产(亿)', '无形资产(亿)',
            '流动负债(亿)', '货币资金(亿)', '存货(亿)', '应收账款(亿)', '合同负债(亿)', '资本公积金(亿)',
            '营业收入(亿)', '营业成本(亿)', '营业利润(亿)', '投资收益(亿)', '净利润(亿)', '扣非净利润(亿)',
            '未分利润(亿)', '经营现金流(亿)', '总现金流(亿)'
        ]

        # 股东与员工数据
        shareholder_cols = [
            '股东人数', '员工人数'
        ]

        # 衍生指标
        derived_cols = [
            '每股净资', '每股公积', '每股未分配', '每股现金流', '利润同比%', '股息率%',
            '市净率', '市现率', '市销率', '行业PE', '贝塔系数', '发行价', '安全分',
            '其它权益工具', 'ABH总市值', '52周最高', '52周最低'
        ]

        # 合并所有基本不变列
        invariant_all = invariant_cols + financial_cols + shareholder_cols + derived_cols

        for col in invariant_all:
            if col in group.columns:
                # 使用前值填充0值
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].ffill().bfill().fillna(0)
                # 统计修复数量
                mask = group[col].isna()
                group_repairs += mask.sum()

        # ========== 5. 插值填充方法 ==========
        # 第4组：市场交易特征指标 → 前后均值修复
        trade_features = [
            '均价', '内盘', '外盘', '内外比',
            '委比%', '量涨速%', '活跃度', '总撤委比%', '散户单增比',
            '开盘金额', '2分钟金额'  # 新增
        ]
        for col in trade_features:
            if col in group.columns:
                # 前值权重60% + 后值权重40%
                prev_val = group[col].shift(1).fillna(0)
                next_val = group[col].shift(-1).fillna(0)
                mask = (group[col] == 0)
                group.loc[mask, col] = prev_val[mask] * 0.6 + next_val[mask] * 0.4
                group_repairs += mask.sum()

        # 第5组：资金流向指标 → 线性插值修复
        capital_flow = [
            '主力净比%', '主力占比%', '净买率%', '总卖占比%',
            '回头波%', '攻击波%', '市盈(动)'  # 新增或修改
        ]
        for col in capital_flow:
            if col in group.columns:
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].interpolate(method='linear', limit_direction='both')
                group[col] = group[col].fillna(0)
                group_repairs += (group[col] == 0).sum()

        # 第6组：市场深度指标 → 前值填充
        market_depth = [
            '总委比%', '总委量差', '总委托笔数', '总成交笔数'
        ]
        for col in market_depth:
            if col in group.columns:
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].ffill().bfill().fillna(0)
                group_repairs += (group[col] == 0).sum()

        # 第7组：稳定指标 → 中位数/前值填充
        stable_features = [
            '贝塔系数', '逐笔均量', '笔均量', '笔换手%'
        ]
        for col in stable_features:
            if col in group.columns:
                # 计算5日中位数
                median_val = group[col].rolling(window=5, min_periods=1).median()
                mask = (group[col] == 0)
                group.loc[mask, col] = median_val[mask]
                # 剩余0值用前值填充
                mask = (group[col] == 0)
                group[col] = group[col].replace(0, np.nan)
                group[col] = group[col].ffill().bfill().fillna(0)
                group_repairs += mask.sum()

        # 第8组：特殊列处理
        # 总委比%：与委比%协同修复
        if '总委比%' in group.columns and '委比%' in group.columns:
            mask = (group['总委比%'] == 0) & (group['委比%'] != 0)
            group.loc[mask, '总委比%'] = group.loc[mask, '委比%']
            group_repairs += mask.sum()

        # 逐笔均量：与笔均量协同修复
        if '逐笔均量' in group.columns and '笔均量' in group.columns:
            mask = (group['逐笔均量'] == 0) & (group['笔均量'] != 0)
            group.loc[mask, '逐笔均量'] = group.loc[mask, '笔均量']
            group_repairs += mask.sum()

            mask = (group['笔均量'] == 0) & (group['逐笔均量'] != 0)
            group.loc[mask, '笔均量'] = group.loc[mask, '逐笔均量']
            group_repairs += mask.sum()

        # ========== 新增：利润同比%为0时的修复 ==========
        if '利润同比%' in group.columns:
            mask_profit_zero = (group['利润同比%'] == 0)

            # 需要修复的列列表
            repair_cols = [
                '开盘抢筹%', '开盘%', '总委比%', '总委量差', '总卖占比%',
                '总撤委比%', '总委托笔数', '总成交笔数', '逐笔均量', '散户单增比'
            ]

            # 线性插值修复
            linear_interp_cols = ['开盘抢筹%']
            for col in linear_interp_cols:
                if col in group.columns:
                    group.loc[mask_profit_zero & (group[col] == 0), col] = np.nan
                    group[col] = group[col].interpolate(method='linear', limit_direction='both').fillna(0)
                    group_repairs += mask_profit_zero.sum()

            # 前后均值修复
            mean_repair_cols = [
                '开盘%', '总委比%', '总委量差', '总卖占比%',
                '总撤委比%', '总委托笔数', '总成交笔数', '逐笔均量', '散户单增比'
            ]
            for col in mean_repair_cols:
                if col in group.columns:
                    prev_val = group[col].shift(1).fillna(0)
                    next_val = group[col].shift(-1).fillna(0)
                    mask = mask_profit_zero & (group[col] == 0)
                    group.loc[mask, col] = prev_val[mask] * 0.6 + next_val[mask] * 0.4
                    group_repairs += mask.sum()

        # ========== 新增：当日资金流向修复 ==========
        flow_cols = [
            '当日___超大单', '当日___大单', '当日___中单', '当日___小单'
        ]

        if all(col in group.columns for col in flow_cols):
            # 检查是否全部为0
            mask_all_zero = (group[flow_cols] == 0).all(axis=1)

            for col in flow_cols + ['散户单增比']:
                if col in group.columns:
                    # 前后均值修复
                    prev_val = group[col].shift(1).fillna(0)
                    next_val = group[col].shift(-1).fillna(0)
                    mask = mask_all_zero & (group[col] == 0)
                    group.loc[mask, col] = prev_val[mask] * 0.6 + next_val[mask] * 0.4
                    group_repairs += mask.sum()

        total_repairs += group_repairs
        repaired_groups.append(group)

    # 合并所有分组
    repaired_df = pd.concat(repaired_groups, axis=0)

    # 重置索引
    repaired_df.reset_index(drop=True, inplace=True)

    # 恢复原始列顺序
    repaired_df = repaired_df[original_columns]

    # 统计修复结果
    repaired_null_counts = repaired_df.isnull().sum().sum()
    print(f"\n数据修复完成！共修复 {total_repairs} 处数据")
    print(f"修复后数据空值数量: {repaired_null_counts}")
    print(f"修复率: {(original_null_counts - repaired_null_counts) / original_null_counts * 100:.2f}%")

    return repaired_df


if __name__ == "__main__":
    # 实际数据处理
    file_path = '通达信数据.csv'
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"\n成功读取数据，共 {len(df)} 条记录")

        # 全量修复
        df_repaired = repair_stock_data(df, Fix_all=True)

        # 保存结果
        output_path = 'repaired_' + file_path
        df_repaired.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据修复完成并已保存到: {output_path}")
    except Exception as e:
        print(f"\n处理数据时出错: {str(e)}")