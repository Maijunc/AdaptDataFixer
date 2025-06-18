import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ​**📌 近日指标提示映射表**​
JINRI_ZHIBIAO_MAPPING = {
    "空头排列": 1, "KDJ死叉": 2, "MACD死叉": 3, "EXPMA死叉": 4, "向下突破平台": 5,
    "跌破BOLL下轨": 6, "顶部放量": 7, "高位回落": 8, "放量下挫": 9, "温和放量下跌": 10,
    "阶段缩量": 11, "均线粘合": 12, "顶部缩量": 13, "-- --": 14, "阶段放量": 15,
    "上穿BOLL上轨": 16, "KDJ金叉": 17, "MACD金叉": 18, "EXPMA金叉": 19, "多头排列": 20,
    "放量上攻": 21, "价量齐升": 22, "向上突破平台": 23, "底部反转": 24
}


def clean_data(df):
    """ 清理股票数据 """
    print("\n📌 正在清理数据...")

    df.loc[:, '代码'] = df['代码'].astype(str).str.replace(r'^="(\d+)"$', r'\1', regex=True)

    if '总金额' in df.columns:
        df.loc[:, '总金额'] = pd.to_numeric(df['总金额'], errors='coerce')
        df = df[df['总金额'].notna() & (df['总金额'] > 0)].copy()

    numeric_cols = ['涨幅%', '现价', '今开', '涨跌', '最高', '最低', '振幅%']
    for col in numeric_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    drop_cols = ['溢价%', '细分行业', '地区', '未匹配量', '开盘昨封比%', '竞价涨停买',
                 '财务更新', '上市日期', '报告期', '交易代码', '自选日', '自选价', '自选收益%',
                 '应计利息', '一二级行业','涨速%','短换手%','开盘竞价',
                 '主营构成', '标记信息', '早盘竞价', '分时简图', '可转债代码', '日线简图', '两日分时图',
                 '1分钟___涨幅%', '1分钟___换手%%','买价', '卖价','税后利润(亿)',
                 '1分钟___主力净额', '1分钟___主力占比%', '1分钟___大单', '1分钟___中单', '1分钟___小单',
                 'B/A股(亿)', 'H股(亿)'
                 ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    special_cols_processing(df)

    amount_cols = ['总金额', '2分钟金额','封单额', '昨封单额', '前封单额', '昨成交额', '开盘金额',
                   '主力净额', '主买净额', '昨开盘金额', '3日成交额',
                   '当日___超大单', '当日___大单', '当日___中单', '当日___小单', '1分钟___超大单'
                   ]
    for col in amount_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 10000

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def special_cols_processing(df):
    """ 处理特殊的列，如流通股本Z、流通市值等 """
    if '流通股本Z' in df.columns:
        df['流通股本Z'] = df['流通股本Z'].astype(str).str.replace('亿', '').astype(float)
    if '流通市值' in df.columns:
        df['流通市值'] = df['流通市值'].astype(str).str.replace('亿', '').astype(float)
    if '流通市值Z' in df.columns:
        df['流通市值Z'] = df['流通市值Z'].astype(str).str.replace('亿', '').astype(float)
    if '市值增减' in df.columns:
        df['市值增减'] = df['市值增减'].astype(str).str.replace('亿', '').astype(float)
    if 'ABH总市值' in df.columns:
        df['ABH总市值'] = df['ABH总市值'].astype(str).str.replace('亿', '').astype(float)
    if 'AB股总市值' in df.columns:
        df['AB股总市值'] = df['AB股总市值'].astype(str).str.replace('亿', '').astype(float)
    if '人均市值' in df.columns:
        df['人均市值'] = df['人均市值'].astype(str).str.replace('万', '').str.strip()
        df['人均市值'] = pd.to_numeric(df['人均市值'], errors='coerce').fillna(0)
    if '每股收益' in df.columns:
        df['每股收益'] = df['每股收益'].astype(str).str.replace('㈢', '').str.strip()
        df['每股收益'] = pd.to_numeric(df['每股收益'], errors='coerce').fillna(0)
    if '净益率%' in df.columns:
        df['净益率%'] = df['净益率%'].astype(str).str.replace('㈢', '').str.strip()
        df['净益率%'] = pd.to_numeric(df['净益率%'], errors='coerce').fillna(0)
    if '几天几板' in df.columns:
        df['几天几板'] = df['几天几板'].apply(
            lambda x: 1 if x == '首板' else (
                pd.to_numeric(re.search(r'(\d+)', str(x)).group(1), errors='coerce')
                if re.search(r'(\d+)', str(x)) else 0
            )
        )
    if '总委量差' in df.columns:
        df['总委量差'] = df['总委量差'].apply(
            lambda x: float(str(x).replace('万', '')) * 10000 if '万' in str(x) else pd.to_numeric(x, errors='coerce'))
        df['总委量差'] = df['总委量差'].fillna(0)
    if '近日指标提示' in df.columns:
        df['近日指标提示'] = df['近日指标提示'].map(JINRI_ZHIBIAO_MAPPING).fillna(0)


def verify_and_fill_data(df):
    """ 在存盘前对数据进行核查，填充空值、NaN 和字符类型值为 0 """
    print("\n📌 正在进行数据核查和填充...")
    protected_cols = ['代码', '名称', '日期']
    for col in df.columns:
        if col not in protected_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def calculate_rank(df):
    """计算每日涨幅排名"""
    print("\n📌 正在计算每日涨幅排名...")

    # 删除已有的排名列
    if '排名' in df.columns:
        df = df.drop(columns=['排名'])

    # 确保必要的列存在
    required_columns = ['代码', '日期', '涨幅%', '量比']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据中缺少 '{col}' 列，无法计算排名。")

    # 创建新的DataFrame避免碎片化
    new_df = df.copy()

    # 计算排名
    ranked = new_df.sort_values(by=['涨幅%', '量比'], ascending=[False, False]) \
                 .groupby('日期') \
                 .cumcount() + 1

    # 添加排名列
    new_df['排名'] = ranked.astype(int)

    return new_df


def read_stock_data(file_path):
    """ 读取股票数据文件，并插入日期列 """
    match = re.search(r'(\d{8})', file_path)
    if not match:
        raise ValueError(f"❌ 文件名格式错误: {file_path}，请确保文件名为 '全部Ａ股YYYYMMDD.xls'")
    date_str = match.group(1)
    df = pd.read_csv(file_path, encoding='gbk', sep='\t', on_bad_lines='skip', low_memory=False)
    df.insert(2, '日期', pd.to_datetime(date_str, format='%Y%m%d'))
    return df


def get_stocks_to_remove(current_dir):
    """获取需要删除的股票列表（包含*或退字样的股票）"""
    a_stock_files = sorted([f for f in os.listdir(current_dir) if f.startswith('全部Ａ股') and f.endswith('.xls')])
    if not a_stock_files:
        return set()

    # 获取最新日期的文件
    latest_file = max(a_stock_files, key=lambda x: x.split('全部Ａ股')[1].split('.xls')[0])
    latest_df = pd.read_csv(os.path.join(current_dir, latest_file), encoding='gbk', sep='\t', on_bad_lines='skip',
                            low_memory=False)

    # 找出名称中包含*或退的股票代码
    mask = latest_df['名称'].str.contains(r'[*★]|退', na=False, regex=True)
    stocks_to_remove = set(latest_df.loc[mask, '代码'].astype(str).str.replace(r'^="(\d+)"$', r'\1', regex=True))

    return stocks_to_remove


def main():
    current_dir = os.getcwd()
    tdx_files = [f for f in os.listdir(current_dir) if f.startswith('通达信数据_') and f.endswith('.csv')]

    # 获取需要删除的股票列表
    stocks_to_remove = get_stocks_to_remove(current_dir)
    print(f"📌 需要删除的股票数量: {len(stocks_to_remove)}")

    if not tdx_files:
        # 处理没有历史数据的情况
        a_stock_files = sorted([f for f in os.listdir(current_dir) if f.startswith('全部Ａ股') and f.endswith('.xls')])
        dfs = [read_stock_data(os.path.join(current_dir, file)) for file in a_stock_files]

        if not dfs:
            print("❌ 没有找到可用的数据文件，程序退出。")
            return

        latest_df = pd.concat(dfs, ignore_index=True)
        cleaned_new_df = clean_data(latest_df)

        # 直接使用清理后的数据
        combined_df = cleaned_new_df.copy()

    else:
        # 选择最新的 CSV 文件
        latest_file = sorted(tdx_files, key=lambda x: x.split('_')[1].split('.csv')[0])[-1]
        latest_df = pd.read_csv(latest_file, low_memory=False)
        latest_df['日期'] = pd.to_datetime(latest_df['日期'])

        # 找到新的数据文件
        a_stock_files = sorted([f for f in os.listdir(current_dir) if f.startswith('全部Ａ股') and f.endswith('.xls')])
        new_files = [f for f in a_stock_files if
                     f.split('全部Ａ股')[1].split('.xls')[0] > latest_file.split('_')[1].split('.csv')[0]]

        if not new_files:
            print("✅ 没有新的数据需要更新。")
            return

        new_dfs = [read_stock_data(os.path.join(current_dir, file)) for file in new_files]
        new_df = pd.concat(new_dfs, ignore_index=True)
        cleaned_new_df = clean_data(new_df)

        # 统一代码数据类型
        if latest_df['代码'].dtype == 'O':
            cleaned_new_df['代码'] = cleaned_new_df['代码'].astype(str)
        else:
            cleaned_new_df['代码'] = cleaned_new_df['代码'].astype(int)

        # 合并数据
        combined_df = pd.concat([latest_df, cleaned_new_df], ignore_index=True)

    # 统一代码数据类型
    if combined_df['代码'].dtype == 'O':
        combined_df['代码'] = combined_df['代码'].astype(str)
    else:
        combined_df['代码'] = combined_df['代码'].astype(int)

    # 删除需要移除的股票
    if stocks_to_remove:
        combined_df = combined_df[~combined_df['代码'].isin(stocks_to_remove)]
        print(f"✅ 已删除 {len(stocks_to_remove)} 只包含*或退字样的股票")

    # 重新排序
    combined_df = combined_df.sort_values(by=['代码', '日期']).reset_index(drop=True)

    # 删除重复数据
    combined_df = combined_df.drop_duplicates(subset=['代码', '日期'])

    # 数据验证和填充
    verified_df = verify_and_fill_data(combined_df)

    # 计算每日涨幅排名（使用优化后的函数）
    ranked_df = calculate_rank(verified_df)

    # 最终排序
    final_df = ranked_df.sort_values(by=['代码', '日期']).reset_index(drop=True)

    # 保存 CSV
    max_date = final_df['日期'].max().strftime('%Y%m%d')
    output_file = f'通达信数据_{max_date}.csv'
    final_df.to_csv(output_file, index=False, encoding='utf_8_sig')

    print(f"✅ 数据已保存到 {output_file}")


if __name__ == "__main__":
    main()