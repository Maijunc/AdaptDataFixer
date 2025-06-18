import os
import re
import pandas as pd
from datetime import datetime


def process_file(file_path):
    """处理单个文件：确保日期格式为纯日期（不带时间）"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None, None

    try:
        # 读取文件
        df = pd.read_csv(file_path, encoding='utf_8_sig')

        if df.empty:
            print(f"文件为空: {file_path}")
            return None, None

        # 处理日期列 - 确保最终格式为 YYYY/M/D（不带时间）
        if '日期' in df.columns:
            # 先转换为datetime对象
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            # 然后转换为日期部分（不带时间）
            df['日期'] = df['日期'].dt.date  # 这会返回datetime.date对象
            # 最后格式化为字符串 (使用平台兼容的格式)
            try:
                # 尝试Windows格式
                df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y/%#m/%#d') if pd.notnull(x) else None)
            except:
                # 如果失败，使用标准格式
                df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y/%m/%d') if pd.notnull(x) else None)
            # 删除无效日期
            df = df.dropna(subset=['日期'])

        # 去除股票代码前导零
        if '代码' in df.columns:
            df['代码'] = df['代码'].astype(str).str.replace(r'^0+', '')

        # 保存文件（覆盖原文件）
        df.to_csv(file_path, index=False, encoding='utf_8_sig')

        # 返回处理后的DataFrame和最大日期
        max_date = df['日期'].max() if '日期' in df.columns and not df.empty else None
        return df, max_date

    except Exception as e:
        print(f"处理文件 {file_path} 出错: {str(e)}")
        return None, None


def find_and_process_latest_file(pattern):
    """严格匹配模式查找最新文件"""
    max_file_date = None
    latest_file = None
    date_pattern = re.compile(pattern, re.IGNORECASE)

    for file in os.listdir('.'):
        if os.path.isfile(file):
            match = date_pattern.fullmatch(file)
            if match:
                file_date = match.group(1)
                if max_file_date is None or file_date > max_file_date:
                    max_file_date = file_date
                    latest_file = file

    if latest_file:
        print(f"找到匹配文件: {latest_file}")
        df, _ = process_file(latest_file)  # We ignore the content date now
        return df, max_file_date  # Return the filename date instead
    else:
        print(f"未找到匹配模式 {pattern} 的文件")
        return None, None


def merge_datasets(existing_df, incremental_df):
    """合并现有数据集和增量数据集"""
    # 处理空数据的情况
    if existing_df is None and incremental_df is None:
        print("警告: 现有数据集和增量数据集都为空")
        return None

    if existing_df is None or len(existing_df) == 0:
        print("使用增量数据集作为合并结果")
        return incremental_df.copy() if incremental_df is not None else None

    if incremental_df is None or len(incremental_df) == 0:
        print("使用现有数据集作为合并结果")
        return existing_df.copy()

    # 合并数据集，增量数据覆盖现有数据
    combined_df = pd.concat([existing_df, incremental_df], ignore_index=True)

    # 去重，保留最后出现的记录（即增量数据优先）
    combined_df = combined_df.drop_duplicates(['代码', '日期'], keep='last')

    # 按代码(数字)和日期排序
    # 创建临时数字代码列用于排序
    combined_df['代码_num'] = combined_df['代码'].astype(int)
    combined_df = combined_df.sort_values(['代码_num', '日期'])
    combined_df.drop('代码_num', axis=1, inplace=True)  # 删除临时列

    return combined_df


def merge_with_tdx_data(moneyflow_df, tdx_df):
    """将资金流向数据合并到通达信数据集中，确保日期格式一致"""
    if moneyflow_df is None or len(moneyflow_df) == 0:
        print("资金流向数据为空，跳过合并")
        return tdx_df

    if tdx_df is None or len(tdx_df) == 0:
        print("通达信数据为空，跳过合并")
        return moneyflow_df

    # 统一日期格式处理
    def clean_date_column(df):
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df['日期'] = df['日期'].dt.date
            try:
                df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y/%#m/%#d') if pd.notnull(x) else None)
            except:
                df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y/%m/%d') if pd.notnull(x) else None)
            df = df.dropna(subset=['日期'])
        return df

    try:
        moneyflow_df = clean_date_column(moneyflow_df)
        tdx_df = clean_date_column(tdx_df)
    except Exception as e:
        print(f"日期格式处理出错: {str(e)}")
        return tdx_df

    # 定义需要合并的所有列（包括净流入和占比列）
    target_columns = [
        '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入',
        '主力净流入占比', '小单占比', '中单占比', '大单占比', '超大单占比',
        '收盘价', '涨跌幅'
    ]

    # 创建合并键
    moneyflow_df['merge_key'] = moneyflow_df['代码'].astype(str) + '_' + moneyflow_df['日期'].astype(str)
    tdx_df['merge_key'] = tdx_df['代码'].astype(str) + '_' + tdx_df['日期'].astype(str)

    # 对资金流向数据中的净流入列进行单位转换（元→亿元）
    for col in ['主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入']:
        if col in moneyflow_df.columns:
            moneyflow_df[col] = pd.to_numeric(moneyflow_df[col], errors='coerce') / 100000000
            print(f"资金流向数据中的 {col} 已转换为亿元单位")

    # 合并所有目标列
    for col in target_columns:
        if col not in tdx_df.columns:
            tdx_df[col] = None  # 如果列不存在则创建

        if col in moneyflow_df.columns:
            # 创建映射字典
            value_map = moneyflow_df.set_index('merge_key')[col].to_dict()

            # 只更新空值或0值
            mask = (tdx_df[col].isna() | (tdx_df[col] == 0)) & tdx_df['merge_key'].isin(value_map)
            tdx_df.loc[mask, col] = tdx_df.loc[mask, 'merge_key'].map(value_map)

            print(f"已合并列: {col}")
        else:
            print(f"注意: 列 '{col}' 在资金流向数据中不存在")

    # 删除临时列
    tdx_df.drop('merge_key', axis=1, inplace=True)

    return tdx_df


def main():
    try:
        # 第一步：严格匹配主数据集
        print("第一步：处理主数据集...")
        main_df, main_file_date = find_and_process_latest_file(r'^资金流向数据_(\d{8})\.csv$')
        print(f"主数据集文件日期: {main_file_date}")

        # 第二步：严格匹配增量数据集
        print("\n处理增量数据集...")
        incremental_df, incremental_file_date = find_and_process_latest_file(r'^增量[_\-]资金流向数据_(\d{8})\.csv$')
        print(f"增量数据集文件日期: {incremental_file_date}")

        # 确定输出日期 - 使用最新的文件名日期
        output_date = incremental_file_date if incremental_file_date else main_file_date
        if not output_date:
            output_date = datetime.now().strftime('%Y%m%d')
        print(f"使用输出日期: {output_date}")

        # 第三步：改进合并逻辑
        print("\n第三步：合并数据集...")
        if main_df is None and incremental_df is None:
            print("错误: 没有可用的数据集")
            return -1
        elif incremental_df is None:
            print("无增量数据，直接使用主数据集")
            moneyflow_df = main_df
        elif main_df is None:
            print("无主数据，直接使用增量数据")
            moneyflow_df = incremental_df
        else:
            if main_df.equals(incremental_df):
                print("警告: 主数据和增量数据相同，使用主数据")
                moneyflow_df = main_df
            else:
                print("执行主数据和增量数据合并")
                moneyflow_df = merge_datasets(main_df, incremental_df)

        # 保存结果 - 使用文件名中的日期
        output_file = f"资金流向数据_{output_date}.csv"
        moneyflow_df.to_csv(output_file, index=False, encoding='utf_8_sig')
        print(f"合并后的数据集已保存为: {output_file}")

        # 第四步：合并到通达信数据
        print("\n第四步：合并到通达信数据集...")
        tdx_file = f"通达信数据_{output_date}.csv"
        if os.path.exists(tdx_file):
            print(f"找到通达信文件: {tdx_file}")
            tdx_df = pd.read_csv(tdx_file, encoding='utf_8_sig')
            merged_tdx_df = merge_with_tdx_data(moneyflow_df, tdx_df)
            if merged_tdx_df is not None:
                # 保存合并后的文件（不再需要单位转换，因为已经在merge_with_tdx_data中处理）
                merged_tdx_df.to_csv(tdx_file, index=False, encoding='utf_8_sig')
                print("通达信数据更新成功（仅增量数据已转换为亿元单位）")
        else:
            print(f"未找到通达信文件: {tdx_file}")

        print("\n处理完成")
        return 0

    except Exception as e:
        print(f"发生错误: {str(e)}")
        return -1


if __name__ == "__main__":
    main()