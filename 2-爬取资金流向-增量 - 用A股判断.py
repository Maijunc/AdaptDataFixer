import os
import re
import time
import random
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timedelta

# 全局配置参数
CONFIG = {
    'RECENT_DAYS': 30,  # 设置要获取的最近天数，可在此统一修改
    'MAX_RETRY': 3,  # 最大重试次数
    'SLEEP_RANGE': (1, 5)  # 随机停留时间范围(秒)
}


def get_latest_a_stock_file(current_dir):
    """获取最新的全部Ａ股文件"""
    a_stock_files = [f for f in os.listdir(current_dir) if f.startswith('全部Ａ股') and f.endswith('.xls')]
    if not a_stock_files:
        raise FileNotFoundError("没有找到以'全部Ａ股'开头的文件")

    def extract_date(filename):
        match = re.search(r'全部Ａ股(\d{4})(\d{2})(\d{2})\.xls', filename)
        if match:
            return match.group(1) + match.group(2) + match.group(3)
        return "0"

    latest_file = max(a_stock_files, key=lambda x: extract_date(x))
    return latest_file, extract_date(latest_file)


def read_stock_codes(file_path):
    """读取股票代码"""
    df = pd.read_csv(file_path, encoding='gbk', sep='\t', on_bad_lines='skip', low_memory=False)
    df['代码'] = df['代码'].astype(str).str.replace(r'^="(\d+)"$', r'\1', regex=True)
    return df['代码'].unique().tolist()


def get_recent_days_data(stock_code, days=CONFIG['RECENT_DAYS'], max_retry=CONFIG['MAX_RETRY']):
    """获取单只股票最近days天的资金流数据"""
    market = '1' if stock_code.startswith(('6', '9', '688')) else '0'
    secid = f"{market}.{stock_code}"

    params = {
        'lmt': str(days),  # 使用配置中的天数
        'klt': '101',
        'secid': secid,
        'fields1': 'f1,f2,f3,f7',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63',
        'ut': 'b2884a393a59ad64002292a3e90d46a5',
        'cb': 'jQuery183003743205523325055_1625891323112',
        '_': str(int(time.time() * 1000))
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://data.eastmoney.com/zjlx/'
    }

    for attempt in range(max_retry):
        try:
            resp = requests.get(
                url="http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
                params=params,
                headers=headers,
                timeout=10
            )

            if not resp.text or '(' not in resp.text or ')' not in resp.text:
                print(f"响应格式异常: {stock_code}")
                return None

            try:
                data = resp.text.split('(')[1].split(')')[0]
                if not data or data == 'null':
                    print(f"空数据: {stock_code}")
                    return None

                json_data = eval(data)
            except Exception as e:
                print(f"解析JSON失败: {stock_code} - {str(e)}")
                return None

            if not isinstance(json_data, dict) or 'data' not in json_data:
                print(f"无有效数据: {stock_code}")
                return None

            if not json_data['data'] or 'klines' not in json_data['data']:
                print(f"无k线数据: {stock_code}")
                return None

            raw_data = json_data['data']['klines']
            if not raw_data:
                print(f"空k线数据: {stock_code}")
                return None

            try:
                df = pd.DataFrame([x.split(',') for x in raw_data])

                if len(df.columns) < 13:  # 检查列数是否足够
                    print(f"数据列不足: {stock_code}")
                    return None

                df.columns = [
                    '日期', '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入',
                    '主力净流入占比', '小单占比', '中单占比', '大单占比', '超大单占比',
                    '收盘价', '涨跌幅'
                ]

                float_cols = ['主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入',
                              '主力净流入占比', '小单占比', '中单占比', '大单占比', '超大单占比',
                              '收盘价', '涨跌幅']
                df[float_cols] = df[float_cols].astype(float)
                df['代码'] = stock_code.lstrip('0')
                df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y/%m/%d')

                return df

            except Exception as e:
                print(f"数据处理失败: {stock_code} - {str(e)}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"请求失败({attempt + 1}/{max_retry}): {stock_code} - {str(e)}")
            time.sleep(2)
        except Exception as e:
            print(f"未知错误({attempt + 1}/{max_retry}): {stock_code} - {str(e)}")
            time.sleep(2)

    print(f"达到最大重试次数: {stock_code}")
    return None


def save_final_data(data, output_file):
    """保存最终数据"""
    try:
        data.to_csv(output_file, index=False, encoding='utf_8_sig')
        print(f"\n最终数据已保存到: {output_file} (记录数: {len(data)})")
    except Exception as e:
        print(f"保存最终数据失败: {str(e)}")


def load_downloaded_codes(output_file):
    """加载已下载的股票代码（修复空文件判断）"""
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            df = pd.read_csv(output_file)
            return set(df['代码'].unique().astype(str))
        except Exception as e:
            print(f"加载已下载代码失败: {str(e)}")
            return set()
    return set()


def get_recent_days_range(days=CONFIG['RECENT_DAYS']):
    """获取最近days天的日期范围"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)
    return start_date.strftime('%Y/%m/%d'), end_date.strftime('%Y/%m/%d')


def filter_recent_data(df, days=CONFIG['RECENT_DAYS']):
    """过滤出最近days天的数据"""
    if df is None or df.empty:
        return df

    # 确保日期列是datetime类型
    df['日期'] = pd.to_datetime(df['日期'])

    # 获取最近days天的日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    # 过滤数据
    mask = (df['日期'] >= start_date) & (df['日期'] <= end_date)
    return df[mask].copy()


def main():
    current_dir = os.getcwd()

    try:
        # 获取最新的全部Ａ股文件
        latest_file, date_str = get_latest_a_stock_file(current_dir)
        file_path = os.path.join(current_dir, latest_file)
        print(f"找到最新文件: {latest_file}")

        # 读取股票代码
        stock_codes = read_stock_codes(file_path)
        print(f"共找到 {len(stock_codes)} 只股票")
        print(f"配置参数: 获取最近{CONFIG['RECENT_DAYS']}天数据, 最大重试{CONFIG['MAX_RETRY']}次")

        # 准备输出文件名
        max_date = datetime.now().strftime('%Y%m%d')
        output_file = f"增量_资金流向数据_{max_date}.csv"

        # 批量下载资金流向数据
        all_data = []
        failed_codes = []

        for i, code in enumerate(tqdm(stock_codes, desc=f"下载最近{CONFIG['RECENT_DAYS']}天资金流向数据"), 1):
            df = get_recent_days_data(code)

            if df is not None:
                # 再次确认只保留配置天数的数据（双重保险）
                df = filter_recent_data(df)
                if not df.empty:
                    all_data.append(df)
            else:
                failed_codes.append(code)

            # 随机停留以防止被封
            if i % 100 == 0 or i == len(stock_codes):
                sleep_time = random.uniform(*CONFIG['SLEEP_RANGE'])
                print(f"进度: {i}/{len(stock_codes)} | 失败: {len(failed_codes)} | 停留: {sleep_time:.2f}秒")
                time.sleep(sleep_time)

        # 保存所有数据
        if all_data:
            try:
                final_df = pd.concat(all_data, ignore_index=True).drop_duplicates(['代码', '日期'])
                save_final_data(final_df, output_file)
                print(f"总股票数: {len(final_df['代码'].unique())}")
                print(f"总记录数: {len(final_df)}")
                if failed_codes:
                    print(f"失败的股票代码({len(failed_codes)}只): {failed_codes[:10]}...")  # 只显示前10个
            except Exception as e:
                print(f"保存最终数据时出错: {str(e)}")
        else:
            print("没有获取到任何有效数据")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()