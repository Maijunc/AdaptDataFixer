#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通达信数据修复主程序
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import random

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handlers.data_loader import load_data, save_data
from src.repair_rules import RepairManager, register_default_rules, get_all_rules
from src.utils.logging_utils import setup_logger

# 程序配置
CONFIG = {
    # 基本配置
    'input_file': "通达信数据.csv",  # 输入文件路径
    'output_file': None,  # 输出文件路径（None表示自动生成名称：repaired_输入文件名）
    'fix_all': True,  # True: 全量修复模式，False: 增量修复模式
    'verbose': True,  # 是否显示详细日志
    
    # 采样配置
    'use_sample': False,  # 是否启用数据采样
    'sample_mode': 'stocks',  # 采样模式: 'stocks' 按股票采样, 'rows' 按行数采样
    'sample_stocks': 10,  # 采样股票数量（当sample_mode='stocks'时生效）
    'sample_size': 1000,  # 采样数据行数（当sample_mode='rows'时生效）
    'random_seed': 42,  # 随机采样的种子值
}


def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    # 设置控制台彩色日志
    logger = setup_logger(__name__, log_level)
    
    # 添加文件处理器用于保存日志到文件
    log_file = f'repair_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def sample_data(df, mode='rows', sample_size=1000, sample_stocks=None, random_seed=42):
    """
    对数据进行采样
    
    Args:
        df (pd.DataFrame): 原始数据框
        mode (str): 采样模式，'rows'按行数采样，'stocks'按股票采样
        sample_size (int): 采样行数
        sample_stocks (int): 采样股票数量
        random_seed (int): 随机种子
        
    Returns:
        pd.DataFrame: 采样后的数据框
    """
    random.seed(random_seed)
    
    # 如果是按股票采样模式，且指定了股票数量
    if mode == 'stocks' and sample_stocks and sample_stocks > 0:
        unique_stocks = df['代码'].unique()
        if sample_stocks >= len(unique_stocks):
            return df  # 如果采样数量大于等于总股票数，返回全部数据
        
        # 随机选择指定数量的股票
        selected_stocks = random.sample(list(unique_stocks), sample_stocks)
        return df[df['代码'].isin(selected_stocks)]
    
    # 否则按行数采样
    if sample_size >= len(df):
        return df  # 如果采样数量大于等于总行数，返回全部数据
    
    return df.sample(sample_size, random_state=random_seed)


def main():
    """主函数"""
    # 从配置中获取参数
    input_file = CONFIG['input_file']
    output_file = CONFIG['output_file'] or f"repaired_{os.path.basename(input_file)}"
    fix_all = CONFIG['fix_all']
    verbose = CONFIG['verbose']
    
    # 采样参数
    use_sample = CONFIG['use_sample']
    sample_mode = CONFIG['sample_mode']
    sample_size = CONFIG['sample_size']
    sample_stocks = CONFIG['sample_stocks']
    random_seed = CONFIG['random_seed']

    # 设置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logging(log_level)

    logger.info("通达信数据修复工具启动")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"修复模式: {'全量修复' if fix_all else '增量修复'}")
    
    if use_sample:
        if sample_mode == 'stocks':
            logger.info(f"采样模式: 随机选择 {sample_stocks} 只股票")
        else:
            logger.info(f"采样模式: 随机选择 {sample_size} 条记录")

    try:
        # 加载数据
        logger.info("正在加载数据...")
        df = load_data(input_file)
        logger.info(f"成功加载数据，共 {len(df)} 条记录，{len(df['代码'].unique())} 只股票")
        
        # 如果启用采样模式，对数据进行采样
        if use_sample:
            logger.info("正在进行数据采样...")
            original_size = len(df)
            original_stocks = len(df['代码'].unique())
            
            df = sample_data(df, sample_mode, sample_size, sample_stocks, random_seed)
            
            logger.info(f"采样完成: 从 {original_size} 条记录减少到 {len(df)} 条")
            logger.info(f"采样股票数: 从 {original_stocks} 只减少到 {len(df['代码'].unique())} 只")
            
            # 保存采样后但未修复的数据
            sample_original_file = f"sample_original_{os.path.basename(input_file)}"
            logger.info(f"正在保存采样数据（未修复）到 {sample_original_file}...")
            save_data(df, sample_original_file)

        # 注册修复规则
        logger.info("正在注册修复规则...")
        register_default_rules()
        rules = get_all_rules()
        logger.info(f"已注册 {len(rules)} 条修复规则")

        # 创建修复管理器
        repair_manager = RepairManager(fix_all=fix_all)
        repair_manager.register_rules(rules)

        # 执行修复
        logger.info("开始执行数据修复...")
        df_repaired = repair_manager.repair(df)

        # 获取修复统计信息
        stats = repair_manager.get_stats()
        logger.info(f"数据修复完成，共修复 {stats['total_repairs']} 处数据")

        # 输出详细统计信息
        for rule_name, repair_count in stats['rule_stats'].items():
            logger.info(f"规则 '{rule_name}' 修复了 {repair_count} 处数据")

        if 'column_repair_rates' in stats:
            logger.info("各列原始0值数量统计:")
            for col, info in stats['column_repair_rates'].items():
                logger.info(f"  {col}: 原始0值数量 {info['original_zeros']}")

        # 保存结果
        logger.info(f"正在保存修复后的数据到 {output_file}...")
        save_data(df_repaired, output_file)

        logger.info("数据修复流程完成")
        return 0

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!