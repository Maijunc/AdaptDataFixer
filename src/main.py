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

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handlers.data_loader import load_data, save_data
from src.repair_rules import RepairManager, register_default_rules, get_all_rules
from src.utils.logging_utils import setup_logger


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


def main():
    """主函数"""
    # 手动设置参数
    input_file = "通达信数据.csv"  # 修改为实际输入文件路径
    output_file = f"repaired_{os.path.basename(input_file)}"
    fix_all = True
    verbose = True

    # 设置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logging(log_level)

    logger.info("通达信数据修复工具启动")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"修复模式: {'全量修复' if fix_all else '增量修复'}")

    try:
        # 加载数据
        logger.info("正在加载数据...")
        df = load_data(input_file)
        logger.info(f"成功加载数据，共 {len(df)} 条记录")

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