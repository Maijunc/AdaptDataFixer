#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复管理器
"""
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)


class RepairManager:
    """修复管理器，负责协调和执行所有修复规则"""

    def __init__(self, fix_all: bool = True):
        """
        初始化修复管理器

        Args:
            fix_all (bool): 是否全量修复，False则只修复最近数据
        """
        self.fix_all = fix_all
        self.rules: List[BaseRepairRule] = []
        self.stats: Dict[str, Any] = {
            'total_repairs': 0,
            'rule_stats': {},
            'column_repair_rates': {}
        }

    def register_rules(self, rules: List[BaseRepairRule]) -> None:
        """
        注册修复规则

        Args:
            rules (List[BaseRepairRule]): 要注册的规则列表
        """
        self.rules.extend(rules)
        logger.info(f"已注册 {len(rules)} 条修复规则")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据，包括数据筛选和基本预处理

        Args:
            df (pd.DataFrame): 原始数据框

        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        # 如果不是全量修复，只保留最近10天的数据
        if not self.fix_all:
            recent_dates = df['日期'].unique()
            recent_dates.sort()
            recent_dates = recent_dates[-10:] if len(recent_dates) >= 10 else recent_dates
            df = df[df['日期'].isin(recent_dates)]
            logger.info(f"增量修复模式，保留最近 {len(recent_dates)} 天数据")

        # 记录原始零值数量
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.stats['column_repair_rates'][col] = {
                'original_zeros': (df[col] == 0).sum(),
                'repaired': 0
            }

        return df

    def _update_stats(self, rule_name: str, repair_count: int, group: pd.DataFrame) -> None:
        """
        更新修复统计信息

        Args:
            rule_name (str): 规则名称
            repair_count (int): 修复数量
            group (pd.DataFrame): 当前处理的数据组
        """
        # 更新总修复数量 - 使用规则报告的修复数量，而不是计算的数量
        self.stats['total_repairs'] += repair_count

        # 更新规则统计
        if rule_name not in self.stats['rule_stats']:
            self.stats['rule_stats'][rule_name] = 0
        self.stats['rule_stats'][rule_name] += repair_count

        # 更新列修复率 - 不再尝试计算每列的修复数量
        # 因为这种计算方式可能导致统计不准确，特别是当多个规则修改同一列时
        # 我们只使用规则自己报告的修复数量

    def _calculate_repair_rates(self) -> None:
        """计算各列的修复率 - 现在只保留原始零值数量的统计"""
        # 由于我们不再跟踪每列的修复数量，这个方法现在只是一个占位符
        # 我们保留原始零值数量的统计，但不再计算修复率
        for col_name, col_info in self.stats['column_repair_rates'].items():
            col_info['repair_rate'] = 0  # 不再计算具体修复率
            col_info['repaired'] = 0     # 不再跟踪每列的修复数量

    def repair(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行数据修复

        Args:
            df (pd.DataFrame): 要修复的数据框

        Returns:
            pd.DataFrame: 修复后的数据框
        """
        logger.info("开始数据修复流程")

        # 准备数据
        df = self._prepare_data(df)

        # 按股票代码分组处理
        groups = []
        stock_codes = df['代码'].unique()

        # 使用tqdm创建进度条
        with tqdm(total=len(stock_codes), desc="修复进度") as pbar:
            for code in stock_codes:
                # 获取当前股票的数据
                group = df[df['代码'] == code].copy()
                group = group.sort_values('日期')

                # 应用每个修复规则
                for rule in self.rules:
                    try:
                        # 应用规则并获取修复结果
                        group, repair_count = rule.apply(group)

                        # 更新统计信息
                        self._update_stats(rule.name, repair_count, group)

                    except Exception as e:
                        logger.error(f"应用规则 '{rule.name}' 到股票 {code} 时发生错误: {str(e)}")
                        continue

                # 添加修复后的分组数据
                groups.append(group)
                pbar.update(1)

        # 合并所有分组
        df_repaired = pd.concat(groups, axis=0)

        # 重置索引
        df_repaired.reset_index(drop=True, inplace=True)

        # 计算修复率
        self._calculate_repair_rates()

        logger.info(f"数据修复完成，共修复 {self.stats['total_repairs']} 处数据")
        return df_repaired

    def get_stats(self) -> Dict[str, Any]:
        """
        获取修复统计信息

        Returns:
            Dict[str, Any]: 包含修复统计信息的字典
        """
        return self.stats
