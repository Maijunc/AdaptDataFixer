#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
规则注册表
"""
import logging
from typing import List, Dict, Optional, Type
from .base_rule import BaseRepairRule

logger = logging.getLogger(__name__)

# 规则注册表
_rule_registry: Dict[str, BaseRepairRule] = {}


def register_rule(rule: BaseRepairRule) -> None:
    """
    注册一个修复规则

    Args:
        rule (BaseRepairRule): 要注册的规则实例
    """
    if rule.name in _rule_registry:
        logger.warning(f"规则 '{rule.name}' 已存在，将被覆盖")

    _rule_registry[rule.name] = rule
    logger.debug(f"已注册规则: {rule.name}")


def get_all_rules() -> List[BaseRepairRule]:
    """
    获取所有注册的规则

    Returns:
        List[BaseRepairRule]: 规则实例列表
    """
    return list(_rule_registry.values())


def get_rule_by_name(name: str) -> Optional[BaseRepairRule]:
    """
    根据名称获取规则

    Args:
        name (str): 规则名称

    Returns:
        Optional[BaseRepairRule]: 如果找到则返回规则实例，否则返回None
    """
    return _rule_registry.get(name)


def clear_rules() -> None:
    """清空规则注册表"""
    _rule_registry.clear()
    logger.debug("已清空规则注册表")


def register_default_rules() -> None:
    """注册默认的修复规则"""
    from .fengdan_repair import FengdanRepairRule
    from .price_repair import PriceRepairRule
    from .trading_repair import TradingRepairRule
    from .financial_repair import FinancialRepairRule
    from .market_feature_repair import MarketFeatureRepairRule

    # 注册封单修复规则
    register_rule(FengdanRepairRule())

    # 注册价格修复规则
    register_rule(PriceRepairRule())

    # 注册交易数据修复规则
    register_rule(TradingRepairRule())

    # 注册财务指标修复规则
    register_rule(FinancialRepairRule())

    # 注册市场特征修复规则
    register_rule(MarketFeatureRepairRule())

    logger.info("已注册所有默认修复规则")
