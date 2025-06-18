#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彩色日志工具模块
"""
import logging
import colorlog
import sys

# 定义日志颜色
COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

def setup_logger(name=None, level=logging.INFO):
    """
    设置彩色日志记录器

    Args:
        name (str, optional): 日志记录器名称. Defaults to None.
        level (int, optional): 日志级别. Defaults to logging.INFO.

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果已经有处理器，说明已经初始化过，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 创建彩色格式器
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors=COLORS
    )

    # 将格式器添加到处理器
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)

    # 阻止日志传递到父记录器
    logger.propagate = False

    return logger

def get_logger(name=None):
    """
    获取已配置的日志记录器，如果不存在则创建新的

    Args:
        name (str, optional): 日志记录器名称. Defaults to None.

    Returns:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger

# 示例用法
if __name__ == '__main__':
    # 设置日志记录器
    logger = get_logger(__name__)

    # 测试不同级别的日志
    logger.debug('这是一条调试日志')
    logger.info('这是一条信息日志')
    logger.warning('这是一条警告日志')
    logger.error('这是一条错误日志')
    logger.critical('这是一条严重错误日志')
