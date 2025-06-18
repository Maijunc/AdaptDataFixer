import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.font_manager as fm
import logging

# 获取模块级别的logger
logger = logging.getLogger(__name__)

def init_logging(log_file='compare_data.log'):
    """初始化日志配置"""
    # 创建一个文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 获取根日志记录器并添加文件处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 移除所有现有的处理器（避免重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加文件处理器
    root_logger.addHandler(file_handler)

def setup_chinese_font():
    """设置中文字体"""
    # 更全面的中文字体列表
    preferred_fonts = [
        # macOS 字体
        "PingFang SC", "STHeiti", "Heiti TC", "Hiragino Sans GB",
        # Windows 字体
        "Microsoft YaHei", "SimHei", "SimSun", "NSimSun",
        # Linux 字体
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK TC",
        # 通用后备字体
        "Arial Unicode MS"
    ]
    
    # 获取系统所有字体
    font_paths = fm.findSystemFonts()
    available_fonts = {}
    
    # 构建字体映射
    for font_path in font_paths:
        try:
            font = fm.FontProperties(fname=font_path)
            font_name = font.get_name()
            available_fonts[font_name] = font_path
            logger.debug(f"找到字体: {font_name} at {font_path}")
        except Exception as e:
            logger.debug(f"处理字体时出错 {font_path}: {e}")
    
    # 查找第一个可用的中文字体
    for font_name in preferred_fonts:
        # 检查完整字体名称
        if font_name in available_fonts:
            logger.info(f"使用中文字体: {font_name}")
            plt.rcParams["font.sans-serif"] = [font_name]
            return True
        
        # 检查部分匹配
        for available_name in available_fonts:
            if font_name.lower() in available_name.lower():
                logger.info(f"使用中文字体: {available_name}")
                plt.rcParams["font.sans-serif"] = [available_name]
                return True
    
    # 如果没有找到任何中文字体，尝试使用系统默认字体
    logger.warning("未找到中文字体，尝试使用系统默认字体")
    fallback_font = fm.findfont(fm.FontProperties(family=["sans-serif"]))
    if fallback_font:
        font_name = fm.FontProperties(fname=fallback_font).get_name()
        logger.info(f"使用系统默认字体: {font_name}")
        plt.rcParams["font.sans-serif"] = [font_name]
    else:
        logger.error("无法找到任何可用字体")
        return False
    
    return True

def setup_matplotlib():
    """设置matplotlib全局配置"""
    # 设置中文字体
    setup_chinese_font()
    # 设置负号显示
    plt.rcParams["axes.unicode_minus"] = False

def compare_dataframes(df1, df2, name1="未修复数据", name2="已修复数据"):
    """比较两个DataFrame的差异"""
    comparison_results = []

    # 1. 基础统计信息
    total_cells1 = df1.size
    total_cells2 = df2.size
    null_cells1 = df1.isnull().sum().sum()
    null_cells2 = df2.isnull().sum().sum()
    zero_cells1 = (df1 == 0).sum().sum()
    zero_cells2 = (df2 == 0).sum().sum()

    comparison_results.append(f"\n=== 基础统计信息 ===")
    comparison_results.append(f"{name1}:")
    comparison_results.append(f"- 总单元格数: {total_cells1}")
    comparison_results.append(f"- 空值数量: {null_cells1}")
    comparison_results.append(f"- 零值数量: {zero_cells1}")
    comparison_results.append(f"- 空值比例: {(null_cells1/total_cells1)*100:.2f}%")
    comparison_results.append(f"- 零值比例: {(zero_cells1/total_cells1)*100:.2f}%")

    comparison_results.append(f"\n{name2}:")
    comparison_results.append(f"- 总单元格数: {total_cells2}")
    comparison_results.append(f"- 空值数量: {null_cells2}")
    comparison_results.append(f"- 零值数量: {zero_cells2}")
    comparison_results.append(f"- 空值比例: {(null_cells2/total_cells2)*100:.2f}%")
    comparison_results.append(f"- 零值比例: {(zero_cells2/total_cells2)*100:.2f}%")

    # 2. 列级别的比较
    comparison_results.append("\n=== 列级别比较 ===")
    for column in df1.columns:
        if column in df2.columns:
            null_before = df1[column].isnull().sum()
            null_after = df2[column].isnull().sum()
            zero_before = (df1[column] == 0).sum()
            zero_after = (df2[column] == 0).sum()

            # 只有当有差异时才显示
            if null_before != null_after or zero_before != zero_after:
                comparison_results.append(f"\n列: {column}")
                if null_before != null_after:
                    repair_rate = ((null_before - null_after) / null_before * 100) if null_before > 0 else 0
                    comparison_results.append(f"- 空值修复: {null_before} -> {null_after} (修复率: {repair_rate:.2f}%)")
                if zero_before != zero_after:
                    repair_rate = ((zero_before - zero_after) / zero_before * 100) if zero_before > 0 else 0
                    comparison_results.append(f"- 零值修复: {zero_before} -> {zero_after} (修复率: {repair_rate:.2f}%)")

    return "\n".join(comparison_results)

def visualize_comparison(df1, df2, name1="未修复数据", name2="已修复数据"):
    """创建数据比较的可视化图表"""
    try:
        # 确保字体设置正确
        logger.info("开始创建可视化图表...")
        
        # 1. 计算每列的空值比例
        null_ratio1 = (df1.isnull().sum() / len(df1)) * 100
        null_ratio2 = (df2.isnull().sum() / len(df2)) * 100

        # 2. 计算每列的零值比例
        zero_ratio1 = (df1 == 0).sum() / len(df1) * 100
        zero_ratio2 = (df2 == 0).sum() / len(df2) * 100

        # 创建图表，设置DPI以提高清晰度
        plt.figure(figsize=(15, 10), dpi=300)
        
        # 设置全局字体大小
        plt.rcParams['font.size'] = 12

        # 空值比例对比图
        plt.subplot(2, 1, 1)
        columns_with_nulls = null_ratio1[null_ratio1 > 0].index.union(null_ratio2[null_ratio2 > 0].index)
        if len(columns_with_nulls) > 0:
            data = pd.DataFrame({
                name1: null_ratio1[columns_with_nulls],
                name2: null_ratio2[columns_with_nulls]
            })
            data.plot(kind='bar', ax=plt.gca())
            plt.title('各列空值比例对比')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('空值比例 (%)')
            plt.legend()

        # 零值比例对比图
        plt.subplot(2, 1, 2)
        columns_with_zeros = zero_ratio1[zero_ratio1 > 0].index.union(zero_ratio2[zero_ratio2 > 0].index)
        if len(columns_with_zeros) > 0:
            data = pd.DataFrame({
                name1: zero_ratio1[columns_with_zeros],
                name2: zero_ratio2[columns_with_zeros]
            })
            data.plot(kind='bar', ax=plt.gca())
            plt.title('各列零值比例对比')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('零值比例 (%)')
            plt.legend()

        plt.tight_layout()
        
        # 保存图表，使用高质量设置
        output_file = 'data_comparison-2.png'
        logger.info(f"保存图表到 {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("图表创建完成")
        
    except Exception as e:
        logger.error(f"创建可视化图表时出错: {e}")
        # 尝试使用最基本的设置重新创建图表
        try:
            plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, f"图表创建失败: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig('data_comparison-error.png')
            plt.close()
            logger.info("已创建错误信息图表")
        except:
            logger.error("无法创建错误信息图表")

def compare_files(file1_path, file2_path, output_path=None):
    """比较两个CSV文件的差异并生成报告"""
    try:
        logger.info(f"开始比较文件: {file1_path} 和 {file2_path}")
        
        # 读取CSV文件
        logger.info(f"读取文件 {file1_path}")
        df1 = pd.read_csv(file1_path, encoding='utf-8-sig')
        logger.info(f"读取文件 {file2_path}")
        df2 = pd.read_csv(file2_path, encoding='utf-8-sig')
        
        logger.info(f"文件1: {len(df1)}行 x {len(df1.columns)}列")
        logger.info(f"文件2: {len(df2)}行 x {len(df2.columns)}列")

        # 生成比较报告
        logger.info("生成比较报告文本")
        comparison_text = compare_dataframes(df1, df2)

        # 创建可视化图表
        logger.info("创建可视化图表")
        visualize_comparison(df1, df2)

        # 保存报告
        if output_path:
            logger.info(f"保存比较报告到 {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(comparison_text)
            logger.info(f"比较报告已保存到: {output_path}")

        logger.info("比较完成")
        return comparison_text

    except Exception as e:
        logger.error(f"比较过程中出错: {str(e)}", exc_info=True)
        return f"比较过程中出错: {str(e)}"

if __name__ == "__main__":
    # 文件路径
    file1_path = "通达信数据.csv"
    file2_path = "repaired_通达信数据.csv"
    output_path = "comparison_report.txt"
    log_path = "data_comparison.log"

    # 初始化日志和matplotlib配置
    init_logging(log_path)
    setup_matplotlib()

    # 执行比较
    result = compare_files(file1_path, file2_path, output_path)
    
    # 只打印比较结果，不打印日志
    if not result.startswith("比较过程中出错"):
        print("\n=== 数据比较结果 ===")
        print(result)
    else:
        print("\n=== 错误信息 ===")
        print(result)
        print(f"详细日志请查看: {log_path}")
