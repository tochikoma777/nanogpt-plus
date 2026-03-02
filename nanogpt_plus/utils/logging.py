"""
结构化日志系统
支持控制台和文件输出，JSON格式便于分析
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取配置好的logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 格式化
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_file_logging(logger: logging.Logger, log_file: str):
    """添加文件日志"""
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
