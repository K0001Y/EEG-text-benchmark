"""配置管理模块。

提供统一的配置加载和管理功能。
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载评估配置。

    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置

    Returns:
        配置字典
    """
    if config_path is None:
        # 使用默认配置
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, "eval_config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典。

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """获取特定模型的配置。

    Args:
        config: 全局配置
        model_name: 模型名称

    Returns:
        模型配置字典
    """
    models_config = config.get("models", {})
    return models_config.get(model_name, {})
