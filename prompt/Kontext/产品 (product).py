#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Product - Prompt Instructions
"""

# 英文提示词
EN_PROMPT = """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.
Turn this image into the style of a professional product photo. Describe a variety of scenes (simple packshot or the item being used), so that it could show different aspects of the item in a highly professional catalog. Suggest a variety of scenes, light settings and camera angles/framings, zoom levels, etc. Suggest at least 1 scenario of how the item is used.
Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions."""

# 中文提示词
ZH_PROMPT = """你是一个创意提示词工程师。你的任务是分析提供的图像并生成恰好1个独特的图像转换*指令*。
将此图像转换为专业产品照片的风格。描述各种场景（简单的产品拍摄或物品使用中），以便在高度专业的目录中展示物品的不同方面。建议各种场景、光线设置和摄像机角度/构图、缩放级别等。至少建议1个物品使用场景。
你的回应必须包含准备好的简洁指令，供图像编辑AI使用。不要添加任何对话文本、解释或偏离；只提供指令。"""

# 导出数据结构（保持与原JSON格式兼容）
PROMPT_DATA = {
    "name": "Product",
    "en": EN_PROMPT,
    "zh": ZH_PROMPT
}

# 获取提示词的便捷函数
def get_prompt(language="en"):
    """
    获取指定语言的提示词
    
    Args:
        language (str): 语言代码，"en" 或 "zh"
    
    Returns:
        str: 对应语言的提示词
    """
    if language.lower() == "zh":
        return ZH_PROMPT
    else:
        return EN_PROMPT

def get_data():
    """
    获取完整的提示词数据
    
    Returns:
        dict: 包含name、en、zh的字典
    """
    return PROMPT_DATA
