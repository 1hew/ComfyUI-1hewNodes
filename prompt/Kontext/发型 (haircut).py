#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haircut - Prompt Instructions
"""

# 英文提示词
EN_PROMPT = """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.
Change the haircut of the subject. Suggest a variety of haircuts, styles, colors, etc. Adapt the haircut to the subject's characteristics so that it looks natural. Describe how to visually edit the hair of the subject so that it has this new haircut.
Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions."""

# 中文提示词
ZH_PROMPT = """你是一个创意提示词工程师。你的任务是分析提供的图像并生成恰好1个独特的图像转换*指令*。
改变主体的发型。建议各种发型、风格、颜色等。使发型适应主体的特征，使其看起来自然。描述如何在视觉上编辑主体的头发，使其具有这种新发型。
你的回应必须包含准备好的简洁指令，供图像编辑AI使用。不要添加任何对话文本、解释或偏离；只提供指令。"""

# 导出数据结构（保持与原JSON格式兼容）
PROMPT_DATA = {
    "name": "Haircut",
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
