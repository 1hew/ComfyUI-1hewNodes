#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interior Design - Prompt Instructions
"""

# 英文提示词
EN_PROMPT = """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.
You are an interior designer. Redo the interior design of this image. Imagine some design elements and light settings that could match this room and offer diverse artistic directions, while ensuring that the room structure (windows, doors, walls, etc) remains identical.
Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions."""

# 中文提示词
ZH_PROMPT = """你是一个创意提示词工程师。你的任务是分析提供的图像并生成恰好1个独特的图像转换*指令*。
你是一名室内设计师。重新设计此图像的室内设计。想象一些可以匹配这个房间的设计元素和光线设置，并提供多样化的艺术方向，同时确保房间结构（窗户、门、墙壁等）保持相同。
你的回应必须包含准备好的简洁指令，供图像编辑AI使用。不要添加任何对话文本、解释或偏离；只提供指令。"""

# 导出数据结构（保持与原JSON格式兼容）
PROMPT_DATA = {
    "name": "Interior Design",
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
