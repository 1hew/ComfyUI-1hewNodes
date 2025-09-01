#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move Camera - Prompt Instructions
"""

# 英文提示词
EN_PROMPT = """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.
Move the camera to reveal new aspects of the scene. Provide highly different types of camera mouvements based on the scene (eg: the camera now gives a top view of the room; side portrait view of the person..etc ).
Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions."""

# 中文提示词
ZH_PROMPT = """你是一个创意提示词工程师。你的任务是分析提供的图像并生成恰好1个独特的图像转换*指令*。
移动摄像机以揭示场景的新方面。根据场景提供高度不同类型的摄像机运动（例如：摄像机现在给出房间的俯视图；人物的侧面肖像视图等）。
你的回应必须包含准备好的简洁指令，供图像编辑AI使用。不要添加任何对话文本、解释或偏离；只提供指令。"""

# 导出数据结构（保持与原JSON格式兼容）
PROMPT_DATA = {
    "name": "Move Camera",
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
