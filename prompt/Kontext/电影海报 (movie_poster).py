#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Movie Poster - Prompt Instructions
"""

# 英文提示词
EN_PROMPT = """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.
Create a movie poster with the subjects of this image as the main characters. Take a random genre (action, comedy, horror, etc) and make it look like a movie poster. Sometimes, the user would provide a title for the movie (not always). In this case the user provided: . Otherwise, you can make up a title based on the image. If a title is provided, try to fit the scene to the title, otherwise get inspired by elements of the image to make up a movie. Make sure the title is stylized and add some taglines too. Add lots of text like quotes and other text we typically see in movie posters.
Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions."""

# 中文提示词
ZH_PROMPT = """你是一个创意提示词工程师。你的任务是分析提供的图像并生成恰好1个独特的图像转换*指令*。
以此图像的主体作为主要角色创建电影海报。选择一个随机类型（动作、喜剧、恐怖等）并使其看起来像电影海报。有时，用户会提供电影标题（并非总是）。在这种情况下，用户提供了：。否则，你可以根据图像编造一个标题。如果提供了标题，尝试使场景适合标题，否则从图像元素中获得灵感来编造电影。确保标题风格化并添加一些标语。添加大量文本，如引用和我们在电影海报中通常看到的其他文本。
你的回应必须包含准备好的简洁指令，供图像编辑AI使用。不要添加任何对话文本、解释或偏离；只提供指令。"""

# 导出数据结构（保持与原JSON格式兼容）
PROMPT_DATA = {
    "name": "Movie Poster",
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
