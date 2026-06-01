"""Prompt 集中管理器。

将所有 LLM Prompt 从代码中抽离到 YAML 配置文件，
支持版本记录、迭代历史和热加载。

用法:
    from config.prompts import prompts
    prompt_text = prompts.get("router")
    version = prompts.version("router")
    history = prompts.changelog("router")  # 面试展示 Prompt 迭代过程
"""

import yaml
from pathlib import Path
from typing import Dict, List


class PromptLoader:
    """集中管理所有 LLM Prompt。

    特性：
    - 启动时加载所有 config/prompts/*.yaml
    - 每个 Prompt 带版本号和 changelog
    - 支持 reload() 热加载（改 YAML 不需要重启服务）
    """

    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        else:
            prompts_dir = Path(prompts_dir)
        self._dir = prompts_dir
        self._cache: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        """加载所有 YAML prompt 文件。"""
        for yaml_file in sorted(self._dir.glob("*.yaml")):
            name = yaml_file.stem
            with open(yaml_file, "r", encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f)

    def get(self, name: str) -> str:
        """获取 prompt 文本。"""
        entry = self._cache.get(name, {})
        return entry.get("prompt", "")

    def version(self, name: str) -> int:
        """获取当前版本号。"""
        return self._cache.get(name, {}).get("version", 0)

    def changelog(self, name: str) -> List[str]:
        """获取迭代历史（面试时展示 Prompt 演进过程）。"""
        return self._cache.get(name, {}).get("changelog", [])

    def meta(self, name: str) -> dict:
        """获取完整元数据（版本+用途+历史）。"""
        entry = self._cache.get(name, {})
        return {
            "version": entry.get("version", 0),
            "purpose": entry.get("purpose", ""),
            "changelog": entry.get("changelog", []),
        }

    def reload(self):
        """热加载所有 prompt（不重启服务）。"""
        self._cache.clear()
        self._load_all()

    @property
    def loaded_prompts(self) -> List[str]:
        """已加载的 prompt 名称列表。"""
        return sorted(self._cache.keys())


# 全局单例
prompts = PromptLoader()
