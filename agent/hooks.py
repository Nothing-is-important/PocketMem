"""Hook 生命周期系统。

在 Agent 执行的 4 个关键节点预留挂载点，允许通过注册 hook 函数扩展功能，
不需修改 Agent 核心代码。

挂载点：
    pre_route      - Router 执行前（查询预处理）
    pre_retrieve   - Retriever 执行前（缓存检查、用户画像注入）
    post_retrieve  - Retriever 执行后（时间衰减、结果过滤）
    post_generate  - Generator 执行后（日志记录、用户画像更新）

设计原则：
    - 每个挂载点可注册多个 hook，按注册顺序串联执行（管道模式）
    - Hook 失败不阻塞主流程——记录日志后继续
    - Hook 函数签名: (state: dict) -> state: dict
"""

from typing import Any, Callable, Dict, List
from utils import get_logger

logger = get_logger("hooks")

HookFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


class HookRegistry:
    """轻量级 Hook 注册中心。

    用法:
        from agent.hooks import hooks

        def my_hook(state):
            state["custom_field"] = "value"
            return state

        hooks.register("pre_retrieve", my_hook)
    """

    VALID_POINTS = {"pre_route", "pre_retrieve", "post_retrieve", "post_generate"}

    def __init__(self):
        self._hooks: Dict[str, List[HookFunc]] = {
            point: [] for point in self.VALID_POINTS
        }

    def register(self, point: str, hook: HookFunc):
        """注册一个 hook 到指定挂载点。

        Args:
            point: 挂载点名称（pre_route/pre_retrieve/post_retrieve/post_generate）
            hook: Hook 函数，签名为 (state: dict) -> state: dict
        """
        if point not in self.VALID_POINTS:
            raise ValueError(
                f"Unknown hook point: '{point}'. "
                f"Available: {sorted(self.VALID_POINTS)}"
            )
        self._hooks[point].append(hook)
        logger.debug("Hook registered at %s: %s", point, hook.__name__)

    def run(self, point: str, state: dict) -> dict:
        """执行指定挂载点的所有 hook。

        Hook 按注册顺序串联执行——每个 hook 的输出作为下一个 hook 的输入。
        任何 hook 失败都会被捕获并记录日志，不阻塞主流程。
        """
        for hook in self._hooks.get(point, []):
            try:
                state = hook(state)
            except Exception as e:
                logger.warning(
                    "Hook '%s' at point '%s' failed: %s",
                    getattr(hook, "__name__", str(hook)), point, e
                )
        return state

    def list_hooks(self) -> Dict[str, List[str]]:
        """列出所有已注册的 hook。"""
        return {
            point: [getattr(h, "__name__", str(h)) for h in hooks_list]
            for point, hooks_list in self._hooks.items()
            if hooks_list
        }

    def clear(self):
        """清空所有 hook（测试用）。"""
        for point in self.VALID_POINTS:
            self._hooks[point].clear()


# 全局单例
hooks = HookRegistry()
