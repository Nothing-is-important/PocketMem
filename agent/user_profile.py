"""用户画像管理器。

通过 Hook 系统自动记录用户查询行为，在下次查询时注入个性化上下文。

特性：
    - 自动提取高频查询词和关注话题
    - 跟踪常用联系人的查询频率
    - 记录记忆片段的访问热度（高频访问 = 重要内容）
    - 通过 post_generate hook 自动更新，零侵入 Agent 核心代码
    - 纯 JSON 存储，不需要额外数据库

用法:
    from agent.user_profile import profile

    # 自动记录（通过 hook 系统，不需要手动调用）
    # profile.record_query(query, intent)

    # 注入到 generator prompt
    # context = profile.inject_context(query)
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import get_logger

logger = get_logger("profile")

# 停用词——查询中常见但不携带意图信息的词
_QUERY_STOP_WORDS = {
    "是", "的", "了", "吗", "呢", "啊", "吧", "什么", "怎么", "为什么",
    "哪里", "哪个", "哪位", "多少", "有没有", "是不是", "能不能",
    "关于", "一个", "一下", "一些", "这个", "那个", "这些", "那些",
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "在", "有", "和", "与", "或", "的", "地", "得",
    "请", "帮", "给", "让", "用", "把",
    "一下", "告诉我", "查一下", "搜一下", "找一下",
}


class UserProfile:
    """用户画像管理器——让 Agent 越用越懂你。

    数据存储在 ~/.pocket_memory/user_profile.json
    """

    MAX_FREQUENT_TERMS = 30  # 最多保留的高频词数量
    MAX_CONTACTS = 20        # 最多跟踪的联系人数量
    SAVE_INTERVAL = 10       # 每 N 次查询保存一次（避免频繁写盘）

    def __init__(self, profile_path: str = None):
        if profile_path is None:
            profile_path = str(Path.home() / ".pocket_memory" / "user_profile.json")
        self._path = Path(profile_path)
        self._data = self._load()
        self._query_count_since_save = 0

    # ═══════════════════════════════════════════════════════
    # 记录（通过 post_generate hook 自动调用）
    # ═══════════════════════════════════════════════════════

    def record_query(self, query: str, intent: str = "", contacts: List[str] = None):
        """记录一次查询（在 post_generate hook 中调用）。

        Args:
            query: 用户查询文本
            intent: Router 分类结果
            contacts: 查询中提及的联系人（从 entity_extractor 获取）
        """
        stats = self._data["query_stats"]
        stats["total_queries"] += 1
        stats["last_session"] = datetime.now().isoformat()

        # 提取高频词（jieba 分词 + 停用词过滤）
        keywords = self._extract_keywords(query)
        freq = stats["frequent_terms"]
        for w in keywords:
            freq[w] = freq.get(w, 0) + 1

        # 保留 Top-N 高频词
        if len(freq) > self.MAX_FREQUENT_TERMS:
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:self.MAX_FREQUENT_TERMS]
            stats["frequent_terms"] = dict(top)

        # 意图分布
        if intent:
            intents = stats.setdefault("frequent_intents", {})
            intents[intent] = intents.get(intent, 0) + 1

        # 联系人提及频率
        if contacts:
            ci = self._data["contact_importance"]
            for name in contacts:
                if name not in ci:
                    ci[name] = {"mention_count": 0}
                ci[name]["mention_count"] += 1
                ci[name]["last_queried"] = datetime.now().isoformat()[:10]

            # 保留 Top-N 联系人
            if len(ci) > self.MAX_CONTACTS:
                top_c = sorted(ci.items(), key=lambda x: x[1]["mention_count"], reverse=True)
                self._data["contact_importance"] = dict(top_c[:self.MAX_CONTACTS])

        self._query_count_since_save += 1
        if self._query_count_since_save >= self.SAVE_INTERVAL:
            self.save()

    def record_access(self, chunk_id: str):
        """记录一次记忆片段访问（热点数据检测）。"""
        history = self._data["access_history"]
        if chunk_id not in history:
            history[chunk_id] = {"access_count": 0}
        history[chunk_id]["access_count"] += 1
        history[chunk_id]["last_accessed"] = datetime.now().isoformat()

    # ═══════════════════════════════════════════════════════
    # 注入（通过 pre_retrieve hook 调用）
    # ═══════════════════════════════════════════════════════

    def inject_context(self, query: str) -> str:
        """生成用户画像上下文文本，注入到 generator prompt。

        Returns:
            描述用户偏好的文本，如：
            "用户经常查询企业合同和技术方案相关的内容。"
            如果画像为空（新用户），返回空字符串。
        """
        parts = []
        freq = self._data["query_stats"].get("frequent_terms", {})

        if freq:
            top_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_terms and top_terms[0][1] >= 2:  # 至少查询 2 次才显示
                terms_str = "、".join(f"「{t}」({c}次)" for t, c in top_terms)
                parts.append(f"用户经常查询与 {terms_str} 相关的内容。")

        contacts = self._data.get("contact_importance", {})
        if contacts:
            top_contacts = sorted(
                contacts.items(),
                key=lambda x: x[1].get("mention_count", 0),
                reverse=True,
            )[:3]
            if top_contacts and top_contacts[0][1].get("mention_count", 0) >= 2:
                contacts_str = "、".join(name for name, _ in top_contacts)
                parts.append(f"用户常提及的人：{contacts_str}。")

        total = self._data["query_stats"]["total_queries"]
        if total >= 5:
            parts.insert(0, f"这是用户第 {total} 次查询。")

        return "\n".join(parts) if parts else ""

    def get_top_contacts(self, n: int = 5) -> List[Tuple[str, int]]:
        """获取 Top-N 常用联系人。"""
        ci = self._data.get("contact_importance", {})
        sorted_c = sorted(ci.items(), key=lambda x: x[1]["mention_count"], reverse=True)
        return [(name, info["mention_count"]) for name, info in sorted_c[:n]]

    # ═══════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════

    def save(self):
        """保存用户画像到磁盘。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        self._query_count_since_save = 0

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load profile, starting fresh")
        return self._default_profile()

    def _default_profile(self) -> dict:
        return {
            "query_stats": {
                "total_queries": 0,
                "last_session": "",
                "frequent_terms": {},
                "frequent_intents": {},
            },
            "contact_importance": {},
            "access_history": {},
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """从查询文本中提取有信息量的关键词。"""
        import jieba
        words = jieba.lcut(text)
        return [
            w for w in words
            if len(w) >= 2
            and w not in _QUERY_STOP_WORDS
            and not all(c in "，。！？；：""''（）【】《》…—·0123456789%￥" for c in w)
        ]

    @property
    def total_queries(self) -> int:
        return self._data["query_stats"]["total_queries"]

    @property
    def is_new_user(self) -> bool:
        return self.total_queries < 5


# 全局单例
profile = UserProfile()
