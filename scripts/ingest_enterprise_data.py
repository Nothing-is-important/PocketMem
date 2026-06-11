"""企业级演示数据生成器 V2 —— 200+ 份文档。

类型分布：
- 日常工作邮件: 50+ 封
- 技术讨论邮件: 30+ 封
- 项目管理邮件: 20+ 封
- 会议纪要: 15+ 份
- 合同文件: 10+ 份（多种合同类型）
- 内部技术文档: 20+ 篇
- 部门公告: 10+ 篇
- 技术问答: 15+ 条

安全分级: public(15%) / internal(70%) / confidential(15%)
数据结构: 每份文档含完整元数据（时间戳 / 参与者 / 分级 / 重要性）
"""

import os, re, random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = "data/demo/enterprise"
random.seed(42)
BASE = datetime.now() - timedelta(days=180)


# ═══════════════════════════════════════════════════════════════
# 企业设定
# ═══════════════════════════════════════════════════════════════

TEAM = {
    "张伟": {"role": "技术总监", "email": "zhangwei@teamind.cn"},
    "李娜": {"role": "算法负责人", "email": "lina@teamind.cn"},
    "王磊": {"role": "后端架构师", "email": "wanglei@teamind.cn"},
    "陈静": {"role": "产品经理", "email": "chenjing@teamind.cn"},
    "赵明": {"role": "前端工程师", "email": "zhaoming@teamind.cn"},
    "刘洋": {"role": "测试工程师", "email": "liuyang@teamind.cn"},
    "周婷": {"role": "安全工程师", "email": "zhouting@teamind.cn"},
    "吴强": {"role": "运维工程师", "email": "wuqiang@teamind.cn"},
    "林芳": {"role": "HR负责人", "email": "linfang@teamind.cn"},
    "马超": {"role": "财务总监", "email": "machao@teamind.cn"},
    "孙悦": {"role": "设计师", "email": "sunyue@teamind.cn"},
    "黄涛": {"role": "数据分析师", "email": "huangtao@teamind.cn"},
}

PROJECTS = {
    "凤凰": "企业知识库智能检索系统",
    "昆仑": "端侧AI推理引擎",
    "泰山": "多模态文档理解平台",
    "凌霄": "Agent编排框架",
    "朱雀": "数据标注与管理平台",
}

ALL_NAMES = list(TEAM.keys())


# ═══════════════════════════════════════════════════════════════
# 生成引擎
# ═══════════════════════════════════════════════════════════════

def _ts(days_ago=None):
    """生成时间戳。"""
    d = days_ago if days_ago is not None else random.randint(0, 180)
    return BASE + timedelta(days=d, hours=random.randint(8, 20), minutes=random.randint(0, 59))


def _pick(people_list, exclude=None, count=1):
    """从人员列表中随机选取。返回单个字符串(count=1)或列表(count>1)。"""
    if isinstance(people_list, str):
        people_list = [people_list]
    pool = [p for p in people_list if p != exclude]
    if count <= 0 or not pool:
        return [] if count <= 0 else []
    count = min(count, len(pool))
    result = random.sample(pool, count)
    return result[0] if count == 1 else result


def _fmt_emails(names):
    """格式化邮件地址列表。"""
    return ", ".join(f"{n} <{TEAM[n]['email']}>" for n in names)


def _write_mail(from_name, to_names, cc_names, subject, body, days_ago, level="internal"):
    """写入一封邮件文件。"""
    ts = _ts(days_ago)
    fname = f"{ts:%Y%m%d_%H%M}_{from_name}_{_safe(subject)}.txt"
    path = os.path.join(OUTPUT_DIR, "emails", fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 确保参数是列表
    if isinstance(to_names, str): to_names = [to_names]
    if isinstance(cc_names, str): cc_names = [cc_names]
    if cc_names is None: cc_names = []

    cc_str = f"\nCc: {_fmt_emails(cc_names)}" if cc_names else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"From: {from_name} <{TEAM[from_name]['email']}>\n")
        f.write(f"To: {_fmt_emails(to_names)}\n")
        if cc_str:
            f.write(cc_str + "\n")
        f.write(f"Date: {ts:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Level: {level}\n")
        f.write(f"\n{body}\n")

def _write_doc(filename, body, level="internal"):
    """写入一份文档文件。"""
    path = os.path.join(OUTPUT_DIR, "docs", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# LEVEL: {level}\n\n{body}")

def _write_meeting(filename, body, level="internal"):
    """写入一份会议纪要。"""
    path = os.path.join(OUTPUT_DIR, "meetings", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)

def _write_contract(filename, body, level="confidential"):
    """写入一份合同文件。"""
    path = os.path.join(OUTPUT_DIR, "contracts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# LEVEL: {level}\n\n{body}")

def _safe(s):
    """安全的文件名片段。"""
    return re.sub(r'[\\/:*?"<>|]', '_', s[:30])


# ═══════════════════════════════════════════════════════════════
# 批量生成模板
# ═══════════════════════════════════════════════════════════════

def _fill(template, **kw):
    """安全格式化：只替换模板中出现的占位符，多余的忽略。"""
    import string
    class SafeDict(dict):
        def __missing__(self, key): return f"{{{key}}}"
    return template.format_map(SafeDict(**kw))

def gen_daily_work_emails(count=50):
    """日常工作邮件。"""
    subjects = [
        "{proj}的进度更新", "{proj}本周计划", "{proj}问题确认",
        "Re: {proj}的需求讨论", "{proj}上线通知", "{proj}测试环境准备",
        "帮忙看一下{proj}的日志", "{proj}的文档已更新", "{proj}的排期调整",
    ]
    bodies = [
        "{user1}你好，{proj}的进度同步一下：已经完成了核心功能开发，正在做联调测试，预计{date}前可以提测。",
        "各位，{proj}项目本周计划如下：\n1. 完成{proj}模块的单元测试\n2. 修复上次review发现的几个问题\n3. 准备{proj}的技术分享材料\n4. 对齐{user1}的需求变更。\n请大家确认各自的任务。",
        "关于{proj}，想确认一下：{user1}是否需要支持新的场景？如果是的话，工作量会增加。请{user2}评估一下优先级。",
        "{user1}今天在测试环境发现了{proj}的一个问题，在边缘场景下会报{error_type}。定位在{proj}模块，麻烦{user2}看一下，紧急程度：{severity}。",
        "{proj}的需求文档已更新，主要变更：\n1. 增加了权限过滤的支持\n2. 优化了查询的交互流程\n3. 去掉了旧版导出功能（使用率低于5%）\n请大家{date}前review反馈。",
        "帮忙看看{proj}的性能数据，{user1}的指标有些异常。感觉是近期的变更导致的，需要{user2}帮忙验证一下。",
    ]
    for i in range(count):
        proj = random.choice(list(PROJECTS.keys()))
        sender = _pick(ALL_NAMES)
        receivers = _pick(ALL_NAMES, sender, random.randint(1, 2))
        cc_count = random.randint(0, 1) if random.random() < 0.3 else 0
        cc = _pick(ALL_NAMES, sender, cc_count) if cc_count else []
        days = random.randint(0, 180)
        date = (_ts(days) + timedelta(days=random.randint(1, 7))).strftime("%m月%d日")
        user1 = _pick(receivers if receivers else ALL_NAMES, sender)
        user2 = _pick(ALL_NAMES, user1)
        subject = _fill(random.choice(subjects), proj=proj, user1=user1, user2=user2, date=date)
        body = _fill(random.choice(bodies), proj=proj, user1=user1, user2=user2, date=date,
                     error_type=random.choice(["超时错误","数据不一致","内存泄漏","响应异常"]),
                     severity=random.choice(["高","中","低"]))
        level = "confidential" if random.random() < 0.05 else ("public" if random.random() < 0.05 else "internal")
        _write_mail(sender, receivers, cc, subject, body, days, level)
    return count


def gen_tech_discussion_emails(count=30):
    """技术讨论邮件：架构决策、代码审查、性能分析。"""
    for i in range(count):
        proj = random.choice(list(PROJECTS.keys()))
        sender = _pick(["李娜", "王磊", "周婷", "吴强"])
        receivers = _pick(ALL_NAMES, sender, random.randint(1, 2))
        days = random.randint(0, 180)
        tech_a, tech_b = random.sample(["ChromaDB", "Elasticsearch", "Milvus", "Redis", "PostgreSQL", "MongoDB", "Kafka", "RabbitMQ"], 2)

        subject = f"{proj} 技术选型：{tech_a} vs {tech_b}"
        body = f"""对 {proj} 项目做了 {tech_a} 和 {tech_b} 的对比测试：

| 方案 | P50延迟 | P95延迟 | 内存占用 | 部署复杂度 |
|------|---------|---------|----------|-----------|
| {tech_a} | {random.randint(30,80)}ms | {random.randint(80,200)}ms | {random.uniform(0.5,3):.1f}GB | {random.choice(['低','中'])} |
| {tech_b} | {random.randint(60,150)}ms | {random.randint(150,400)}ms | {random.uniform(1,6):.1f}GB | {random.choice(['中','高'])} |

结论：推荐 {tech_a}，理由是：轻量级、部署简单、10万文档量下延迟可接受。
如果后续数据量超过 50 万，可以考虑迁移到 {tech_b}。

详细测试代码见附件。"""
        level = "confidential" if random.random() < 0.10 else "internal"
        _write_mail(sender, receivers, [], subject, body, days, level)
    return count


def gen_project_mgmt_emails(count=20):
    """项目管理邮件。"""
    for i in range(count):
        proj = random.choice(list(PROJECTS.keys()))
        sender = _pick(["张伟", "陈静", "李娜"])
        receivers = _pick(ALL_NAMES, sender, random.randint(2, 5))
        days = random.randint(0, 180)
        ts = _ts(days)
        sprint_n = random.randint(5, 20)
        milestone = random.choice(["MVP", "v1.0", "v1.5", "v2.0"])

        if random.random() < 0.5:
            subject = f"{proj}项目 Sprint {sprint_n} 规划"
            body = f"各位，{proj}项目 Sprint {sprint_n} 规划如下：\n\n本Sprint目标：\n1. 完成 {random.choice(['权限模块','检索优化','文档解析','前端重构','API对接'])} 开发\n2. 修复上一轮压测发现的 {random.randint(3,8)} 个 Bug\n3. {random.choice(['通过客户验收','完成内部演示','部署到预发布环境'])}\n\n时间：{ts:%m月%d日} 至 {(ts+timedelta(days=14)):%m月%d日}\n风险：{random.choice(['人力不足','第三方依赖延期','需求变更'])}"
        else:
            subject = f"{proj}项目里程碑评审 —— {milestone}"
            body = f"各位，{proj}项目{milestone}里程碑评审结论：\n\n达成项：\n- {random.choice(['核心功能完成','客户UAT通过','性能达标','安全审计通过'])}\n\n延期项：\n- {random.choice(['扫描件OCR','知识图谱','多租户'])} 推迟到下一里程碑\n\n关键指标：\n- 代码覆盖率：{random.randint(72,92)}%\n- P95延迟：{random.randint(1800,4500)}ms\n\n下一里程碑：{(ts+timedelta(days=30)):%m月%d日}"
        _write_mail(sender, receivers, [random.choice(ALL_NAMES)], subject, body, days, "internal")
    return count


def gen_meeting_notes(count=15):
    """会议纪要。"""
    for i in range(count):
        proj = random.choice(list(PROJECTS.keys()))
        days = random.randint(0, 180)
        ts = _ts(days)
        attendees = random.sample(ALL_NAMES, random.randint(4, 7))

        type_name = random.choice(["周会", "技术评审", "Sprint回顾", "安全评审", "架构讨论", "需求对齐", "线上问题复盘"])
        decisions = [
            f"- 决定采用 {random.choice(['ChromaDB', 'PostgreSQL', 'Redis', 'gRPC'])} 作为 {random.choice(['检索引擎', '缓存层', '数据存储', '通信协议'])}",
            f"- 确认 {random.choice(['v1.2', 'v2.0', 'Q3版本'])} 的交付日期为 {(_ts(days) + timedelta(days=random.randint(14,60))).strftime('%m月%d日')}",
            f"- 分配 {random.choice(ALL_NAMES)} 负责 {random.choice(['性能优化', '安全加固', '文档完善', '客户对接'])}",
        ]
        todos = [
            f"- [{random.choice(ALL_NAMES)}] {random.choice(['完成技术方案', '修复线上问题', '准备演示材料', '更新API文档'])} —— {(_ts(days) + timedelta(days=random.randint(1,7))).strftime('%m月%d日')}",
            f"- [{random.choice(ALL_NAMES)}] {random.choice(['补充单元测试', '对接第三方接口', '性能压测', '代码review'])} —— {(_ts(days) + timedelta(days=random.randint(3,14))).strftime('%m月%d日')}",
            f"- [{random.choice(ALL_NAMES)}] {random.choice(['输出技术方案文档', '组织技术分享', '排查线上告警', '更新deploy脚本'])} —— {(_ts(days) + timedelta(days=random.randint(1,10))).strftime('%m月%d日')}",
        ]

        body = f"""# {proj}项目{type_name}纪要

日期：{ts:%Y-%m-%d %H:%M}  参会人：{'、'.join(attendees)}

## 讨论内容

1. {random.choice(['上周进展回顾', '技术方案讨论', '问题排查', '需求变更讨论'])}
2. {random.choice(['性能指标review', '安全风险评估', '排期调整', '资源协调'])}

## 决策

{chr(10).join(decisions)}

## 待办

{chr(10).join(todos)}

## 下次会议

{(_ts(days) + timedelta(days=7)).strftime('%m月%d日')} {random.choice(['14:00', '10:00', '16:00'])}"""
        _write_meeting(f"{ts:%Y%m%d}_{proj}_{type_name}.md", body)
    return count


def gen_contracts(count=12):
    """多种类型的企业合同。"""
    contract_types = [
        ("软件采购合同", "某科技有限公司", "TeamMind企业版 V2.0", random.randint(15, 80), "万"),
        ("技术服务合同", "某金融集团", "知识库检索系统定制开发", random.randint(30, 150), "万"),
        ("SaaS服务协议", "某电商平台", "TeamMind SaaS年度订阅", random.randint(8, 50), "万/年"),
        ("保密协议(NDA)", "某投资机构", "项目尽职调查", 0, ""),
        ("数据安全协议(DPA)", "某医疗集团", "患者数据脱敏与存储方案", 0, ""),
        ("人力外包合同", "某IT外包公司", "AI平台部测试人力外包", random.randint(20, 60), "万/年"),
        ("GPU采购合同", "某硬件代理商", "NVIDIA RTX 4060 8GB × 20张", random.randint(4, 6), "万"),
        ("知识产权转让协议", "某离职员工", "在职期间开发代码的IP归属确认", 0, ""),
        ("战略合作协议", "某AI芯片公司", "联合研发端侧推理方案", 0, ""),
        ("软件授权许可", "某教育机构", "TeamMind教育版 50用户授权", random.randint(5, 20), "万/年"),
        ("云服务采购合同", "某云厂商", "GPU云服务器年度租赁", random.randint(10, 40), "万/年"),
        ("员工竞业限制协议", "全体核心技术员工", "离职后2年内不得加入竞对", 0, ""),
    ]

    for ctype, party, desc, amount, unit in contract_types:
        ts = _ts(random.randint(10, 170))
        days = random.randint(10, 170)
        sign_date = _ts(days)
        expire_date = sign_date + timedelta(days=random.choice([365, 730, 1095]))

        body = f"""# {ctype}

合同编号：YF-{ts:%Y}-{random.randint(100,999)}
甲方：云帆科技有限公司
乙方：{party}
签订日期：{sign_date:%Y-%m-%d}
{'到期日期：' + expire_date.strftime('%Y-%m-%d') if amount > 0 else ''}

## 合同标的

{desc}

## 核心条款摘要

### 1. 服务范围
乙方为甲方提供{desc}，具体内容详见附件A《技术规格说明书》。

### 2. 交付标准
- 系统可用性 ≥ 99.5%
- 故障响应时间 ≤ 2小时（P0级别）
- 数据安全合规：符合《数据安全法》《个人信息保护法》要求

### 3. 费用与支付
{chr(10)}{'合同总金额：¥' + str(amount) + unit if amount > 0 else '本协议不涉及直接金钱交易'}
支付方式：签约后30% / 验收后50% / 质保期满后20%

### 4. 保密条款
双方承诺对合同内容及履行过程中获知的对方商业秘密保密。
保密期限：合同终止后3年。

### 5. 违约责任
- 逾期交付：每延迟一日，支付合同金额 0.1% 的违约金
- 数据泄露：乙方承担全部直接和间接损失
- 知识产权侵权：乙方承担全部法律责任

### 6. 争议解决
双方协商不成的，提交甲方所在地人民法院管辖。

## 风险提示
{random.choice(['注意：乙方提供的第三方组件授权需单独确认。', '注意：合同含自动续约条款，需提前90天书面通知终止。', '注意：验收标准中性能指标较严格，需技术团队确认可行性。'])}"""
        _write_contract(f"{ts:%Y%m%d}_{_safe(ctype)}_{_safe(party)}.md", body)

    return count


def gen_internal_docs(count=25):
    """内部技术文档。"""
    types = ["架构设计", "编码规范", "运维手册", "安全指南", "性能优化", "部署手册", "API文档", "新人onboarding"]
    for i in range(count):
        proj = random.choice(list(PROJECTS.keys()))
        author = random.choice(["李娜", "王磊", "周婷", "吴强"])
        ver_major, ver_minor = random.randint(0, 3), random.randint(0, 9)
        days = random.randint(0, 180)
        ts = _ts(days)
        doc_type = types[i % len(types)]
        fname = f"{proj}_{doc_type}_v{ver_major}.{ver_minor}.md"

        body = f"""# {proj} {doc_type}文档

版本：v{ver_major}.{ver_minor}  作者：{author}  日期：{ts:%Y-%m-%d}

## 概述
本文档描述{proj}项目的{doc_type}相关内容，供团队成员参考。

## 核心内容
- 技术栈：Python + FastAPI + LangGraph + ChromaDB + PyTorch
- 部署方式：Docker Compose 单机部署，K8s 集群部署（可选）
- 推理后端：支持云端API和本地量化模型双模式
- 安全要求：三级文档分级 + RBAC + 审计日志

## 关键配置
- Embedding: BGE-large-zh-v1.5 (1024维)
- LLM: Qwen3-8B W4A8 (本地) / DeepSeek API (云端)
- 检索: ChromaDB + BM25 + RRF + Cross-Encoder
- 安全: AccessGuard + 审计日志 + TLS加密

## 相关文档
- 系统架构设计文档
- 检索策略设计文档
- W4A8量化部署指南
- RAG系统评估标准"""
        level = "public" if "onboarding" in doc_type or "API文档" in doc_type else "internal"
        _write_doc(fname, body, level)
    return count


def gen_announcements(count=10):
    """部门公告。"""
    for i in range(count):
        sender = _pick(["张伟", "林芳"])
        receivers = [n for n in ALL_NAMES if n != sender]
        days = random.randint(0, 180)
        ts = _ts(days)

        items = [
            (f"【通知】{random.choice(['Q2', 'Q3', 'Q4', '本月'])}绩效考核安排",
             f"各位同事，{random.choice(['Q2', 'Q3', 'Q4'])}绩效考核时间安排如下：\n\n自评截止：{ts.strftime('%m月%d日')}\n主管评估：{(_ts(days) + timedelta(days=7)).strftime('%m月%d日')} 前\n结果沟通：{(_ts(days) + timedelta(days=14)).strftime('%m月%d日')} 前\n\n考核维度：业务成果(50%) + 技术能力(30%) + 团队协作(20%)\n\n如有疑问请联系HR。"),
            (f"【通知】{random.choice(['下周三', '本周五', '下周五'])}技术分享会 —— {random.choice(['《RAG系统从0到1》', '《W4A8量化实践》', '《企业级AI安全设计》', '《LangGraph Agent编排》'])}",
             f"大家好，{random.choice(['下周三', '本周五'])}下午 {random.choice(['14:00', '15:00', '16:00'])} 在 {random.choice(['3号会议室', '线上腾讯会议'])} 举行技术分享会。\n\n主题：{random.choice(['《RAG系统从0到1》', '《W4A8量化实践》', '《企业级AI安全设计》'])}\n主讲：{random.choice(['李娜', '王磊', '周婷'])}\n时长：约 {random.choice(['45', '60'])} 分钟\n\n欢迎所有感兴趣的同学参加。"),
            (f"【通知】{random.choice(['国庆', '春节', '劳动节'])}放假安排及值班表",
             f"各位，{random.choice(['国庆', '春节'])}假期安排如下：\n\n放假时间：{ts.strftime('%m月%d日')} 至 {(_ts(days) + timedelta(days=7)).strftime('%m月%d日')}\n值班安排：\n- {random.choice(['王磊', '吴强', '李娜'])} 负责线上问题应急（P0级别）\n- 其他问题节后处理\n\n请大家在放假前完成代码提交和文档更新，避免假期期间出现阻断性问题。"),
            (f"【通知】团队建设活动 —— {random.choice(['羽毛球比赛', '户外徒步', '桌游之夜', '密室逃脱'])}",
             f"大家好！为增进团队凝聚力，定于 {(_ts(days) + timedelta(days=random.randint(3,14))).strftime('%m月%d日')} 举行 {random.choice(['羽毛球比赛', '户外徒步', '桌游之夜'])} 活动。\n\n地点：{random.choice(['公司附近体育馆', '西山国家森林公园', '公司茶水间'])}\n时间：{random.choice(['14:00-17:00', '10:00-16:00'])}\n\n请大家提前安排好工作，欢迎携带家属参加。"),
        ]
        subject, body = random.choice(items)
        level = "internal"
        _write_mail(sender, receivers, [], subject, body, days, level)

    return count


def gen_tech_qa(count=15):
    """技术问答邮件。"""
    qa_pairs = [
        ("ChromaDB 的 HNSW 索引参数怎么调优？",
         "我现在 M=16, ef_construction=200, ef_search=100。10万条文档查询延迟 80ms，想优化到 50ms 以下。有什么建议吗？",
         "M 调到 32，ef_construction 调到 400，ef_search 降到 50。预期延迟降到 40-50ms。ef_search=100 太高了，这是延迟的主要来源。"),
        ("4-bit 量化后模型精度下降太多怎么办？",
         "用 bitsandbytes 做了 Qwen3-8B 4-bit，PPL 从 8.2 跳到 12.5。是不是配置有问题？",
         "12.5 不正常。bitsandbytes 是通用算法，精度损失大。建议用 LLMC 的 QuRot+LWC+GPTQ 方法加 WikiText 校准，PPL 损失可控制在 5% 左右。生产环境推荐 LLMC 离线量化。"),
        ("LangGraph 的 Reflect 循环怎么控制？",
         "检索不足时补搜，但有时无限循环。怎么控制？",
         "在 state 加 reflect_count，Judge 里检查。max_iterations=1 就够了。我们 50 个测试查询上 85% 的情况一次补搜就够了。"),
        ("Cross-Encoder 重排的显存够用吗？",
         "4060 8GB 加 bge-reranker-base 会 OOM 吗？",
         "不会。实测显存峰值 6.8GB，有 1.2GB 余量。如果不够，可以降级用轻量规则重排——基于内容长度和关键词重叠调整权重，零额外显存。"),
        ("FastAPI 异步模式下 LLM 调用会阻塞吗？",
         "`asyncio.to_thread` 和直接在 async handler 里调 `model.generate()` 有什么区别？",
         "直接用会阻塞事件循环，导致所有请求排队。用 `asyncio.to_thread` 把同步调用放到线程池，其他请求不受影响。注意线程安全。"),
    ]

    generated = 0
    for i in range(count):
        asker = _pick([n for n in ALL_NAMES if n not in ["张伟", "林芳"]])
        answerer = _pick(["李娜", "王磊", "周婷", "吴强"], asker)
        days = random.randint(0, 180)
        subject, question, answer = qa_pairs[i % len(qa_pairs)]
        _write_mail(asker, [answerer], [], subject, question, days, "internal")
        _write_mail(answerer, [asker], [], f"Re: {subject}", answer, days + 1, "internal")
        generated += 2
    return generated


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    print(f"生成企业演示数据 → {OUTPUT_DIR}")
    print("=" * 50)

    n = gen_daily_work_emails(50)
    total += n
    print(f"  日常工作邮件: {n} 封")

    n = gen_tech_discussion_emails(30)
    total += n
    print(f"  技术讨论邮件: {n} 封")

    n = gen_project_mgmt_emails(20)
    total += n
    print(f"  项目管理邮件: {n} 封")

    n = gen_tech_qa(15)
    total += n
    print(f"  技术问答邮件: {n} 封（含回复）")

    n = gen_meeting_notes(15)
    total += n
    print(f"  会议纪要: {n} 份")

    n = gen_contracts(12)
    total += n
    print(f"  合同文件: {n} 份")

    n = gen_internal_docs(25)
    total += n
    print(f"  内部技术文档: {n} 篇")

    n = gen_announcements(10)
    total += n
    print(f"  部门公告: {n} 封")

    print("=" * 50)
    print(f"总计: {total} 份文档")
    print(f"类型: 邮件+纪要+合同+文档+公告+问答")
    print(f"分级: public/internal/confidential")
    print(f"参加者: {len(TEAM)} 位团队成员")
    print(f"项目: {len(PROJECTS)} 个项目")
    print()
    print("运行以下命令索引数据:")
    print("  rm -r -Force data/chroma_db")
    print("  python scripts/run_demo.py --serve")


if __name__ == "__main__":
    main()
