"""
main.py
-------
Multi-Agent 投资建议报告系统 — 命令行入口

用法：
  python main.py --entity "Apple Inc."                          # 单家公司报告（生成后可评分）
  python main.py --entity NVDA                                  # 也可以直接传 ticker
  python main.py --compare "Apple Inc." "Microsoft Corporation" # 多家公司对比
  python main.py --history                                      # 查看历史报告记录
  python main.py --history --entity "Apple Inc."                # 只看某家公司

环境变量：
  ANTHROPIC_API_KEY      必需
  SEC_EDGAR_USER_AGENT   推荐（SEC 申报数据）
  FRED_API_KEY           推荐（宏观数据）

公司名需在 tools/ticker_map.py 的映射表中，否则请直接传 ticker。
"""

import argparse
import sys

import config


def _check_api_key():
    if not config.get_anthropic_api_key():
        print("⚠️  未设置 ANTHROPIC_API_KEY 环境变量")
        print("    请运行：export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)


def _prompt_for_rating(store, session_id):
    """交互式评分：+1 / 0 / -1，回车跳过"""
    if not session_id:
        return
    try:
        raw = input("\n对这份报告评分？(+1/0/-1，回车跳过): ").strip()
        if raw in ("+1", "1"):
            store.rate(session_id, 1, input("备注（可选）：").strip())
        elif raw == "0":
            store.rate(session_id, 0, "")
        elif raw == "-1":
            store.rate(session_id, -1, input("备注（可选）：").strip())
    except (KeyboardInterrupt, EOFError):
        pass


def cmd_report(entity: str):
    _check_api_key()
    from feedback_store import FeedbackStore
    from orchestrator import OrchestratorAgent

    store = FeedbackStore()
    orch = OrchestratorAgent()

    print(f"\n{'='*60}\n  Multi-Agent 投资建议报告：{entity}\n{'='*60}\n")
    session_id, report = orch.generate_report(entity, feedback_store=store)
    print("\n" + report)
    _prompt_for_rating(store, session_id)


def cmd_compare(entities: list[str]):
    _check_api_key()
    from feedback_store import FeedbackStore
    from orchestrator import OrchestratorAgent

    print(f"\n{'='*60}\n  Multi-Agent 多公司对比报告\n{'='*60}\n")
    report = OrchestratorAgent().generate_comparison_report(entities, feedback_store=FeedbackStore())
    print("\n" + report)


def cmd_history(entity: str | None):
    from feedback_store import FeedbackStore
    FeedbackStore().print_history(entity=entity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Agent 投资建议报告系统（实时数据 + 知识图谱）")
    parser.add_argument("--entity", type=str, help="公司名或 ticker")
    parser.add_argument("--compare", type=str, nargs="+", metavar="ENTITY", help="多家公司对比")
    parser.add_argument("--history", action="store_true", help="查看历史报告记录（可配合 --entity 过滤）")
    # 兼容旧命令 python main.py --multi-agent --entity ...（现在只有这一种模式）
    parser.add_argument("--multi-agent", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.history:
        cmd_history(args.entity)
    elif args.compare:
        cmd_compare(args.compare)
    elif args.entity:
        cmd_report(args.entity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
