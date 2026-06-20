"""
main.py
-------
FinDKG + Claude 资产配置建议系统 — 端到端入口

用法：
  python main.py                                    # 交互式命令行
  python main.py --entity "Apple Inc."              # 单实体快速查询
  python main.py --compare "Apple Inc." "Microsoft Corporation"
  python main.py --search "tesla"                   # 模糊搜索实体名
  python main.py --backtest "Apple Inc." --date 2021-06-06   # 单点回测
  python main.py --rolling "Apple Inc."             # 滚动回测
  python main.py --history                          # 查看历史建议记录
  python main.py --report                           # 信号准确率报告

Multi-Agent 模式（需设置 FRED_API_KEY）：
  python main.py --multi-agent --entity "Apple Inc."
  python main.py --multi-agent --compare "Apple Inc." "Microsoft Corporation"
"""

import argparse
import sys
import os


def _check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("⚠️  未设置 ANTHROPIC_API_KEY 环境变量")
        print("    请运行：export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)
    return key


def _init_components(with_llm: bool = True):
    """初始化所有组件（KG图、预测器、反馈存储、LLM）"""
    from kg_query import FinDKGGraph
    from kg_predictor import KGPredictor
    from feedback_store import FeedbackStore

    graph = FinDKGGraph()
    predictor = KGPredictor()          # 无 checkpoint 时自动禁用
    store = FeedbackStore()

    advisor = None
    if with_llm:
        _check_api_key()
        from llm_advisor import AssetAdvisor
        advisor = AssetAdvisor(predictor=predictor, feedback_store=store)

    return graph, predictor, store, advisor


# ── 各命令实现 ────────────────────────────────────────────────────

def cmd_search(graph, keyword: str):
    results = graph.fuzzy_search(keyword, top_k=20)
    if not results:
        print(f"未找到包含 '{keyword}' 的实体")
    else:
        print(f"找到 {len(results)} 个相关实体：")
        for r in results:
            print(f"  - {r}")


def cmd_query(graph, entity: str, weeks: int):
    try:
        df = graph.query_entity(entity, n_recent_weeks=weeks)
        print(f"\n【{entity}】最近 {weeks} 周 KG 事件（共 {len(df)} 条）：\n")
        print(df.head(20).to_string(index=False) if not df.empty else "  无数据")
    except ValueError as e:
        print(f"错误：{e}")


def cmd_advise(advisor, graph, entity: str, weeks: int, question: str = None):
    try:
        session_id, advice = advisor.advise(entity, graph, n_recent_weeks=weeks, user_question=question)
        print(f"\n{'='*60}\n  资产配置建议：{entity}\n{'='*60}\n")
        print(advice)

        # 交互式评分
        if session_id:
            try:
                raw = input("\n对这次建议评分？(+1/0/-1，回车跳过): ").strip()
                if raw in ("+1", "1"):
                    note = input("备注（可选）：").strip()
                    advisor.feedback(session_id, 1, note)
                elif raw == "0":
                    advisor.feedback(session_id, 0)
                elif raw == "-1":
                    note = input("备注（可选）：").strip()
                    advisor.feedback(session_id, -1, note)
            except (KeyboardInterrupt, EOFError):
                pass
    except ValueError as e:
        print(f"错误：{e}")


def cmd_compare(advisor, graph, entities: list[str], weeks: int):
    analysis = advisor.advise_comparison(entities, graph, n_recent_weeks=weeks)
    print(f"\n{'='*60}\n  多资产对比分析\n{'='*60}\n")
    print(analysis)


def cmd_backtest(graph, entity: str, date: str, weeks: int, forward: int):
    from backtester import Backtester
    bt = Backtester(graph)
    try:
        result = bt.run(entity, decision_date=date,
                        n_recent_weeks=weeks, n_forward_weeks=forward)
        print(f"\n{'='*60}\n  回测结果\n{'='*60}\n")
        print(result.summary())
        if result.detail:
            print(f"\n后续 KG 事件（前10条）：")
            for d in result.detail[:10]:
                print(f"  [{d['date']}] {d['subject']} --[{d['relation']}]--> {d['object']}")
    except ValueError as e:
        print(f"错误：{e}")


def cmd_rolling(graph, entity: str, weeks: int, forward: int, n_windows: int):
    from backtester import Backtester
    bt = Backtester(graph)
    try:
        bt.rolling_backtest(entity, n_windows=n_windows,
                            n_recent_weeks=weeks, n_forward_weeks=forward)
    except ValueError as e:
        print(f"错误：{e}")


def cmd_history(store, entity: str = None, limit: int = 10):
    store.print_history(entity=entity, limit=limit)


def cmd_report(store):
    import json
    report = store.signal_accuracy_report()
    print(f"\n{'='*60}\n  信号准确率报告\n{'='*60}\n")
    if report["total_rated"] == 0:
        print("暂无评分记录。先使用 advise 命令并对建议评分后再查看报告。")
        return
    print(f"已评分建议数：{report['total_rated']}")
    print(f"用户正面评价率：{report['positive_rate']:.1%}")
    print("\n各信号类型平均评分：")
    for rel, stat in sorted(report["signal_stats"].items(),
                            key=lambda x: x[1]["avg_rating"], reverse=True):
        bar = "+" * max(0, int(stat["avg_rating"] * 10)) or "-"
        print(f"  {rel:<30} avg={stat['avg_rating']:+.3f}  n={stat['count']}  {bar}")


# ── 交互模式 ─────────────────────────────────────────────────────

def interactive_mode(graph, advisor, store):
    print("\n" + "="*60)
    print("  FinDKG 资产配置建议系统（交互模式）")
    print("="*60)
    print("命令：")
    print("  search <关键词>              模糊搜索实体")
    print("  query <实体名>               查看 KG 原始数据")
    print("  advise <实体名>              获取配置建议（含评分）")
    print("  compare <实体1> | <实体2>    多资产对比")
    print("  backtest <实体名> <日期>     单点回测（如 backtest 'Apple Inc.' 2021-06-06）")
    print("  rolling <实体名>             滚动回测")
    print("  history [实体名]             查看历史建议记录")
    print("  report                       信号准确率报告")
    print("  stats                        数据集统计")
    print("  quit                         退出")
    print()

    while True:
        try:
            raw = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("退出。")
            break

        elif cmd == "stats":
            import json
            print(json.dumps(graph.stats(), ensure_ascii=False, indent=2))

        elif cmd == "search":
            cmd_search(graph, arg) if arg else print("用法：search <关键词>")

        elif cmd == "query":
            cmd_query(graph, arg, 12) if arg else print("用法：query <实体名>")

        elif cmd == "advise":
            if not arg:
                print("用法：advise <实体名>")
            else:
                custom_q = input("  自定义问题（直接回车跳过）：").strip() or None
                cmd_advise(advisor, graph, arg, 12, custom_q)

        elif cmd == "compare":
            entities = [e.strip() for e in arg.split("|") if e.strip()]
            if len(entities) < 2:
                print("用法：compare <实体1> | <实体2> [| <实体3>]")
            else:
                cmd_compare(advisor, graph, entities, 12)

        elif cmd == "backtest":
            sub_parts = arg.split() if arg else []
            if len(sub_parts) < 2:
                print("用法：backtest <实体名> <日期 YYYY-MM-DD>")
            else:
                # 最后一个词当日期，其余为实体名
                date = sub_parts[-1]
                entity = " ".join(sub_parts[:-1]).strip("'\"")
                cmd_backtest(graph, entity, date, 12, 8)

        elif cmd == "rolling":
            entity = arg.strip("'\"")
            cmd_rolling(graph, entity, 12, 8, 10) if entity else print("用法：rolling <实体名>")

        elif cmd == "history":
            entity = arg.strip("'\"") or None
            cmd_history(store, entity)

        elif cmd == "report":
            cmd_report(store)

        else:
            print(f"未知命令：{cmd}")


# ── CLI 入口 ─────────────────────────────────────────────────────

def cmd_multi_agent(entity: str, weeks: int):
    """Multi-Agent 模式：Macro Agent + News Agent → Orchestrator → 报告"""
    _check_api_key()

    from kg_query import FinDKGGraph
    from kg_predictor import KGPredictor
    from llm_advisor import AssetAdvisor
    from macro_agent import MacroAgent
    from orchestrator import OrchestratorAgent

    graph       = FinDKGGraph()
    predictor   = KGPredictor()
    news_agent  = AssetAdvisor(predictor=predictor)
    macro_agent = MacroAgent()
    orch        = OrchestratorAgent()

    print(f"\n{'='*60}")
    print(f"  Multi-Agent 投资建议报告：{entity}")
    print(f"{'='*60}\n")

    news_signal  = news_agent.get_news_signal(entity, graph, n_recent_weeks=weeks)
    macro_signal = macro_agent.analyze()
    report       = orch.generate_report(entity, news_signal, macro_signal)

    print("\n" + report)


def cmd_multi_agent_compare(entities: list[str], weeks: int):
    """Multi-Agent 多实体对比报告"""
    _check_api_key()

    from kg_query import FinDKGGraph
    from kg_predictor import KGPredictor
    from llm_advisor import AssetAdvisor
    from macro_agent import MacroAgent
    from orchestrator import OrchestratorAgent

    graph       = FinDKGGraph()
    predictor   = KGPredictor()
    news_agent  = AssetAdvisor(predictor=predictor)
    macro_agent = MacroAgent()
    orch        = OrchestratorAgent()

    print(f"\n{'='*60}")
    print(f"  Multi-Agent 多实体对比报告")
    print(f"{'='*60}\n")

    news_signals = [
        news_agent.get_news_signal(e, graph, n_recent_weeks=weeks)
        for e in entities
    ]
    macro_signal = macro_agent.analyze()
    report = orch.generate_comparison_report(entities, news_signals, macro_signal)

    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="FinDKG + Claude 资产配置建议系统")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--compare", type=str, nargs="+")
    parser.add_argument("--search", type=str)
    parser.add_argument("--query-only", action="store_true", help="只查 KG，不调用 LLM")
    parser.add_argument("--backtest", type=str, metavar="ENTITY")
    parser.add_argument("--rolling", type=str, metavar="ENTITY")
    parser.add_argument("--date", type=str, default=None, help="回测决策时间点 YYYY-MM-DD")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--forward", type=int, default=8, help="回测验证周数")
    parser.add_argument("--windows", type=int, default=10, help="滚动回测窗口数")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--multi-agent", action="store_true",
                        help="启用 Multi-Agent 模式（需设置 FRED_API_KEY）")
    args = parser.parse_args()

    # ── Multi-Agent 模式 ──────────────────────────────────────────
    if args.multi_agent:
        if args.entity:
            cmd_multi_agent(args.entity, args.weeks)
        elif args.compare:
            cmd_multi_agent_compare(args.compare, args.weeks)
        else:
            print("Multi-Agent 模式用法：")
            print("  python main.py --multi-agent --entity \"Apple Inc.\"")
            print("  python main.py --multi-agent --compare \"Apple Inc.\" \"Microsoft Corporation\"")
        return

    # ── 无 LLM 的操作 ─────────────────────────────────────────────
    no_llm_mode = (
        args.search or
        (args.query_only and args.entity) or
        args.backtest or
        args.rolling or
        args.history or
        args.report
    )

    if no_llm_mode:
        from kg_query import FinDKGGraph
        from feedback_store import FeedbackStore
        graph = FinDKGGraph(data_dir=args.data_dir)
        store = FeedbackStore()

        if args.search:
            cmd_search(graph, args.search)
        elif args.query_only and args.entity:
            cmd_query(graph, args.entity, args.weeks)
        elif args.backtest:
            if not args.date:
                print("错误：--backtest 需要配合 --date YYYY-MM-DD")
                sys.exit(1)
            cmd_backtest(graph, args.backtest, args.date, args.weeks, args.forward)
        elif args.rolling:
            cmd_rolling(graph, args.rolling, args.weeks, args.forward, args.windows)
        elif args.history:
            cmd_history(store, entity=args.entity)
        elif args.report:
            cmd_report(store)
        return

    # ── 需要 LLM 的操作 ───────────────────────────────────────────
    graph, predictor, store, advisor = _init_components(with_llm=True)

    if args.entity:
        cmd_advise(advisor, graph, args.entity, args.weeks)
    elif args.compare:
        cmd_compare(advisor, graph, args.compare, args.weeks)
    else:
        interactive_mode(graph, advisor, store)


if __name__ == "__main__":
    main()
