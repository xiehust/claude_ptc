#!/usr/bin/env python3
"""
Programmatic Tool Calling 基础示例

演示如何使用自定义沙箱实现 Programmatic Tool Calling
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandboxed_ptc import ProgrammaticToolOrchestrator, ToolCallerType
from sandboxed_ptc.orchestrator import OrchestratorConfig
from sandboxed_ptc.sandbox import SandboxConfig


# ============================================================
# 模拟数据库和 API（实际使用时替换为真实实现）
# ============================================================

# 模拟销售数据
MOCK_SALES_DATA = {
    "East": [
        {"date": "2024-01-15", "product": "Widget A", "revenue": 15000, "units": 150},
        {"date": "2024-01-20", "product": "Widget B", "revenue": 22000, "units": 110},
        {"date": "2024-02-01", "product": "Widget A", "revenue": 18000, "units": 180},
    ],
    "West": [
        {"date": "2024-01-10", "product": "Widget A", "revenue": 25000, "units": 250},
        {"date": "2024-01-25", "product": "Widget C", "revenue": 30000, "units": 100},
        {"date": "2024-02-05", "product": "Widget B", "revenue": 12000, "units": 60},
    ],
    "Central": [
        {"date": "2024-01-12", "product": "Widget B", "revenue": 45000, "units": 225},
        {"date": "2024-01-28", "product": "Widget A", "revenue": 38000, "units": 380},
        {"date": "2024-02-03", "product": "Widget C", "revenue": 52000, "units": 173},
    ],
}

MOCK_SERVERS = {
    "us-east-1": {"status": "degraded", "cpu": 85, "memory": 72},
    "us-west-2": {"status": "healthy", "cpu": 45, "memory": 55},
    "eu-west-1": {"status": "healthy", "cpu": 38, "memory": 48},
    "ap-south-1": {"status": "offline", "cpu": 0, "memory": 0},
}


# ============================================================
# 初始化 Orchestrator
# ============================================================

def create_orchestrator() -> ProgrammaticToolOrchestrator:
    """创建并配置 Orchestrator"""
    # 配置
    config = OrchestratorConfig(
        model="global.anthropic.claude-opus-4-5-20251101-v1:0",
        max_tokens=4096,
        max_iterations=10,
        sandbox_config=SandboxConfig(
            memory_limit="256m",
            timeout_seconds=60.0,
            network_disabled=True,
        )
    )

    orchestrator = ProgrammaticToolOrchestrator(
        config=config
    )

    # ========================================================
    # 注册工具
    # ========================================================

    @orchestrator.register_tool(
        description="查询销售数据库。返回指定区域的销售记录列表。",
        output_description="返回 list[dict]，每个 dict 包含 date, product, revenue, units 字段",
        allowed_callers=[ToolCallerType.CODE_EXECUTION]
    )
    def query_sales(region: str, start_date: str = None, end_date: str = None) -> list[dict]:
        """
        查询销售数据

        Args:
            region: 区域名称 (East, West, Central)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        data = MOCK_SALES_DATA.get(region, [])

        # 日期过滤
        if start_date:
            data = [d for d in data if d["date"] >= start_date]
        if end_date:
            data = [d for d in data if d["date"] <= end_date]

        return data

    @orchestrator.register_tool(
        description="获取服务器健康状态",
        output_description="返回 dict，包含 status (healthy/degraded/offline), cpu (%), memory (%)",
        allowed_callers=[ToolCallerType.CODE_EXECUTION]
    )
    def check_server_health(server_id: str) -> dict:
        """
        检查服务器健康状态

        Args:
            server_id: 服务器 ID
        """
        return MOCK_SERVERS.get(server_id, {"status": "unknown", "cpu": 0, "memory": 0})

    @orchestrator.register_tool(
        description="获取所有服务器 ID 列表",
        output_description="返回 list[str]，服务器 ID 列表",
        allowed_callers=[ToolCallerType.CODE_EXECUTION]
    )
    def list_servers() -> list[str]:
        """列出所有服务器"""
        return list(MOCK_SERVERS.keys())

    @orchestrator.register_tool(
        description="获取产品信息",
        output_description="返回 dict，包含 name, price, category",
        allowed_callers=[ToolCallerType.CODE_EXECUTION]
    )
    def get_product_info(product_name: str) -> dict:
        """获取产品详情"""
        products = {
            "Widget A": {"name": "Widget A", "price": 100, "category": "Electronics"},
            "Widget B": {"name": "Widget B", "price": 200, "category": "Electronics"},
            "Widget C": {"name": "Widget C", "price": 300, "category": "Premium"},
        }
        return products.get(product_name, {"name": "Unknown", "price": 0, "category": "Unknown"})

    return orchestrator


# ============================================================
# 示例运行
# ============================================================

async def example_sales_analysis():
    """示例：销售数据分析"""
    print("\n" + "=" * 60)
    print("示例 1: 销售数据分析")
    print("=" * 60)

    orchestrator = create_orchestrator()

    result = await orchestrator.run(
        "分析 East、West、Central 三个区域的销售数据，"
        "告诉我哪个区域的总收入最高，以及各区域的产品销售分布情况。"
    )

    print("\n Claude 回复:")
    print(result)


async def example_server_monitoring():
    """示例：服务器监控"""
    print("\n" + "=" * 60)
    print("示例 2: 服务器健康检查")
    print("=" * 60)

    orchestrator = create_orchestrator()

    result = await orchestrator.run(
        "检查所有服务器的健康状态，找出有问题的服务器并生成报告。"
        "如果发现健康的服务器就报告找到了可用服务器。"
    )

    print("\n Claude 回复:")
    print(result)


async def example_complex_query():
    """示例：复杂查询"""
    print("\n" + "=" * 60)
    print("示例 3: 复杂数据处理")
    print("=" * 60)

    orchestrator = create_orchestrator()

    result = await orchestrator.run(
        "我需要一个综合报告：\n"
        "1. 查询所有区域的销售数据\n"
        "2. 计算每个产品的总销售额和销售量\n"
        "3. 获取每个产品的详细信息（价格、类别）\n"
        "4. 分析哪个产品类别表现最好\n"
        "请用表格形式展示结果。"
    )

    print("\n Claude 回复:")
    print(result)


async def example_streaming():
    """示例：流式输出"""
    print("\n" + "=" * 60)
    print("示例 4: 流式响应")
    print("=" * 60)

    orchestrator = create_orchestrator()

    print("\nClaude 回复 (流式):")
    async for chunk in orchestrator.run_streaming(
        "简要分析一下 Central 区域的销售情况。"
    ):
        print(chunk, end="", flush=True)
    print()


async def main():
    """主函数"""
    print("Programmatic Tool Calling 示例")
    print("使用自定义沙箱实现")

    # 运行示例
    try:
        # await example_sales_analysis()
        # await example_server_monitoring()
        # await example_complex_query()
        await example_streaming()  # 取消注释以测试流式输出
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
