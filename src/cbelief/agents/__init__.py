# 功能说明：
# 1. 暴露 Agentic C-BELIEF 多智能体推理入口。
# 2. 该模块嵌入原有 src/cbelief/agents 架构，不另建独立包。

from .agentic_cbelief import AgenticCBeliefPipeline

__all__ = ["AgenticCBeliefPipeline"]
