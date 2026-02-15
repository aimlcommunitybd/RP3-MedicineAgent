"""
OpenRouter Usage Tracker
Tracks model usage, tokens, and costs for all LLM calls.
"""

import os
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

# Cost per 1M tokens (approximate - update as needed)
MODEL_COSTS = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Qwen
    "qwen/qwen3-8b": {"input": 0.20, "output": 0.40},
    "qwen/qwen3-4b": {"input": 0.10, "output": 0.20},
    "qwen/qwen2.5-7b": {"input": 0.20, "output": 0.40},
    # Meta Llama
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.20, "output": 0.20},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.90, "output": 0.90},
    # Mistral
    "mistralai/mistral-7b-instruct": {"input": 0.20, "output": 0.20},
    # Anthropic
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-opus": {"input": 15.00, "output": 75.00},
}

# Default cost for unknown models
DEFAULT_COST = {"input": 0.50, "output": 0.50}


@dataclass
class UsageRecord:
    """Record of a single LLM call"""

    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    duration_ms: int


@dataclass
class UsageTracker:
    """Tracks all LLM usage"""

    records: list[UsageRecord] = field(default_factory=list)
    start_time: Optional[str] = None

    def add_record(
        self, model: str, input_tokens: int, output_tokens: int, duration_ms: int = 0
    ):
        """Add a usage record"""
        # Get model name without provider prefix
        model_name = model.split("/")[-1] if "/" in model else model

        # Get costs
        costs = MODEL_COSTS.get(model_name, MODEL_COSTS.get(model, DEFAULT_COST))

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            duration_ms=duration_ms,
        )
        self.records.append(record)

        return record

    def get_summary(self) -> dict:
        """Get usage summary"""
        if not self.records:
            return {"total_calls": 0, "total_cost": 0, "total_tokens": 0}

        total_calls = len(self.records)
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        total_tokens = total_input + total_output
        total_cost = sum(r.total_cost for r in self.records)

        # Group by model
        by_model = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0,
                }
            by_model[r.model]["calls"] += 1
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost"] += r.total_cost

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "by_model": by_model,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 OPENROUTER USAGE SUMMARY")
        print("=" * 60)

        if summary["total_calls"] == 0:
            print("No LLM calls recorded yet.")
            return

        print(f"\n📅 Session: {summary['start_time']} → {summary['end_time']}")
        print(f"\n💰 Total Cost: ${summary['total_cost']:.6f}")
        print(f"🔢 Total Calls: {summary['total_calls']}")
        print(f"📝 Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"📝 Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"📝 Total Tokens: {summary['total_tokens']:,}")

        print("\n" + "-" * 60)
        print("📱 BY MODEL:")
        print("-" * 60)

        for model, stats in summary["by_model"].items():
            print(f"\n🤖 {model}")
            print(f"   Calls: {stats['calls']}")
            print(f"   Input: {stats['input_tokens']:,} tokens")
            print(f"   Output: {stats['output_tokens']:,} tokens")
            print(f"   Cost: ${stats['cost']:.6f}")

        print("\n" + "=" * 60)

    def reset(self):
        """Reset tracker"""
        self.records = []
        self.start_time = datetime.now().isoformat()


# Global tracker instance
_global_tracker = UsageTracker()


def get_tracker() -> UsageTracker:
    """Get the global tracker instance"""
    if not _global_tracker.start_time:
        _global_tracker.start_time = datetime.now().isoformat()
    return _global_tracker


def reset_tracker():
    """Reset the global tracker"""
    _global_tracker.reset()
