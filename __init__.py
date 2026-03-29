"""
agents/__init__.py
BistroBrain agent registry.
"""
from agents.inventory_agent import InventoryAgent
from agents.insight_agent import InsightAgent
from agents.marketing_agent import MarketingAgent
from agents.pricing_agent import PricingAgent

__all__ = ["InventoryAgent", "InsightAgent", "MarketingAgent", "PricingAgent"]
