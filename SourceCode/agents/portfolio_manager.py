import json
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from graph.state import AgentState, show_agent_reasoning

class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")

class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")

def make_deterministic_decision(signals_by_ticker: dict, max_shares: dict, portfolio: dict) -> PortfolioManagerOutput:
    """
    Makes a deterministic trading decision based on aggregated analyst signals.
    """
    decisions = {}
    for ticker, signals in signals_by_ticker.items():
        if not signals:
            decisions[ticker] = PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"No signals available for {ticker}, defaulting to hold."
            )
            continue

        bullish_confidences = []
        bearish_confidences = []
        
        for agent, signal_data in signals.items():
            if signal_data.get('signal') == 'bullish':
                bullish_confidences.append(signal_data.get('confidence', 0))
            elif signal_data.get('signal') == 'bearish':
                bearish_confidences.append(signal_data.get('confidence', 0))

        # New logic: Sum of confidences
        total_bullish_score = sum(bullish_confidences)
        total_bearish_score = sum(bearish_confidences)
        
        net_score = total_bullish_score - total_bearish_score

        # Determine action based on net score
        action = "hold"
        reasoning = f"Mixed signals for {ticker}. Net score: {net_score:.2f}. Holding position."
        
        # Use the higher of the two scores for final confidence
        final_confidence = 0
        if total_bullish_score > total_bearish_score:
            final_confidence = (total_bullish_score / (total_bullish_score + total_bearish_score)) * 100 if (total_bullish_score + total_bearish_score) > 0 else 0
        elif total_bearish_score > total_bullish_score:
            final_confidence = (total_bearish_score / (total_bullish_score + total_bearish_score)) * 100 if (total_bullish_score + total_bearish_score) > 0 else 0
        
        current_long_position = portfolio.get("positions", {}).get(ticker, {}).get("long", 0)

        # More decisive thresholds
        if net_score > 50:  # Threshold for bullish signal
            action = "buy"
            reasoning = f"Strong bullish consensus for {ticker}. Total bullish score: {total_bullish_score:.2f} vs bearish: {total_bearish_score:.2f}."
        elif net_score < -50: # Threshold for bearish signal
            if current_long_position > 0:
                action = "sell"
                reasoning = f"Strong bearish consensus for {ticker}. Total bearish score: {total_bearish_score:.2f} vs bullish: {total_bullish_score:.2f}. Selling existing position."
            else:
                action = "short"
                reasoning = f"Strong bearish consensus for {ticker}. Total bearish score: {total_bearish_score:.2f} vs bullish: {total_bullish_score:.2f}. Initiating short position."

        # Calculate quantity
        quantity = 0
        if action in ["buy", "short"]:
            # Scale quantity by confidence, up to max_shares
            quantity = int(max_shares.get(ticker, 0) * (final_confidence / 100))
        elif action == "sell":
            # Sell a portion of the position based on confidence
            quantity = int(current_long_position * (final_confidence / 100))
        
        # Ensure quantity is not zero if action is not 'hold'
        if action != "hold" and quantity == 0 and max_shares.get(ticker, 0) > 0:
            quantity = 1 # trade at least one share if a decision is made

        decisions[ticker] = PortfolioDecision(
            action=action,
            quantity=quantity,
            confidence=final_confidence,
            reasoning=reasoning
        )
        
    return PortfolioManagerOutput(decisions=decisions)

def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_manager", ticker, "Processing analyst signals")

        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        if current_prices.get(ticker, 0) > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_manager", None, "Generating trading decisions")

    # Use the deterministic function instead of an LLM call
    result = make_deterministic_decision(signals_by_ticker, max_shares, portfolio)

    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_manager",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status("portfolio_manager", None, "Done")

    data_update = {
        "final_decisions": {
            ticker: decision.model_dump() for ticker, decision in result.decisions.items()
        }
    }

    return {
        "messages": [message],
        "data": data_update,
    }
