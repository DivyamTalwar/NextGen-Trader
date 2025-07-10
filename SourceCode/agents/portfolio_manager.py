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
    Makes a deterministic trading decision based on aggregated analyst signals,
    weighted by confidence scores, and generates detailed, narrative-style reasoning.
    """
    decisions = {}
    for ticker, signals in signals_by_ticker.items():
        if not signals:
            decisions[ticker] = PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"Insufficient data: No analyst signals were provided for {ticker}."
            )
            continue

        weighted_bullish_score = 0
        weighted_bearish_score = 0
        bullish_analysts = []
        bearish_analysts = []

        for agent, signal_data in signals.items():
            confidence = signal_data.get('confidence', 0) / 100.0
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            if signal_data.get('signal') == 'bullish':
                weighted_bullish_score += confidence
                bullish_analysts.append(f"{agent_name} ({confidence:.0%})")
            elif signal_data.get('signal') == 'bearish':
                weighted_bearish_score += confidence
                bearish_analysts.append(f"{agent_name} ({confidence:.0%})")

        net_score = weighted_bullish_score - weighted_bearish_score
        total_confidence_weight = weighted_bullish_score + weighted_bearish_score

        # Calculate confidence based on the conviction of the weighted signals
        if total_confidence_weight > 0:
            # The confidence is the magnitude of the net score relative to the total conviction
            confidence_score = (abs(net_score) / total_confidence_weight) * 100
        else:
            confidence_score = 0
            
        # Cap the final confidence at 80%
        final_confidence = min(confidence_score, 80.0)

        current_long_position = portfolio.get("positions", {}).get(ticker, {}).get("long", 0)
        
        action = "hold"
        if net_score > 0.5:
            action = "buy"
        elif net_score < -0.5:
            action = "sell" if current_long_position > 0 else "short"

        # Generate detailed, narrative-style reasoning
        if action == "buy":
            reasoning = (f"A strong bullish consensus has emerged for {ticker}, driven by positive signals from {', '.join(bullish_analysts)}. "
                         f"The cumulative analysis, reflected in a net weighted score of {net_score:.2f}, points to a favorable risk/reward profile, justifying a new long position. "
                         "This decision is based on a confluence of factors, including positive sentiment and strong fundamentals.")
        elif action == "sell":
            reasoning = (f"A significant bearish sentiment from {', '.join(bearish_analysts)} has prompted a defensive SELL action for {ticker}. "
                         f"The analysis, with a net weighted score of {net_score:.2f}, suggests deteriorating fundamentals or unfavorable market conditions, warranting an exit from the current long position. "
                         "This move is designed to preserve capital and mitigate downside risk.")
        elif action == "short":
            reasoning = (f"A compelling bearish case for {ticker} has been presented by {', '.join(bearish_analysts)}. "
                         f"The net weighted score of {net_score:.2f} indicates a high-conviction shorting opportunity, based on the collective intelligence of the analyst swarm. "
                         "This action is taken to capitalize on the anticipated downward price movement.")
        else:
            reasoning = (f"The analyst swarm holds a neutral stance on {ticker}, with a net weighted score of {net_score:.2f}. "
                         f"The signals are mixed, with bullish conviction from {', '.join(bullish_analysts) if bullish_analysts else 'None'} "
                         f"and bearish sentiment from {', '.join(bearish_analysts) if bearish_analysts else 'None'}. "
                         "Holding the current position is the most prudent action until a clearer consensus emerges, avoiding unnecessary risk in a divided market.")

        # Calculate quantity
        quantity = 0
        if action in ["buy", "short"]:
            quantity = int(max_shares.get(ticker, 0) * (final_confidence / 100))
        elif action == "sell":
            quantity = int(current_long_position * (final_confidence / 100))
        
        if action != "hold" and quantity == 0 and max_shares.get(ticker, 0) > 0:
            quantity = 1

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
