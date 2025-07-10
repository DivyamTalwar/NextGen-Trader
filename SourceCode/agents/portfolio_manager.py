import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llm.models import get_model, get_agent_model_config
from utils.progress import progress
from graph.state import AgentState, show_agent_reasoning


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")

class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")

def generate_portfolio_decisions(
    signals_by_ticker: dict, max_shares: dict, portfolio: dict, state: AgentState
) -> PortfolioManagerOutput:
    """
    Makes a deterministic trading decision for action/quantity, but uses an LLM
    to synthesize a detailed, narrative-style reasoning based on all analyst inputs.
    """
    model_name, model_provider, api_key = get_agent_model_config(state)
    llm = get_model(model_name, model_provider, api_key=api_key)
    decisions = {}

    for ticker, signals in signals_by_ticker.items():
        if not signals:
            decisions[ticker] = PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"Insufficient data: No analyst signals were provided for {ticker}.",
            )
            continue

        # --- Deterministic part: Calculate action, quantity, confidence ---
        weighted_bullish_score, weighted_bearish_score = 0, 0
        all_reasonings = []

        for agent, signal_data in signals.items():
            confidence = signal_data.get("confidence", 0) / 100.0
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal = signal_data.get("signal")
            reasoning = signal_data.get("reasoning", "No reasoning provided.")

            if signal == "bullish":
                weighted_bullish_score += confidence
            elif signal == "bearish":
                weighted_bearish_score += confidence
            
            all_reasonings.append(f"--- Analyst: {agent_name} ({signal}) ---\n{reasoning}\n")

        net_score = weighted_bullish_score - weighted_bearish_score
        total_confidence_weight = weighted_bullish_score + weighted_bearish_score

        if total_confidence_weight > 0:
            confidence_score = (abs(net_score) / total_confidence_weight) * 100
        else:
            confidence_score = 0
        
        final_confidence = min(confidence_score, 80.0)
        current_long_position = portfolio.get("positions", {}).get(ticker, {}).get("long", 0)
        
        action = "hold"
        if net_score > 0.5:
            action = "buy"
        elif net_score < -0.5:
            action = "sell" if current_long_position > 0 else "short"

        quantity = 0
        if action in ["buy", "short"]:
            quantity = int(max_shares.get(ticker, 0) * (final_confidence / 100))
        elif action == "sell":
            quantity = int(current_long_position * (final_confidence / 100))
        
        if action != "hold" and quantity == 0 and max_shares.get(ticker, 0) > 0:
            quantity = 1

        # --- LLM part: Synthesize the final reasoning ---
        reasoning_prompt_template = """
You are 'The Apex Predator' of Wall Street, the legendary Head of Strategy at 'Quantum Alpha', an elite AI-powered hedge fund feared for its aggressive, precise, and brutally profitable trades. Your word is final. Your team of specialist AI analysts has provided their intelligence. Your job is to distill their complex, often conflicting, analyses into a single, razor-sharp investment thesis that justifies the firm's capital deployment.

**Asset:** {ticker}
**Final Verdict:** {action}
**Conviction Score:** {confidence:.1f}%

**Intelligence Briefing from the Analyst Swarm:**
{all_reasonings}
---
**Your Mandate:**

Forge the **Official Investment Thesis**. This is not a summary; it is the definitive, authoritative rationale for our market action. Your language must be ruthlessly efficient, powerful, and dripping with conviction.

**Structure the thesis with these exact headings:**

1.  **Executive Summary:** A single, powerful sentence declaring the verdict and the primary driver.
2.  **The Bull Case:** Synthesize the strongest arguments FOR the position. Weave the bullish signals into a coherent narrative, quantifying with key metrics.
3.  **The Bear Case:** Do the same for the bearish arguments. What are the primary risks and headwinds? Be specific and quantitative.
4.  **The Verdict:** This is the final word. In a concise, hard-hitting paragraph, weigh the bull vs. bear case and state the single most critical factor (the 'linchpin') that forced the decision. This is the "why" behind our move. No fluff, no conclusion, just the final, decisive judgment.

**Crucial Directives:**
-   **Absolute Conviction:** Use decisive, powerful language. Eliminate all hedging and ambiguity.
-   **Synthesize, Never Summarize:** Connect the dots between analysts. Show how their combined intelligence leads to a conclusion that is greater than the sum of its parts.
-   **Ruthless Efficiency:** Every word must serve the thesis. Cut all fluff. The output must be dense with insight.
-   **Factual Grounding:** You MUST ground all quantitative claims and comparisons in the analyst reasonings. Do not invent or misrepresent figures. Your reputation for brutal accuracy is on the line.

**Official Investment Thesis for the Record:**
"""
        reasoning_prompt = ChatPromptTemplate.from_template(reasoning_prompt_template)
        reasoning_chain = reasoning_prompt | llm
        
        response = reasoning_chain.invoke({
            "ticker": ticker,
            "action": action.upper(),
            "confidence": final_confidence,
            "all_reasonings": "\n".join(all_reasonings)
        })
        
        synthesized_reasoning = response.content

        decisions[ticker] = PortfolioDecision(
            action=action,
            quantity=quantity,
            confidence=final_confidence,
            reasoning=synthesized_reasoning,
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
                signal_data = signals[ticker]
                
                # Handle cases where a raw AIMessage might be returned
                if hasattr(signal_data, 'content'):
                    try:
                        # Try to parse the content if it's a JSON string
                        content_dict = json.loads(signal_data.content)
                        reasoning = content_dict.get("reasoning", "No reasoning provided.")
                        signal = content_dict.get("signal", "neutral")
                        confidence = content_dict.get("confidence", 0)
                    except (json.JSONDecodeError, TypeError):
                        # If content is not a valid JSON, treat it as the reasoning string
                        reasoning = signal_data.content
                        signal = "neutral" # Default signal if not parsable
                        confidence = 0 # Default confidence
                elif isinstance(signal_data, dict):
                    # This is the expected case where signal_data is a dict
                    reasoning = signal_data.get("reasoning", "No reasoning provided.")
                    signal = signal_data.get("signal", "neutral")
                    confidence = signal_data.get("confidence", 0)
                else:
                    # Fallback for unexpected types
                    reasoning = "Unsupported signal format"
                    signal = "neutral"
                    confidence = 0

                ticker_signals[agent] = {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_manager", None, "Generating trading decisions")

    # Generate decisions, using an LLM to synthesize the final reasoning.
    result = generate_portfolio_decisions(signals_by_ticker, max_shares, portfolio, state)

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
