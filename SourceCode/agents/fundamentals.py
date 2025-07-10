from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics

def _analyze_metric_group(
    metrics_obj, 
    metric_definitions: list[tuple], 
    higher_is_better: bool = True
) -> dict:
    """
    Analyzes a group of metrics against their thresholds to generate a score and signal.

    Args:
        metrics_obj: The Pydantic model object containing the financial metrics.
        metric_definitions: A list of tuples, each being (metric_name, threshold).
        higher_is_better: If True, score increases if metric > threshold.
                          If False (for valuation), score increases if metric < threshold.

    Returns:
        A dictionary containing the score, signal, and a details string.
    """
    score = 0
    details_list = []

    for metric_name, threshold in metric_definitions:
        metric_value = getattr(metrics_obj, metric_name, None)

        if metric_value is not None:
            # The core scoring logic
            if (higher_is_better and metric_value > threshold) or \
               (not higher_is_better and metric_value < threshold):
                score += 1
            
            # Cleanly format the detail string for this metric
            display_name = metric_name.replace('_', ' ').title()
            if isinstance(metric_value, float) and abs(metric_value) < 1.5 and metric_name.endswith('growth'):
                details_list.append(f"{display_name}: {metric_value:.2%}")
            else:
                details_list.append(f"{display_name}: {metric_value:.2f}")
        else:
            details_list.append(f"{metric_name.replace('_', ' ').title()}: N/A")
            
    # Determine the signal based on the score
    # A score of 2 or 3 is bullish, 1 is neutral, 0 is bearish.
    if score >= 2:
        signal = "bullish" if higher_is_better else "bearish"
    elif score == 0:
        signal = "bearish" if higher_is_better else "bullish"
    else: # score == 1
        signal = "neutral"
        
    return {
        "score": score,
        "signal": signal,
        "details": ", ".join(details_list)
    }
    
    

##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_analyst_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_analyst_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing profitability")
        profitability_defs = [("return_on_equity", 0.15), ("net_margin", 0.20), ("operating_margin", 0.15)]
        profitability_result = _analyze_metric_group(metrics, profitability_defs)
        signals.append(profitability_result['signal'])
        reasoning["profitability_signal"] = profitability_result

        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing growth")
        growth_defs = [("revenue_growth", 0.10), ("earnings_growth", 0.10), ("book_value_growth", 0.10)]
        growth_result = _analyze_metric_group(metrics, growth_defs)
        signals.append(growth_result['signal'])
        reasoning["growth_signal"] = growth_result
        
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:  # Strong FCF conversion
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing valuation ratios")
        valuation_defs = [("price_to_earnings_ratio", 25), ("price_to_book_ratio", 3), ("price_to_sales_ratio", 5)]
        valuation_result = _analyze_metric_group(metrics, valuation_defs, higher_is_better=False)
        signals.append(valuation_result['signal'])
        reasoning["price_ratios_signal"] = valuation_result

        progress.update_status("fundamentals_analyst_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")
        total_signals = len(signals)

        net_score = bullish_signals - bearish_signals

        # Confidence is the magnitude of the net score relative to the total possible signals
        if total_signals > 0:
            confidence = (abs(net_score) / total_signals) * 100
        else:
            confidence = 0
        
        # Cap the confidence at 80%
        confidence = min(confidence, 80.0)

        if net_score > 0:
            overall_signal = "bullish"
        elif net_score < 0:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_analyst_agent", ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_analyst_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_analyst_agent"] = fundamental_analysis

    progress.update_status("fundamentals_analyst_agent", None, "Done")
    
    data_update = {
        "analyst_signals": {
            "fundamentals_analyst_agent": fundamental_analysis
        }
    }

    # The return statement should be:
    return {"messages": [message], "data": data_update}
