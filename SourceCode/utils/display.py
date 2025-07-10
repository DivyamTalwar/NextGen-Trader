from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import json

def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.
    """
    decisions = result.get("decisions")
    if not decisions:
        print(f"{Fore.RED}No trading decisions available{Style.RESET_ALL}")
        return

    for ticker, decision in decisions.items():
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals or agent == "risk_management_agent":
                continue

            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)
            signal_color = {"BULLISH": Fore.GREEN, "BEARISH": Fore.RED, "NEUTRAL": Fore.YELLOW}.get(signal_type, Fore.WHITE)
            
            reasoning_str = json.dumps(signal.get("reasoning"), indent=2) if isinstance(signal.get("reasoning"), dict) else str(signal.get("reasoning", ""))
            
            table_data.append([
                f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                f"{signal_color}{signal_type}{Style.RESET_ALL}",
                f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                f"{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}",
            ])

        table_data = sort_agent_signals(table_data)
        print(f"\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(table_data, headers=[f"{Fore.WHITE}Agent", "Signal", "Confidence", "Reasoning"], tablefmt="fancy_grid", colalign=("left", "center", "right", "left")))

        action = decision.get("action", "").upper()
        action_color = {"BUY": Fore.GREEN, "SELL": Fore.RED, "HOLD": Fore.YELLOW, "COVER": Fore.GREEN, "SHORT": Fore.RED}.get(action, Fore.WHITE)
        
        decision_data = [
            ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
            ["Quantity", f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}"],
            ["Confidence", f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}"],
            ["Reasoning", f"{Fore.WHITE}{decision.get('reasoning', '')}{Style.RESET_ALL}"],
        ]
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(decision_data, tablefmt="fancy_grid", colalign=("left", "left")))

    print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    portfolio_data = [
        [
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{decision.get('action', '').upper()}{Style.RESET_ALL}",
            f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}",
            f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
        ]
        for ticker, decision in decisions.items()
    ]
    
    print(tabulate(portfolio_data, headers=[f"{Fore.WHITE}Ticker", "Action", "Quantity", "Confidence"], tablefmt="fancy_grid", colalign=("left", "center", "right", "right")))

def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    
    # Display latest portfolio summary
    if summary_rows:
        latest_summary = summary_rows[-1]
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")

        # Extract values and remove commas before converting to float
        cash_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest_summary[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Return: {latest_summary[9]}")
        
        # Display performance metrics if available
        if latest_summary[10]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest_summary[10]}")
        if latest_summary[11]:  # Sortino ratio
            print(f"Sortino Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Max drawdown
            print(f"Max Drawdown: {latest_summary[12]}")

    # Add vertical spacing
    print("\n" * 2)

    print(tabulate(ticker_rows, headers=["Date", "Ticker", "Action", "Quantity", "Price", "Shares", "Position Value", "Bullish", "Bearish", "Neutral"], tablefmt="fancy_grid", colalign=("left", "left", "center", "right", "right", "right", "right", "right", "right", "right")))

    # Add vertical spacing
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    shares_owned: float,
    position_value: float,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    action_color = {"BUY": Fore.GREEN, "COVER": Fore.GREEN, "SELL": Fore.RED, "SHORT": Fore.RED, "HOLD": Fore.WHITE}.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "", "", "", "",
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",
            f"{Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}",
        ]
