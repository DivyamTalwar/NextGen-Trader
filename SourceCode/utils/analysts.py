"""Constants and utilities related to analysts configuration."""

from agents.aswath_damodaran import aswath_damodaran_agent
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.cathie_wood import cathie_wood_agent
from agents.charlie_munger import charlie_munger_agent
from agents.fundamentals import fundamentals_analyst_agent
from agents.phil_fisher import phil_fisher_agent
from agents.sentiment import sentiment_analyst_agent
from agents.stanley_druckenmiller import stanley_druckenmiller_agent
from agents.technicals import technical_analyst_agent
from agents.valuation import valuation_analyst_agent
from agents.warren_buffett import warren_buffett_agent
from agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "aswath_damodaran": {
        "name": "Aswath Damodaran",
        "display_name": "Aswath Damodaran",
        "agent_func": aswath_damodaran_agent,
        "order": 0,
    },
    "ben_graham": {
        "name": "Ben Graham",
        "display_name": "Ben Graham",
        "agent_func": ben_graham_agent,
        "order": 1,
    },
    "bill_ackman": {
        "name": "Bill Ackman",
        "display_name": "Bill Ackman",
        "agent_func": bill_ackman_agent,
        "order": 2,
    },
    "cathie_wood": {
        "name": "Cathie Wood",
        "display_name": "Cathie Wood",
        "agent_func": cathie_wood_agent,
        "order": 3,
    },
    "charlie_munger": {
        "name": "Charlie Munger",
        "display_name": "Charlie Munger",
        "agent_func": charlie_munger_agent,
        "order": 4,
    },
    "phil_fisher": {
        "name": "Phil Fisher",
        "display_name": "Phil Fisher",
        "agent_func": phil_fisher_agent,
        "order": 7,
    },
    "rakesh_jhunjhunwala": {
        "name": "Rakesh Jhunjhunwala",
        "display_name": "Rakesh Jhunjhunwala",
        "agent_func": rakesh_jhunjhunwala_agent,
        "order": 8,
    },
    "stanley_druckenmiller": {
        "name": "Stanley Druckenmiller",
        "display_name": "Stanley Druckenmiller",
        "agent_func": stanley_druckenmiller_agent,
        "order": 9,
    },
    "warren_buffett": {
        "name": "Warren Buffett",
        "display_name": "Warren Buffett",
        "agent_func": warren_buffett_agent,
        "order": 10,
    },
    "technical_analyst": {
        "name": "Technical Analyst",
        "display_name": "Technical Analyst",
        "agent_func": technical_analyst_agent,
        "order": 11,
    },
    "fundamentals_analyst": {
        "name": "Fundamentals Analyst",
        "display_name": "Fundamentals Analyst",
        "agent_func": fundamentals_analyst_agent,
        "order": 12,
    },
    "sentiment_analyst": {
        "name": "Sentiment Analyst",
        "display_name": "Sentiment Analyst",
        "agent_func": sentiment_analyst_agent,
        "order": 13,
    },
    "valuation_analyst": {
        "name": "Valuation Analyst",
        "display_name": "Valuation Analyst",
        "agent_func": valuation_analyst_agent,
        "order": 14,
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}
