"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


# ============= Base Response Models =============
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel):
    status: ResponseStatus
    message: str
    data: Optional[Any] = None


# ============= Trading Actions =============
class TradingAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SHORT = "short"
    COVER = "cover"


class SignalType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ============= Portfolio Models =============
class Position(BaseModel):
    long: int = 0
    short: int = 0
    long_cost_basis: float = 0.0
    short_cost_basis: float = 0.0


class Portfolio(BaseModel):
    total_cash: float
    margin_requirement: float = 0.2
    positions: Dict[str, Position]


class PortfolioUpdate(BaseModel):
    ticker: str
    action: TradingAction
    quantity: int = Field(ge=0)
    price: Optional[float] = None


# ============= Agent Models =============
class AnalystSignal(BaseModel):
    signal: SignalType
    confidence: float = Field(ge=0, le=100)
    reasoning: str


class AgentAnalysisRequest(BaseModel):
    tickers: List[str] = Field(min_items=1)
    start_date: str
    end_date: str
    agent_name: str
    model_name: Optional[str] = "llama-3.3-70b-versatile"
    model_provider: Optional[str] = "groq"
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values:
            if v < values['start_date']:
                raise ValueError("End date must be after start date")
        return v


class AgentAnalysisResponse(BaseModel):
    agent_name: str
    analysis_date: str
    signals: Dict[str, AnalystSignal]
    execution_time: float


# ============= Workflow Models =============
class WorkflowRequest(BaseModel):
    tickers: List[str] = Field(min_items=1)
    start_date: str
    end_date: str
    initial_capital: float = Field(default=100000, gt=0)
    selected_analysts: Optional[List[str]] = None
    model_name: Optional[str] = "llama-3.3-70b-versatile"
    model_provider: Optional[str] = "groq"
    show_reasoning: bool = True
    margin_requirement: float = Field(default=0.2, ge=0, le=1)
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class TradingDecision(BaseModel):
    action: TradingAction
    quantity: int
    confidence: float = Field(ge=0, le=100)
    reasoning: str


class WorkflowResponse(BaseModel):
    decisions: Dict[str, TradingDecision]
    analyst_signals: Dict[str, Dict[str, AnalystSignal]]
    portfolio_state: Portfolio
    execution_time: float


# ============= Backtesting Models =============
class BacktestRequest(BaseModel):
    tickers: List[str] = Field(min_items=1)
    start_date: str
    end_date: str
    initial_capital: float = Field(default=100000, gt=0)
    selected_analysts: Optional[List[str]] = None
    model_name: Optional[str] = "llama-3.3-70b-versatile"
    model_provider: Optional[str] = "groq"
    lookback_days: int = Field(default=30, ge=1)
    margin_requirement: float = Field(default=0.2, ge=0, le=1)
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class BacktestMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    average_win: float
    average_loss: float
    best_trade: float
    worst_trade: float
    final_portfolio_value: float


class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    portfolio_values: List[float]
    trades: List[Dict[str, Any]]
    dates: List[str]
    execution_time: float


# ============= Market Data Models =============
class PriceDataRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class PriceData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


class FinancialMetricsRequest(BaseModel):
    ticker: str
    limit: int = Field(default=10, ge=1, le=100)


class FinancialMetrics(BaseModel):
    ticker: str
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    revenue_growth: Optional[float]
    profit_margin: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]


class NewsRequest(BaseModel):
    ticker: str
    limit: int = Field(default=10, ge=1, le=50)


class NewsArticle(BaseModel):
    title: str
    url: str
    published_date: str
    source: str
    summary: Optional[str] = None


class InsiderTradeRequest(BaseModel):
    ticker: str
    limit: int = Field(default=20, ge=1, le=100)


class InsiderTrade(BaseModel):
    insider_name: str
    position: str
    transaction_date: str
    transaction_type: str
    shares: int
    price: float
    value: float


# ============= Model Configuration =============
class ModelInfo(BaseModel):
    name: str
    provider: str
    context_window: int
    supports_vision: bool
    supports_tools: bool
    is_available: bool


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    default_provider: str


# ============= Error Models =============
class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    type: str = "validation_error"


class ErrorResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.ERROR
    message: str
    errors: Optional[List[ErrorDetail]] = None
    data: None = None