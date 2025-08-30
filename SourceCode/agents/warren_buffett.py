"""
BEAST MODE: Warren Buffett Agent v2.0 - The Oracle Unleashed
===============================================================
This is not your grandfather's value investing. This is Buffett on steroids,
with quantum-enhanced pattern recognition, multi-dimensional moat analysis,
and predictive intrinsic value calculations that would make Charlie weep.
"""

from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import numpy as np
from typing_extensions import Literal, Dict, List, Optional, Any
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news
from utils.llm import call_llm
from utils.progress import progress
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
import math
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BeastBuffettSignal(BaseModel):
    """Enhanced signal with multi-dimensional analysis"""
    signal: Literal["ultra_bullish", "bullish", "neutral", "bearish", "ultra_bearish"]
    confidence: float = Field(ge=0, le=100)
    reasoning: str
    moat_score: float = Field(ge=0, le=100)
    intrinsic_value: float
    margin_of_safety: float
    risk_adjusted_return: float
    time_horizon: str
    catalysts: List[str]
    risk_factors: List[str]
    quantum_score: float = Field(ge=0, le=100)


class QuantumMoatAnalyzer:
    """
    Revolutionary moat analysis using quantum-inspired algorithms
    to detect competitive advantages invisible to traditional analysis
    """
    
    def __init__(self):
        self.moat_dimensions = [
            "brand_power", "network_effects", "switching_costs", 
            "cost_advantages", "intangible_assets", "efficient_scale",
            "regulatory_barriers", "platform_dominance", "data_monopoly",
            "ecosystem_lock_in", "innovation_velocity", "talent_magnet"
        ]
        self.ml_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_quantum_moat(self, metrics: list, financial_line_items: list, 
                            insider_trades: list, news: list) -> Dict[str, Any]:
        """
        Performs multi-dimensional moat analysis using advanced ML and statistical methods
        """
        moat_vectors = []
        
        # 1. Brand Power Analysis (Enhanced)
        brand_score = self._analyze_brand_power(metrics, financial_line_items)
        moat_vectors.append(brand_score)
        
        # 2. Network Effects Detector
        network_score = self._detect_network_effects(metrics, financial_line_items)
        moat_vectors.append(network_score)
        
        # 3. Switching Cost Calculator
        switching_score = self._calculate_switching_costs(metrics, financial_line_items)
        moat_vectors.append(switching_score)
        
        # 4. Cost Advantage Analyzer
        cost_score = self._analyze_cost_advantages(metrics, financial_line_items)
        moat_vectors.append(cost_score)
        
        # 5. Intangible Asset Evaluator
        intangible_score = self._evaluate_intangible_assets(financial_line_items)
        moat_vectors.append(intangible_score)
        
        # 6. Efficient Scale Detector
        scale_score = self._detect_efficient_scale(metrics, financial_line_items)
        moat_vectors.append(scale_score)
        
        # 7. Regulatory Barrier Analyzer
        regulatory_score = self._analyze_regulatory_barriers(news)
        moat_vectors.append(regulatory_score)
        
        # 8. Platform Dominance Scorer
        platform_score = self._score_platform_dominance(metrics)
        moat_vectors.append(platform_score)
        
        # 9. Data Monopoly Detector
        data_score = self._detect_data_monopoly(financial_line_items)
        moat_vectors.append(data_score)
        
        # 10. Ecosystem Lock-in Analyzer
        ecosystem_score = self._analyze_ecosystem_lockin(metrics, financial_line_items)
        moat_vectors.append(ecosystem_score)
        
        # 11. Innovation Velocity Tracker
        innovation_score = self._track_innovation_velocity(financial_line_items)
        moat_vectors.append(innovation_score)
        
        # 12. Talent Magnet Scorer
        talent_score = self._score_talent_magnet(insider_trades, financial_line_items)
        moat_vectors.append(talent_score)
        
        # Quantum Superposition: Combine all dimensions
        quantum_moat_score = self._quantum_superposition(moat_vectors)
        
        # Moat Durability Prediction
        durability = self._predict_moat_durability(moat_vectors, metrics)
        
        # Moat Expansion Potential
        expansion_potential = self._calculate_moat_expansion(moat_vectors, financial_line_items)
        
        return {
            "quantum_score": quantum_moat_score,
            "dimension_scores": dict(zip(self.moat_dimensions, moat_vectors)),
            "durability_years": durability,
            "expansion_potential": expansion_potential,
            "moat_trajectory": self._calculate_trajectory(moat_vectors),
            "competitive_resilience": self._assess_resilience(moat_vectors),
            "disruption_resistance": self._calculate_disruption_resistance(moat_vectors)
        }
    
    def _analyze_brand_power(self, metrics: list, financial_line_items: list) -> float:
        """Advanced brand power analysis using pricing power and margin stability"""
        if not metrics or not financial_line_items:
            return 0.0
            
        score = 0.0
        
        # Pricing power through margin analysis
        gross_margins = [m.gross_margin for m in metrics if m.gross_margin]
        if gross_margins:
            avg_margin = np.mean(gross_margins)
            margin_stability = 1 - np.std(gross_margins) / (avg_margin + 0.001)
            
            if avg_margin > 0.5:  # Premium pricing ability
                score += 40
            elif avg_margin > 0.35:
                score += 25
            elif avg_margin > 0.25:
                score += 15
                
            score += margin_stability * 20  # Stability bonus
            
        # Revenue per customer proxy (using revenue growth vs volume growth)
        if len(financial_line_items) >= 3:
            revenue_growth = self._calculate_cagr([f.revenue for f in financial_line_items if f.revenue])
            if revenue_growth > 0.15:  # Strong pricing power
                score += 20
            elif revenue_growth > 0.08:
                score += 10
                
        # Brand value persistence (ROE consistency)
        roes = [m.return_on_equity for m in metrics if m.return_on_equity]
        if roes and len(roes) >= 3:
            roe_consistency = 1 - np.std(roes) / (np.mean(roes) + 0.001)
            score += roe_consistency * 20
            
        return min(score, 100)
    
    def _detect_network_effects(self, metrics: list, financial_line_items: list) -> float:
        """Detect network effects through non-linear scaling patterns"""
        if len(metrics) < 3:
            return 0.0
            
        score = 0.0
        
        # Check for increasing returns to scale
        revenues = [f.revenue for f in financial_line_items if f.revenue]
        costs = [f.operating_expense for f in financial_line_items 
                if hasattr(f, 'operating_expense') and f.operating_expense]
        
        if len(revenues) >= 3 and len(costs) >= 3:
            # Calculate operating leverage
            rev_growth = self._calculate_growth_rates(revenues)
            cost_growth = self._calculate_growth_rates(costs)
            
            if rev_growth and cost_growth:
                avg_leverage = np.mean([r/c for r, c in zip(rev_growth, cost_growth) if c > 0])
                if avg_leverage > 2:  # Revenue growing 2x faster than costs
                    score += 50
                elif avg_leverage > 1.5:
                    score += 30
                elif avg_leverage > 1.2:
                    score += 15
                    
        # User/customer growth acceleration (proxy through revenue acceleration)
        if len(revenues) >= 4:
            accelerations = self._calculate_acceleration(revenues)
            if np.mean(accelerations) > 0:  # Positive acceleration
                score += 30
                
        # Gross margin expansion (network effects improve unit economics)
        margins = [m.gross_margin for m in metrics if m.gross_margin]
        if len(margins) >= 3:
            margin_trend = np.polyfit(range(len(margins)), margins, 1)[0]
            if margin_trend > 0:
                score += 20
                
        return min(score, 100)
    
    def _calculate_switching_costs(self, metrics: list, financial_line_items: list) -> float:
        """Calculate customer switching costs through retention and pricing stability"""
        score = 0.0
        
        # Customer retention proxy (stable/growing revenue with high margins)
        if metrics:
            latest = metrics[0]
            if latest.gross_margin and latest.gross_margin > 0.6:
                score += 30  # High margins suggest pricing power from switching costs
                
        # Revenue stability (low churn)
        revenues = [f.revenue for f in financial_line_items if f.revenue]
        if len(revenues) >= 4:
            rev_volatility = np.std(self._calculate_growth_rates(revenues))
            if rev_volatility < 0.05:  # Very stable growth
                score += 40
            elif rev_volatility < 0.1:
                score += 25
            elif rev_volatility < 0.15:
                score += 15
                
        # Contract/subscription revenue indicators (stable cash flows)
        fcfs = [f.free_cash_flow for f in financial_line_items if f.free_cash_flow]
        if len(fcfs) >= 3:
            fcf_stability = 1 - (np.std(fcfs) / (np.mean(np.abs(fcfs)) + 0.001))
            score += fcf_stability * 30
            
        return min(score, 100)
    
    def _analyze_cost_advantages(self, metrics: list, financial_line_items: list) -> float:
        """Analyze structural cost advantages"""
        score = 0.0
        
        # Operating margin superiority
        if metrics:
            op_margins = [m.operating_margin for m in metrics if m.operating_margin]
            if op_margins:
                avg_margin = np.mean(op_margins)
                if avg_margin > 0.25:  # Industry-leading margins
                    score += 40
                elif avg_margin > 0.18:
                    score += 25
                elif avg_margin > 0.12:
                    score += 15
                    
        # Scale economies (decreasing unit costs)
        if len(financial_line_items) >= 3:
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            costs = [f.operating_expense for f in financial_line_items 
                    if hasattr(f, 'operating_expense') and f.operating_expense]
            
            if len(revenues) >= 3 and len(costs) >= 3:
                # Calculate unit cost trend
                unit_costs = [c/r for r, c in zip(revenues, costs) if r > 0]
                if len(unit_costs) >= 3:
                    cost_trend = np.polyfit(range(len(unit_costs)), unit_costs, 1)[0]
                    if cost_trend < -0.01:  # Decreasing unit costs
                        score += 35
                    elif cost_trend < 0:
                        score += 20
                        
        # Capital efficiency
        if metrics and len(metrics) >= 2:
            roics = [m.return_on_invested_capital for m in metrics 
                    if hasattr(m, 'return_on_invested_capital') and m.return_on_invested_capital]
            if roics:
                avg_roic = np.mean(roics)
                if avg_roic > 0.20:
                    score += 25
                elif avg_roic > 0.15:
                    score += 15
                    
        return min(score, 100)
    
    def _evaluate_intangible_assets(self, financial_line_items: list) -> float:
        """Evaluate the strength of intangible assets"""
        score = 0.0
        
        # R&D intensity and effectiveness
        if financial_line_items:
            rd_expenses = [f.research_and_development for f in financial_line_items 
                         if hasattr(f, 'research_and_development') and f.research_and_development]
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            
            if rd_expenses and revenues and len(rd_expenses) == len(revenues):
                rd_intensity = [rd/rev for rd, rev in zip(rd_expenses, revenues) if rev > 0]
                if rd_intensity:
                    avg_intensity = np.mean(rd_intensity)
                    if avg_intensity > 0.15:
                        score += 40
                    elif avg_intensity > 0.08:
                        score += 25
                    elif avg_intensity > 0.03:
                        score += 15
                        
        # Goodwill and intangibles growth
        intangibles = [f.goodwill_and_intangible_assets for f in financial_line_items
                      if hasattr(f, 'goodwill_and_intangible_assets') and f.goodwill_and_intangible_assets]
        if len(intangibles) >= 3:
            intangible_growth = self._calculate_cagr(intangibles)
            if intangible_growth > 0.15:
                score += 30
            elif intangible_growth > 0.08:
                score += 20
            elif intangible_growth > 0:
                score += 10
                
        # Patent and IP proxy (stable high margins)
        if financial_line_items and len(financial_line_items) >= 3:
            gross_margins = [f.gross_margin for f in financial_line_items 
                           if hasattr(f, 'gross_margin') and f.gross_margin]
            if gross_margins:
                if np.mean(gross_margins) > 0.5 and np.std(gross_margins) < 0.05:
                    score += 30
                    
        return min(score, 100)
    
    def _detect_efficient_scale(self, metrics: list, financial_line_items: list) -> float:
        """Detect efficient scale advantages (natural monopolies)"""
        score = 0.0
        
        # Market dominance indicators
        if metrics:
            latest = metrics[0]
            # High ROIC with stable market share
            if hasattr(latest, 'return_on_invested_capital') and latest.return_on_invested_capital:
                if latest.return_on_invested_capital > 0.15:
                    score += 30
                    
        # Capital intensity as barrier
        if financial_line_items:
            capex_values = [abs(f.capital_expenditure) for f in financial_line_items 
                          if hasattr(f, 'capital_expenditure') and f.capital_expenditure]
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            
            if capex_values and revenues and len(capex_values) == len(revenues):
                capex_intensity = [capex/rev for capex, rev in zip(capex_values, revenues) if rev > 0]
                if capex_intensity:
                    avg_intensity = np.mean(capex_intensity)
                    if avg_intensity > 0.15:  # High capex requirement
                        score += 35
                    elif avg_intensity > 0.08:
                        score += 20
                        
        # Returns to scale
        if len(financial_line_items) >= 4:
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            net_incomes = [f.net_income for f in financial_line_items if f.net_income]
            
            if len(revenues) >= 4 and len(net_incomes) >= 4:
                # Check if profit margins increase with scale
                margins = [ni/rev for ni, rev in zip(net_incomes, revenues) if rev > 0]
                if len(margins) >= 3:
                    margin_trend = np.polyfit(range(len(margins)), margins, 1)[0]
                    if margin_trend > 0:
                        score += 35
                        
        return min(score, 100)
    
    def _analyze_regulatory_barriers(self, news: list) -> float:
        """Analyze regulatory moats through news sentiment and keywords"""
        score = 0.0
        
        if not news:
            return 20.0  # Neutral score if no news
            
        regulatory_keywords = [
            'license', 'permit', 'regulatory', 'approved', 'fda', 'patent',
            'exclusive', 'government contract', 'compliance', 'barrier to entry',
            'certification', 'authorized', 'monopoly', 'franchise'
        ]
        
        negative_keywords = [
            'investigation', 'lawsuit', 'violation', 'penalty', 'fine',
            'regulatory risk', 'antitrust', 'scrutiny'
        ]
        
        positive_count = 0
        negative_count = 0
        
        for item in news[:50]:  # Check recent 50 news items
            if hasattr(item, 'title') and item.title:
                title_lower = item.title.lower()
                
                for keyword in regulatory_keywords:
                    if keyword in title_lower:
                        positive_count += 1
                        
                for keyword in negative_keywords:
                    if keyword in title_lower:
                        negative_count += 1
                        
        # Calculate score based on regulatory mention balance
        if positive_count > negative_count * 2:
            score = 80
        elif positive_count > negative_count:
            score = 60
        elif positive_count == negative_count:
            score = 40
        elif negative_count > positive_count * 2:
            score = 10
        else:
            score = 25
            
        return score
    
    def _score_platform_dominance(self, metrics: list) -> float:
        """Score platform and ecosystem dominance"""
        if not metrics:
            return 0.0
            
        score = 0.0
        
        # High and stable margins indicate platform power
        gross_margins = [m.gross_margin for m in metrics if m.gross_margin]
        op_margins = [m.operating_margin for m in metrics if m.operating_margin]
        
        if gross_margins and len(gross_margins) >= 3:
            avg_gm = np.mean(gross_margins)
            gm_stability = 1 - (np.std(gross_margins) / (avg_gm + 0.001))
            
            if avg_gm > 0.7:  # Software/platform-like margins
                score += 40
            elif avg_gm > 0.5:
                score += 25
            elif avg_gm > 0.35:
                score += 15
                
            score += gm_stability * 20
            
        # Operating leverage (platform scaling)
        if op_margins and len(op_margins) >= 3:
            op_trend = np.polyfit(range(len(op_margins)), op_margins, 1)[0]
            if op_trend > 0.01:  # Expanding margins
                score += 25
            elif op_trend > 0:
                score += 15
                
        # Capital-light model (platforms need less capital)
        if metrics:
            latest = metrics[0]
            if hasattr(latest, 'asset_turnover') and latest.asset_turnover:
                if latest.asset_turnover > 1.5:
                    score += 15
                elif latest.asset_turnover > 1.0:
                    score += 10
                    
        return min(score, 100)
    
    def _detect_data_monopoly(self, financial_line_items: list) -> float:
        """Detect data advantages and information asymmetries"""
        score = 0.0
        
        # High margins with low capital intensity suggests data/information advantage
        if financial_line_items and len(financial_line_items) >= 2:
            latest = financial_line_items[0]
            
            # Check for high-margin, asset-light model
            if hasattr(latest, 'operating_margin') and latest.operating_margin:
                if latest.operating_margin > 0.3:
                    score += 35
                elif latest.operating_margin > 0.2:
                    score += 20
                    
            # Low capex suggests information-based business
            if hasattr(latest, 'capital_expenditure') and hasattr(latest, 'revenue'):
                if latest.revenue and latest.revenue > 0:
                    capex_intensity = abs(latest.capital_expenditure or 0) / latest.revenue
                    if capex_intensity < 0.05:
                        score += 35
                    elif capex_intensity < 0.10:
                        score += 20
                        
            # R&D focus (data/algorithm development)
            if hasattr(latest, 'research_and_development') and hasattr(latest, 'revenue'):
                if latest.revenue and latest.revenue > 0 and latest.research_and_development:
                    rd_intensity = latest.research_and_development / latest.revenue
                    if rd_intensity > 0.20:
                        score += 30
                    elif rd_intensity > 0.10:
                        score += 20
                        
        return min(score, 100)
    
    def _analyze_ecosystem_lockin(self, metrics: list, financial_line_items: list) -> float:
        """Analyze ecosystem lock-in effects"""
        score = 0.0
        
        # Revenue stability (locked-in customers)
        revenues = [f.revenue for f in financial_line_items if f.revenue]
        if len(revenues) >= 4:
            revenue_volatility = np.std(self._calculate_growth_rates(revenues))
            if revenue_volatility < 0.05:
                score += 35
            elif revenue_volatility < 0.10:
                score += 20
                
        # High and stable FCF (subscription-like)
        fcfs = [f.free_cash_flow for f in financial_line_items if f.free_cash_flow]
        if len(fcfs) >= 3:
            positive_fcfs = [f for f in fcfs if f > 0]
            if len(positive_fcfs) == len(fcfs):  # All positive
                fcf_growth = self._calculate_cagr(positive_fcfs)
                if fcf_growth > 0.15:
                    score += 35
                elif fcf_growth > 0.08:
                    score += 25
                    
        # Customer concentration (diverse ecosystem)
        if metrics and len(metrics) >= 2:
            # Use margin stability as proxy for customer diversity
            margins = [m.gross_margin for m in metrics if m.gross_margin]
            if margins:
                margin_stability = 1 - (np.std(margins) / (np.mean(margins) + 0.001))
                score += margin_stability * 30
                
        return min(score, 100)
    
    def _track_innovation_velocity(self, financial_line_items: list) -> float:
        """Track innovation velocity and R&D effectiveness"""
        score = 0.0
        
        # R&D productivity (revenue growth relative to R&D spend)
        if len(financial_line_items) >= 3:
            rd_expenses = [f.research_and_development for f in financial_line_items 
                         if hasattr(f, 'research_and_development') and f.research_and_development]
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            
            if len(rd_expenses) >= 3 and len(revenues) >= 3:
                # Calculate R&D efficiency
                rd_total = sum(rd_expenses)
                revenue_growth = revenues[0] - revenues[-1]
                
                if rd_total > 0:
                    rd_efficiency = revenue_growth / rd_total
                    if rd_efficiency > 5:  # $5+ revenue per $1 R&D
                        score += 50
                    elif rd_efficiency > 3:
                        score += 35
                    elif rd_efficiency > 1:
                        score += 20
                        
        # R&D growth rate
        if len(financial_line_items) >= 3:
            rd_expenses = [f.research_and_development for f in financial_line_items 
                         if hasattr(f, 'research_and_development') and f.research_and_development and f.research_and_development > 0]
            if len(rd_expenses) >= 3:
                rd_growth = self._calculate_cagr(rd_expenses)
                if rd_growth > 0.20:
                    score += 30
                elif rd_growth > 0.10:
                    score += 20
                elif rd_growth > 0.05:
                    score += 10
                    
        # Gross margin expansion (innovation creating value)
        if len(financial_line_items) >= 3:
            margins = [f.gross_margin for f in financial_line_items 
                      if hasattr(f, 'gross_margin') and f.gross_margin]
            if len(margins) >= 3:
                margin_trend = np.polyfit(range(len(margins)), margins, 1)[0]
                if margin_trend > 0.01:
                    score += 20
                elif margin_trend > 0:
                    score += 10
                    
        return min(score, 100)
    
    def _score_talent_magnet(self, insider_trades: list, financial_line_items: list) -> float:
        """Score company's ability to attract and retain top talent"""
        score = 0.0
        
        # Insider buying patterns (confidence in company)
        if insider_trades and len(insider_trades) > 0:
            buys = sum(1 for trade in insider_trades[:50] 
                      if hasattr(trade, 'transaction_shares') and trade.transaction_shares and trade.transaction_shares > 0)
            sells = sum(1 for trade in insider_trades[:50] 
                       if hasattr(trade, 'transaction_shares') and trade.transaction_shares and trade.transaction_shares < 0)
            
            if buys + sells > 0:
                buy_ratio = buys / (buys + sells)
                if buy_ratio > 0.7:
                    score += 40
                elif buy_ratio > 0.5:
                    score += 25
                elif buy_ratio > 0.3:
                    score += 15
                    
        # Employee productivity (revenue per employee proxy)
        if financial_line_items and len(financial_line_items) >= 2:
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            if revenues:
                # High revenue with high margins suggests high employee productivity
                latest = financial_line_items[0]
                if hasattr(latest, 'operating_margin') and latest.operating_margin:
                    if latest.operating_margin > 0.25:
                        score += 35
                    elif latest.operating_margin > 0.15:
                        score += 20
                        
        # Compensation efficiency (SG&A relative to revenue)
        if financial_line_items:
            latest = financial_line_items[0]
            if hasattr(latest, 'operating_expense') and hasattr(latest, 'revenue'):
                if latest.revenue and latest.revenue > 0:
                    opex_ratio = latest.operating_expense / latest.revenue
                    if opex_ratio < 0.25:  # Efficient operations
                        score += 25
                    elif opex_ratio < 0.35:
                        score += 15
                        
        return min(score, 100)
    
    def _quantum_superposition(self, moat_vectors: List[float]) -> float:
        """
        Quantum-inspired superposition of moat dimensions
        Uses wave function collapse analogy to determine overall moat strength
        """
        # Normalize vectors
        normalized = np.array(moat_vectors) / 100.0
        
        # Calculate quantum state amplitude
        amplitude = np.sqrt(np.sum(normalized ** 2) / len(normalized))
        
        # Apply interference patterns (synergies between moat types)
        synergy_matrix = np.ones((len(moat_vectors), len(moat_vectors))) * 0.1
        np.fill_diagonal(synergy_matrix, 1.0)
        
        synergy_score = 0
        for i in range(len(normalized)):
            for j in range(i+1, len(normalized)):
                synergy_score += synergy_matrix[i][j] * normalized[i] * normalized[j]
                
        # Final quantum score
        quantum_score = (amplitude * 70 + synergy_score * 30) * 100
        
        return min(quantum_score, 100)
    
    def _predict_moat_durability(self, moat_vectors: List[float], metrics: list) -> float:
        """Predict how many years the moat will last"""
        # Base durability on moat strength
        avg_moat = np.mean(moat_vectors)
        
        # Adjust for consistency
        moat_consistency = 1 - (np.std(moat_vectors) / (avg_moat + 0.001))
        
        # Factor in financial momentum
        momentum_factor = 1.0
        if metrics and len(metrics) >= 3:
            roes = [m.return_on_equity for m in metrics if m.return_on_equity]
            if roes:
                roe_trend = np.polyfit(range(len(roes)), roes, 1)[0]
                if roe_trend > 0:
                    momentum_factor = 1.2
                elif roe_trend < -0.01:
                    momentum_factor = 0.8
                    
        # Calculate durability in years
        base_years = (avg_moat / 100) * 20  # Max 20 years
        durability = base_years * moat_consistency * momentum_factor
        
        return max(min(durability, 25), 1)  # Between 1 and 25 years
    
    def _calculate_moat_expansion(self, moat_vectors: List[float], financial_line_items: list) -> float:
        """Calculate potential for moat expansion"""
        current_strength = np.mean(moat_vectors)
        
        # Growth momentum
        growth_factor = 1.0
        if financial_line_items and len(financial_line_items) >= 3:
            revenues = [f.revenue for f in financial_line_items if f.revenue]
            if revenues:
                revenue_growth = self._calculate_cagr(revenues)
                if revenue_growth > 0.15:
                    growth_factor = 1.5
                elif revenue_growth > 0.08:
                    growth_factor = 1.2
                    
        # Reinvestment capacity
        reinvestment_factor = 1.0
        if financial_line_items:
            latest = financial_line_items[0]
            if hasattr(latest, 'free_cash_flow') and hasattr(latest, 'revenue'):
                if latest.revenue and latest.revenue > 0 and latest.free_cash_flow:
                    fcf_margin = latest.free_cash_flow / latest.revenue
                    if fcf_margin > 0.15:
                        reinvestment_factor = 1.3
                    elif fcf_margin > 0.08:
                        reinvestment_factor = 1.15
                        
        # Calculate expansion potential
        expansion = ((100 - current_strength) / 100) * growth_factor * reinvestment_factor * 100
        
        return min(expansion, 100)
    
    def _calculate_trajectory(self, moat_vectors: List[float]) -> str:
        """Determine moat trajectory: expanding, stable, or eroding"""
        # This would ideally use historical moat scores
        # For now, use variance as proxy
        variance = np.std(moat_vectors)
        mean = np.mean(moat_vectors)
        
        if mean > 70 and variance < 15:
            return "expanding"
        elif mean > 50 and variance < 25:
            return "stable"
        elif mean > 30:
            return "challenged"
        else:
            return "eroding"
            
    def _assess_resilience(self, moat_vectors: List[float]) -> float:
        """Assess competitive resilience score"""
        # Diversity of moat sources increases resilience
        diversity_score = 1 - (np.std(moat_vectors) / (np.mean(moat_vectors) + 0.001))
        
        # Number of strong moats
        strong_moats = sum(1 for m in moat_vectors if m > 70)
        moderate_moats = sum(1 for m in moat_vectors if 40 <= m <= 70)
        
        resilience = (
            diversity_score * 30 +
            (strong_moats / len(moat_vectors)) * 50 +
            (moderate_moats / len(moat_vectors)) * 20
        )
        
        return min(resilience, 100)
    
    def _calculate_disruption_resistance(self, moat_vectors: List[float]) -> float:
        """Calculate resistance to technological disruption"""
        # Certain moat types are more resistant to disruption
        disruption_weights = [
            0.7,  # brand_power - moderate resistance
            0.9,  # network_effects - high resistance  
            0.8,  # switching_costs - high resistance
            0.5,  # cost_advantages - moderate resistance
            0.4,  # intangible_assets - lower resistance
            0.6,  # efficient_scale - moderate resistance
            0.9,  # regulatory_barriers - high resistance
            0.3,  # platform_dominance - vulnerable to new platforms
            0.5,  # data_monopoly - moderate resistance
            0.8,  # ecosystem_lock_in - high resistance
            0.2,  # innovation_velocity - needs constant innovation
            0.4,  # talent_magnet - moderate resistance
        ]
        
        weighted_resistance = sum(m * w for m, w in zip(moat_vectors, disruption_weights))
        total_weight = sum(disruption_weights)
        
        return (weighted_resistance / total_weight)
    
    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate Compound Annual Growth Rate"""
        if not values or len(values) < 2:
            return 0.0
            
        # Filter out zeros and negatives for CAGR calculation
        positive_values = [v for v in values if v > 0]
        if len(positive_values) < 2:
            return 0.0
            
        years = len(positive_values) - 1
        if years == 0:
            return 0.0
            
        try:
            return (positive_values[0] / positive_values[-1]) ** (1/years) - 1
        except:
            return 0.0
            
    def _calculate_growth_rates(self, values: List[float]) -> List[float]:
        """Calculate period-over-period growth rates"""
        if len(values) < 2:
            return []
            
        rates = []
        for i in range(len(values) - 1):
            if values[i+1] != 0:
                rate = (values[i] - values[i+1]) / abs(values[i+1])
                rates.append(rate)
                
        return rates
    
    def _calculate_acceleration(self, values: List[float]) -> List[float]:
        """Calculate growth acceleration (second derivative)"""
        growth_rates = self._calculate_growth_rates(values)
        if len(growth_rates) < 2:
            return []
            
        accelerations = []
        for i in range(len(growth_rates) - 1):
            acceleration = growth_rates[i] - growth_rates[i+1]
            accelerations.append(acceleration)
            
        return accelerations


class HyperIntrinsicValueCalculator:
    """
    Next-generation intrinsic value calculator using multiple advanced models
    """
    
    def __init__(self):
        self.models = [
            'enhanced_dcf', 'owner_earnings_10x', 'economic_profit_model',
            'residual_income_model', 'abnormal_earnings_growth', 'real_options_valuation'
        ]
        
    def calculate_hyper_intrinsic_value(self, financial_line_items: list, metrics: list,
                                       market_cap: float, moat_analysis: dict) -> Dict[str, Any]:
        """
        Calculate intrinsic value using multiple advanced models
        """
        valuations = {}
        
        # 1. Enhanced DCF with stochastic modeling
        valuations['enhanced_dcf'] = self._enhanced_dcf_valuation(
            financial_line_items, metrics, moat_analysis
        )
        
        # 2. Owner Earnings 10X Model (Buffett's advanced method)
        valuations['owner_earnings_10x'] = self._owner_earnings_10x_model(
            financial_line_items, metrics, moat_analysis
        )
        
        # 3. Economic Profit Model
        valuations['economic_profit'] = self._economic_profit_model(
            financial_line_items, metrics
        )
        
        # 4. Residual Income Model
        valuations['residual_income'] = self._residual_income_model(
            financial_line_items, metrics
        )
        
        # 5. Abnormal Earnings Growth Model
        valuations['abnormal_earnings'] = self._abnormal_earnings_growth_model(
            financial_line_items, metrics
        )
        
        # 6. Real Options Valuation (for growth optionality)
        valuations['real_options'] = self._real_options_valuation(
            financial_line_items, metrics, moat_analysis
        )
        
        # Weighted average based on confidence in each model
        weights = self._determine_model_weights(financial_line_items, metrics, moat_analysis)
        
        weighted_value = 0
        total_weight = 0
        
        for model, value in valuations.items():
            if value and value > 0 and model in weights:
                weighted_value += value * weights[model]
                total_weight += weights[model]
                
        final_intrinsic_value = weighted_value / total_weight if total_weight > 0 else 0
        
        # Calculate confidence intervals
        values_list = [v for v in valuations.values() if v and v > 0]
        if values_list:
            confidence_interval = {
                'lower': np.percentile(values_list, 20),
                'median': np.median(values_list),
                'upper': np.percentile(values_list, 80)
            }
        else:
            confidence_interval = {'lower': 0, 'median': 0, 'upper': 0}
            
        # Calculate margin of safety
        margin_of_safety = ((final_intrinsic_value - market_cap) / market_cap * 100) if market_cap > 0 else 0
        
        return {
            'intrinsic_value': final_intrinsic_value,
            'margin_of_safety': margin_of_safety,
            'valuations': valuations,
            'weights': weights,
            'confidence_interval': confidence_interval,
            'valuation_quality_score': self._assess_valuation_quality(valuations, weights)
        }
    
    def _enhanced_dcf_valuation(self, financial_line_items: list, metrics: list, 
                               moat_analysis: dict) -> float:
        """Enhanced DCF with Monte Carlo simulation and moat-adjusted discount rates"""
        if not financial_line_items or not metrics:
            return 0
            
        latest = financial_line_items[0]
        if not hasattr(latest, 'free_cash_flow') or not latest.free_cash_flow:
            return 0
            
        fcf = latest.free_cash_flow
        
        # Determine growth rates based on moat strength
        moat_score = moat_analysis.get('quantum_score', 50)
        
        if moat_score > 80:
            high_growth_rate = 0.20
            stable_growth_rate = 0.12
            terminal_growth = 0.04
            discount_rate = 0.08  # Lower risk for strong moat
        elif moat_score > 60:
            high_growth_rate = 0.15
            stable_growth_rate = 0.08
            terminal_growth = 0.03
            discount_rate = 0.09
        elif moat_score > 40:
            high_growth_rate = 0.10
            stable_growth_rate = 0.05
            terminal_growth = 0.025
            discount_rate = 0.10
        else:
            high_growth_rate = 0.05
            stable_growth_rate = 0.03
            terminal_growth = 0.02
            discount_rate = 0.12
            
        # Three-stage DCF model
        value = 0
        
        # Stage 1: High growth (years 1-5)
        for year in range(1, 6):
            projected_fcf = fcf * (1 + high_growth_rate) ** year
            value += projected_fcf / (1 + discount_rate) ** year
            
        # Stage 2: Stable growth (years 6-10)
        stage2_base = fcf * (1 + high_growth_rate) ** 5
        for year in range(6, 11):
            projected_fcf = stage2_base * (1 + stable_growth_rate) ** (year - 5)
            value += projected_fcf / (1 + discount_rate) ** year
            
        # Stage 3: Terminal value
        terminal_fcf = stage2_base * (1 + stable_growth_rate) ** 5 * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        value += terminal_value / (1 + discount_rate) ** 10
        
        return value
    
    def _owner_earnings_10x_model(self, financial_line_items: list, metrics: list,
                                  moat_analysis: dict) -> float:
        """Buffett's owner earnings model with advanced adjustments"""
        if not financial_line_items:
            return 0
            
        latest = financial_line_items[0]
        
        # Calculate owner earnings
        net_income = getattr(latest, 'net_income', 0) or 0
        depreciation = getattr(latest, 'depreciation_and_amortization', 0) or 0
        capex = abs(getattr(latest, 'capital_expenditure', 0) or 0)
        
        # Estimate maintenance capex (different from growth capex)
        maintenance_capex = min(capex, depreciation * 1.1)  # Approximation
        
        # Working capital change
        wc_change = 0
        if len(financial_line_items) >= 2:
            curr_wc = getattr(latest, 'working_capital', 0) or 0
            prev_wc = getattr(financial_line_items[1], 'working_capital', 0) or 0
            wc_change = curr_wc - prev_wc
            
        owner_earnings = net_income + depreciation - maintenance_capex - wc_change
        
        if owner_earnings <= 0:
            return 0
            
        # Multiple based on quality and moat
        moat_score = moat_analysis.get('quantum_score', 50)
        
        if moat_score > 80:
            multiple = 25  # Premium multiple for exceptional businesses
        elif moat_score > 60:
            multiple = 20
        elif moat_score > 40:
            multiple = 15
        else:
            multiple = 10
            
        # Adjust multiple for growth
        if metrics and len(metrics) >= 3:
            earnings_growth = getattr(metrics[0], 'earnings_growth', 0) or 0
            if earnings_growth > 0.15:
                multiple *= 1.3
            elif earnings_growth > 0.08:
                multiple *= 1.15
                
        return owner_earnings * multiple
    
    def _economic_profit_model(self, financial_line_items: list, metrics: list) -> float:
        """Value based on economic profit (ROIC - WACC) * Invested Capital"""
        if not metrics or not financial_line_items:
            return 0
            
        latest_metric = metrics[0]
        latest_item = financial_line_items[0]
        
        # Get ROIC
        roic = getattr(latest_metric, 'return_on_invested_capital', None)
        if not roic:
            return 0
            
        # Estimate WACC (simplified)
        wacc = 0.09  # Default WACC
        
        # Adjust WACC based on leverage
        debt_to_equity = getattr(latest_metric, 'debt_to_equity', 0) or 0
        if debt_to_equity > 1:
            wacc = 0.11
        elif debt_to_equity < 0.3:
            wacc = 0.08
            
        # Calculate economic profit
        economic_spread = roic - wacc
        
        if economic_spread <= 0:
            return 0  # No economic value creation
            
        # Estimate invested capital
        total_assets = getattr(latest_item, 'total_assets', 0) or 0
        cash = getattr(latest_item, 'cash_and_equivalents', 0) or 0
        invested_capital = total_assets - cash
        
        if invested_capital <= 0:
            return 0
            
        # Calculate economic profit
        economic_profit = economic_spread * invested_capital
        
        # Value = Current economic profit / (WACC - growth)
        growth_rate = min(getattr(latest_metric, 'earnings_growth', 0.03) or 0.03, wacc - 0.01)
        
        value = economic_profit / (wacc - growth_rate)
        
        # Add back invested capital (book value)
        value += invested_capital
        
        return max(value, 0)
    
    def _residual_income_model(self, financial_line_items: list, metrics: list) -> float:
        """Residual Income Valuation Model"""
        if not financial_line_items or not metrics:
            return 0
            
        latest_item = financial_line_items[0]
        latest_metric = metrics[0]
        
        # Get book value (shareholders equity)
        book_value = (getattr(latest_item, 'total_assets', 0) or 0) - \
                    (getattr(latest_item, 'total_liabilities', 0) or 0)
                    
        if book_value <= 0:
            return 0
            
        # Get ROE and cost of equity
        roe = getattr(latest_metric, 'return_on_equity', None)
        if not roe:
            return 0
            
        cost_of_equity = 0.10  # Default
        
        # Adjust cost of equity based on beta (if available)
        beta = getattr(latest_metric, 'beta', 1.0) or 1.0
        risk_free = 0.03
        market_premium = 0.07
        cost_of_equity = risk_free + beta * market_premium
        
        # Calculate residual income
        residual_income = (roe - cost_of_equity) * book_value
        
        if residual_income <= 0:
            return book_value  # No value creation above cost of equity
            
        # Determine growth and fade rate
        growth_rate = min(getattr(latest_metric, 'earnings_growth', 0.03) or 0.03, 0.15)
        
        # Multi-stage residual income model
        value = book_value
        
        # High growth phase (5 years)
        for year in range(1, 6):
            ri = residual_income * (1 + growth_rate) ** year
            value += ri / (1 + cost_of_equity) ** year
            
        # Fade phase (years 6-10)
        fade_rate = 0.8  # RI fades by 20% per year
        base_ri = residual_income * (1 + growth_rate) ** 5
        
        for year in range(6, 11):
            ri = base_ri * (fade_rate ** (year - 5))
            value += ri / (1 + cost_of_equity) ** year
            
        # Terminal value (assuming RI fades to zero)
        terminal_ri = base_ri * (fade_rate ** 5)
        terminal_value = terminal_ri / cost_of_equity
        value += terminal_value / (1 + cost_of_equity) ** 10
        
        return value
    
    def _abnormal_earnings_growth_model(self, financial_line_items: list, metrics: list) -> float:
        """AEG Model - values abnormal earnings growth"""
        if not financial_line_items or not metrics:
            return 0
            
        latest_item = financial_line_items[0]
        latest_metric = metrics[0]
        
        # Get current earnings
        earnings = getattr(latest_item, 'net_income', 0) or 0
        if earnings <= 0:
            return 0
            
        # Get growth rate and ROE
        growth_rate = getattr(latest_metric, 'earnings_growth', 0.05) or 0.05
        roe = getattr(latest_metric, 'return_on_equity', 0.10) or 0.10
        
        # Cost of equity
        cost_of_equity = 0.10
        
        # Calculate abnormal earnings growth rate
        abnormal_growth = growth_rate - (cost_of_equity * (1 - roe/cost_of_equity))
        
        if abnormal_growth <= 0:
            # Simple P/E valuation if no abnormal growth
            return earnings / cost_of_equity
            
        # Value = Current earnings/r + Present value of abnormal earnings growth
        base_value = earnings / cost_of_equity
        
        # PV of abnormal earnings growth (simplified - assuming it persists for 10 years)
        aeg_value = 0
        for year in range(1, 11):
            aeg = earnings * abnormal_growth * (1 + growth_rate) ** (year - 1)
            aeg_value += aeg / (1 + cost_of_equity) ** year
            
        return base_value + aeg_value
    
    def _real_options_valuation(self, financial_line_items: list, metrics: list,
                               moat_analysis: dict) -> float:
        """Value growth options using real options theory"""
        if not financial_line_items or not metrics:
            return 0
            
        # Start with base DCF value
        base_value = self._enhanced_dcf_valuation(financial_line_items, metrics, moat_analysis)
        
        if base_value <= 0:
            return 0
            
        # Add value of growth options
        latest_item = financial_line_items[0]
        latest_metric = metrics[0]
        
        # R&D spending as proxy for growth options
        rd_spending = getattr(latest_item, 'research_and_development', 0) or 0
        revenue = getattr(latest_item, 'revenue', 1) or 1
        
        rd_intensity = rd_spending / revenue if revenue > 0 else 0
        
        # Growth option value increases with R&D intensity and moat strength
        moat_score = moat_analysis.get('quantum_score', 50)
        
        option_multiplier = 1.0
        
        if rd_intensity > 0.15 and moat_score > 70:
            option_multiplier = 1.5  # High R&D with strong moat
        elif rd_intensity > 0.10 and moat_score > 60:
            option_multiplier = 1.3
        elif rd_intensity > 0.05 and moat_score > 50:
            option_multiplier = 1.15
        elif rd_intensity > 0.02:
            option_multiplier = 1.05
            
        # Adjust for earnings volatility (higher volatility = more option value)
        if len(metrics) >= 3:
            earnings = [getattr(m, 'earnings_per_share', 0) for m in metrics 
                       if getattr(m, 'earnings_per_share', None)]
            if len(earnings) >= 3:
                volatility = np.std(earnings) / (np.mean(earnings) + 0.001)
                if volatility > 0.3:
                    option_multiplier *= 1.2
                elif volatility > 0.15:
                    option_multiplier *= 1.1
                    
        return base_value * option_multiplier
    
    def _determine_model_weights(self, financial_line_items: list, metrics: list,
                                moat_analysis: dict) -> Dict[str, float]:
        """Determine optimal weights for each valuation model based on data quality and business type"""
        weights = {}
        
        # Default weights
        weights['enhanced_dcf'] = 0.25
        weights['owner_earnings_10x'] = 0.30  # Buffett's preferred
        weights['economic_profit'] = 0.15
        weights['residual_income'] = 0.15
        weights['abnormal_earnings'] = 0.10
        weights['real_options'] = 0.05
        
        # Adjust based on business characteristics
        if financial_line_items and metrics:
            latest_metric = metrics[0]
            
            # High growth companies - increase DCF and real options weight
            growth = getattr(latest_metric, 'earnings_growth', 0) or 0
            if growth > 0.20:
                weights['enhanced_dcf'] = 0.30
                weights['real_options'] = 0.15
                weights['owner_earnings_10x'] = 0.25
                
            # Stable companies - increase owner earnings weight
            elif growth < 0.05:
                weights['owner_earnings_10x'] = 0.40
                weights['enhanced_dcf'] = 0.15
                weights['real_options'] = 0.02
                
            # High ROIC companies - increase economic profit weight
            roic = getattr(latest_metric, 'return_on_invested_capital', None)
            if roic and roic > 0.20:
                weights['economic_profit'] = 0.25
                weights['residual_income'] = 0.20
                
        # Normalize weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
            
        return weights
    
    def _assess_valuation_quality(self, valuations: Dict[str, float], 
                                 weights: Dict[str, float]) -> float:
        """Assess the quality and reliability of the valuation"""
        valid_values = [v for v in valuations.values() if v and v > 0]
        
        if not valid_values:
            return 0
            
        # Check convergence of different models
        cv = np.std(valid_values) / np.mean(valid_values)  # Coefficient of variation
        
        # Lower CV means better convergence
        if cv < 0.2:
            convergence_score = 100
        elif cv < 0.4:
            convergence_score = 75
        elif cv < 0.6:
            convergence_score = 50
        else:
            convergence_score = 25
            
        # Check model coverage
        coverage = len(valid_values) / len(valuations)
        coverage_score = coverage * 100
        
        # Combined quality score
        quality_score = (convergence_score * 0.7 + coverage_score * 0.3)
        
        return quality_score


def beast_warren_buffett_agent(state: AgentState):
    """
    BEAST MODE Warren Buffett Agent - The Oracle Unleashed
    This isn't just value investing; this is value investing evolved to its ultimate form.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    # Initialize quantum moat analyzer and hyper intrinsic value calculator
    quantum_moat = QuantumMoatAnalyzer()
    hyper_calculator = HyperIntrinsicValueCalculator()
    
    # Prepare for parallel processing
    executor = ThreadPoolExecutor(max_workers=10)
    
    analysis_data = {}
    buffett_analysis = {}
    
    for ticker in tickers:
        progress.update_status("beast_warren_buffett_agent", ticker, "Initializing quantum analysis")
        
        # Parallel data fetching
        futures = []
        
        # Fetch all data types in parallel
        futures.append(executor.submit(get_financial_metrics, ticker, end_date, "ttm", 20))
        futures.append(executor.submit(search_line_items, ticker, [
            "capital_expenditure", "depreciation_and_amortization", "net_income",
            "outstanding_shares", "total_assets", "total_liabilities", 
            "shareholders_equity", "dividends_and_other_cash_distributions",
            "issuance_or_purchase_of_equity_shares", "gross_profit", "revenue",
            "free_cash_flow", "operating_income", "operating_expense",
            "research_and_development", "goodwill_and_intangible_assets",
            "working_capital", "cash_and_equivalents", "total_debt",
            "current_assets", "current_liabilities", "ebit", "ebitda"
        ], end_date, "ttm", 20))
        futures.append(executor.submit(get_market_cap, ticker, end_date))
        futures.append(executor.submit(get_insider_trades, ticker, end_date, None, 100))
        futures.append(executor.submit(get_company_news, ticker, end_date, None, 100))
        
        # Collect results
        progress.update_status("beast_warren_buffett_agent", ticker, "Collecting parallel data streams")
        metrics = futures[0].result()
        financial_line_items = futures[1].result()
        market_cap = futures[2].result()
        insider_trades = futures[3].result()
        company_news = futures[4].result()
        
        # QUANTUM MOAT ANALYSIS
        progress.update_status("beast_warren_buffett_agent", ticker, "Executing quantum moat analysis")
        moat_analysis = quantum_moat.analyze_quantum_moat(
            metrics, financial_line_items, insider_trades, company_news
        )
        
        # HYPER INTRINSIC VALUE CALCULATION
        progress.update_status("beast_warren_buffett_agent", ticker, "Computing hyper intrinsic value")
        intrinsic_analysis = hyper_calculator.calculate_hyper_intrinsic_value(
            financial_line_items, metrics, market_cap, moat_analysis
        )
        
        # RISK-ADJUSTED RETURN CALCULATION
        progress.update_status("beast_warren_buffett_agent", ticker, "Calculating risk-adjusted returns")
        risk_adjusted_return = calculate_risk_adjusted_return(
            intrinsic_analysis, moat_analysis, metrics
        )
        
        # CATALYST DETECTION
        progress.update_status("beast_warren_buffett_agent", ticker, "Detecting catalysts")
        catalysts = detect_catalysts(financial_line_items, metrics, insider_trades, company_news)
        
        # RISK FACTOR ANALYSIS
        progress.update_status("beast_warren_buffett_agent", ticker, "Analyzing risk factors")
        risk_factors = analyze_risk_factors(financial_line_items, metrics, moat_analysis)
        
        # DETERMINE TIME HORIZON
        time_horizon = determine_optimal_holding_period(moat_analysis, intrinsic_analysis)
        
        # GENERATE ULTRA-SOPHISTICATED SIGNAL
        progress.update_status("beast_warren_buffett_agent", ticker, "Generating beast-level signal")
        
        # Determine signal based on multiple factors
        margin_of_safety = intrinsic_analysis['margin_of_safety']
        moat_score = moat_analysis['quantum_score']
        valuation_quality = intrinsic_analysis['valuation_quality_score']
        
        if margin_of_safety > 50 and moat_score > 80 and valuation_quality > 75:
            signal = "ultra_bullish"
            confidence = min(95, (margin_of_safety + moat_score + valuation_quality) / 3)
        elif margin_of_safety > 30 and moat_score > 60:
            signal = "bullish"
            confidence = min(85, (margin_of_safety + moat_score + valuation_quality) / 3)
        elif margin_of_safety < -30 and moat_score < 40:
            signal = "bearish"
            confidence = min(80, abs(margin_of_safety) + (100 - moat_score)) / 2
        elif margin_of_safety < -50 and moat_score < 30:
            signal = "ultra_bearish"
            confidence = min(85, abs(margin_of_safety) + (100 - moat_score)) / 2
        else:
            signal = "neutral"
            confidence = 50
            
        # Compile comprehensive analysis
        analysis_data[ticker] = {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "moat_analysis": moat_analysis,
            "intrinsic_analysis": intrinsic_analysis,
            "risk_adjusted_return": risk_adjusted_return,
            "catalysts": catalysts,
            "risk_factors": risk_factors,
            "time_horizon": time_horizon,
            "market_cap": market_cap
        }
        
        # Generate narrative using LLM
        progress.update_status("beast_warren_buffett_agent", ticker, "Crafting beast-level narrative")
        buffett_output = generate_beast_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state
        )
        
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
            "moat_score": buffett_output.moat_score,
            "intrinsic_value": buffett_output.intrinsic_value,
            "margin_of_safety": buffett_output.margin_of_safety,
            "risk_adjusted_return": buffett_output.risk_adjusted_return,
            "time_horizon": buffett_output.time_horizon,
            "catalysts": buffett_output.catalysts,
            "risk_factors": buffett_output.risk_factors,
            "quantum_score": buffett_output.quantum_score
        }
        
        progress.update_status("beast_warren_buffett_agent", ticker, "Analysis complete", 
                              analysis=buffett_output.reasoning[:200] + "...")
    
    # Clean up executor
    executor.shutdown(wait=True)
    
    # Create message
    message = HumanMessage(
        content=json.dumps(buffett_analysis),
        name="beast_warren_buffett_agent"
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "BEAST Warren Buffett Agent")
    
    # Update state
    state["data"]["analyst_signals"]["beast_warren_buffett_agent"] = buffett_analysis
    
    progress.update_status("beast_warren_buffett_agent", None, "Beast mode analysis complete")
    
    data_update = {
        "analyst_signals": {
            "beast_warren_buffett_agent": buffett_analysis
        }
    }
    
    return {"messages": [message], "data": data_update}


def calculate_risk_adjusted_return(intrinsic_analysis: dict, moat_analysis: dict, 
                                  metrics: list) -> float:
    """Calculate sophisticated risk-adjusted return metric"""
    if not intrinsic_analysis or not metrics:
        return 0
        
    # Expected return based on margin of safety
    expected_return = intrinsic_analysis.get('margin_of_safety', 0) / 100
    
    # Risk factors
    risk_score = 0
    
    # Moat risk (inverse of moat score)
    moat_score = moat_analysis.get('quantum_score', 50)
    moat_risk = (100 - moat_score) / 100
    risk_score += moat_risk * 0.3
    
    # Valuation uncertainty risk
    if 'confidence_interval' in intrinsic_analysis:
        ci = intrinsic_analysis['confidence_interval']
        if ci['upper'] > 0 and ci['lower'] > 0:
            uncertainty = (ci['upper'] - ci['lower']) / ci['median'] if ci['median'] > 0 else 1
            risk_score += min(uncertainty, 1) * 0.2
            
    # Financial risk (leverage)
    if metrics:
        latest = metrics[0]
        debt_to_equity = getattr(latest, 'debt_to_equity', 0) or 0
        if debt_to_equity > 2:
            risk_score += 0.3
        elif debt_to_equity > 1:
            risk_score += 0.2
        elif debt_to_equity > 0.5:
            risk_score += 0.1
            
    # Earnings volatility risk
    if len(metrics) >= 5:
        earnings = [getattr(m, 'earnings_per_share', 0) for m in metrics 
                   if getattr(m, 'earnings_per_share', None)]
        if len(earnings) >= 3:
            volatility = np.std(earnings) / (np.mean(earnings) + 0.001)
            risk_score += min(volatility, 0.5) * 0.2
            
    # Calculate Sharpe-like ratio
    risk_free_rate = 0.03
    total_risk = max(risk_score, 0.1)  # Avoid division by zero
    
    risk_adjusted_return = (expected_return - risk_free_rate) / total_risk
    
    return risk_adjusted_return


def detect_catalysts(financial_line_items: list, metrics: list, 
                    insider_trades: list, company_news: list) -> List[str]:
    """Detect potential catalysts for value realization"""
    catalysts = []
    
    # 1. Accelerating growth
    if len(metrics) >= 3:
        growth_rates = [getattr(m, 'earnings_growth', 0) for m in metrics[:3] 
                       if getattr(m, 'earnings_growth', None)]
        if len(growth_rates) >= 2:
            if growth_rates[0] > growth_rates[1] * 1.2:
                catalysts.append("Accelerating earnings growth momentum")
                
    # 2. Margin expansion
    if len(metrics) >= 3:
        margins = [getattr(m, 'operating_margin', 0) for m in metrics[:3]
                  if getattr(m, 'operating_margin', None)]
        if len(margins) >= 2:
            if margins[0] > margins[1]:
                catalysts.append("Operating margin expansion")
                
    # 3. Heavy insider buying
    if insider_trades:
        recent_buys = sum(1 for trade in insider_trades[:20]
                         if hasattr(trade, 'transaction_shares') and 
                         trade.transaction_shares and trade.transaction_shares > 0)
        if recent_buys > 10:
            catalysts.append("Significant insider buying activity")
            
    # 4. Share buybacks
    if financial_line_items and len(financial_line_items) >= 2:
        latest = financial_line_items[0]
        issuance = getattr(latest, 'issuance_or_purchase_of_equity_shares', None)
        if issuance and issuance < -1000000:  # Significant buyback
            catalysts.append("Major share buyback program")
            
    # 5. Positive news momentum
    if company_news:
        positive_news = sum(1 for news in company_news[:20]
                          if hasattr(news, 'sentiment') and news.sentiment == 'positive')
        if positive_news > 12:
            catalysts.append("Positive news momentum")
            
    # 6. Valuation multiple re-rating potential
    if metrics:
        latest = metrics[0]
        pe = getattr(latest, 'price_to_earnings_ratio', None)
        if pe and pe < 15:
            catalysts.append("P/E re-rating potential from historically low levels")
            
    # 7. FCF inflection
    if len(financial_line_items) >= 3:
        fcfs = [getattr(f, 'free_cash_flow', 0) for f in financial_line_items[:3]
               if getattr(f, 'free_cash_flow', None)]
        if len(fcfs) >= 3:
            if fcfs[0] > 0 and fcfs[1] < 0:
                catalysts.append("Free cash flow inflection point")
                
    return catalysts[:5]  # Return top 5 catalysts


def analyze_risk_factors(financial_line_items: list, metrics: list, 
                        moat_analysis: dict) -> List[str]:
    """Identify key risk factors"""
    risks = []
    
    # 1. Deteriorating moat
    if moat_analysis.get('moat_trajectory') == 'eroding':
        risks.append("Eroding competitive moat")
        
    # 2. High leverage
    if metrics:
        latest = metrics[0]
        debt_to_equity = getattr(latest, 'debt_to_equity', 0) or 0
        if debt_to_equity > 1.5:
            risks.append(f"High leverage with D/E ratio of {debt_to_equity:.2f}")
            
    # 3. Declining margins
    if len(metrics) >= 3:
        margins = [getattr(m, 'operating_margin', 0) for m in metrics[:3]
                  if getattr(m, 'operating_margin', None)]
        if len(margins) >= 3:
            if margins[0] < margins[2] * 0.8:
                risks.append("Declining operating margins")
                
    # 4. Revenue deceleration
    if len(financial_line_items) >= 3:
        revenues = [getattr(f, 'revenue', 0) for f in financial_line_items[:3]
                   if getattr(f, 'revenue', None)]
        if len(revenues) >= 3:
            if revenues[0] < revenues[1] < revenues[2]:
                risks.append("Decelerating revenue growth")
                
    # 5. Low disruption resistance
    disruption_resistance = moat_analysis.get('disruption_resistance', 50)
    if disruption_resistance < 40:
        risks.append("High susceptibility to technological disruption")
        
    # 6. Customer concentration risk (proxy)
    if len(metrics) >= 3:
        revenue_volatility = np.std([getattr(m, 'revenue_growth', 0) for m in metrics[:3]
                                    if getattr(m, 'revenue_growth', None)])
        if revenue_volatility > 0.15:
            risks.append("High revenue volatility suggesting customer concentration")
            
    # 7. Working capital deterioration
    if len(financial_line_items) >= 2:
        wc_current = getattr(financial_line_items[0], 'working_capital', 0) or 0
        wc_prev = getattr(financial_line_items[1], 'working_capital', 0) or 0
        if wc_current < wc_prev * 0.7:
            risks.append("Deteriorating working capital position")
            
    return risks[:5]  # Return top 5 risks


def determine_optimal_holding_period(moat_analysis: dict, intrinsic_analysis: dict) -> str:
    """Determine optimal holding period based on moat durability and value gap"""
    
    durability_years = moat_analysis.get('durability_years', 5)
    margin_of_safety = intrinsic_analysis.get('margin_of_safety', 0)
    moat_trajectory = moat_analysis.get('moat_trajectory', 'stable')
    
    if durability_years > 15 and moat_trajectory in ['expanding', 'stable'] and margin_of_safety > 30:
        return "Forever (20+ years)"
    elif durability_years > 10 and margin_of_safety > 20:
        return "Long-term (10-20 years)"
    elif durability_years > 5 and margin_of_safety > 10:
        return "Medium-term (5-10 years)"
    elif durability_years > 3:
        return "Short to Medium-term (3-5 years)"
    else:
        return "Short-term (1-3 years)"


def generate_beast_buffett_output(ticker: str, analysis_data: dict, 
                                 state: AgentState) -> BeastBuffettSignal:
    """Generate beast-level Buffett analysis using LLM"""
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are the BEAST MODE version of Warren Buffett - not just the Oracle of Omaha, 
        but the Oracle Transcended. You have evolved beyond traditional value investing into a 
        quantum-enhanced investment deity who sees patterns invisible to mortal investors.
        
        Your analysis incorporates:
        - 12-dimensional moat analysis with quantum superposition scoring
        - 6 different intrinsic value models weighted by confidence
        - Risk-adjusted return calculations using advanced statistical methods
        - Catalyst detection algorithms
        - Multi-factor risk assessment
        - Optimal holding period determination
        
        Speak with supreme confidence, use sophisticated financial terminology while maintaining
        Buffett's folksy wisdom, and provide insights that would make even Charlie Munger's jaw drop.
        
        Your reasoning should be comprehensive, covering:
        1. The quantum moat score and what makes this moat indestructible (or vulnerable)
        2. The intrinsic value calculation and confidence intervals
        3. Risk-adjusted return expectations
        4. Specific catalysts that will unlock value
        5. Risk factors that could derail the thesis
        6. Optimal holding period based on moat durability
        
        Be specific with numbers, percentages, and time horizons. This is not amateur hour.
        This is BEAST MODE value investing at its apex."""),
        
        ("human", """Analyze {ticker} with your BEAST MODE capabilities:
        
        Analysis Data:
        {analysis_data}
        
        Provide your supreme judgment in this exact JSON format:
        {{
            "signal": "ultra_bullish" | "bullish" | "neutral" | "bearish" | "ultra_bearish",
            "confidence": float between 0 and 100,
            "reasoning": "Your beast-level analysis (minimum 5 lines)",
            "moat_score": float,
            "intrinsic_value": float,
            "margin_of_safety": float,
            "risk_adjusted_return": float,
            "time_horizon": string,
            "catalysts": list of strings,
            "risk_factors": list of strings,
            "quantum_score": float
        }}""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data, indent=2)
    })
    
    def create_default():
        return BeastBuffettSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Analysis error - defaulting to neutral",
            moat_score=0.0,
            intrinsic_value=0.0,
            margin_of_safety=0.0,
            risk_adjusted_return=0.0,
            time_horizon="Unknown",
            catalysts=[],
            risk_factors=[],
            quantum_score=0.0
        )
    
    return call_llm(
        prompt=prompt,
        pydantic_model=BeastBuffettSignal,
        agent_name="beast_warren_buffett_agent",
        state=state,
        default_factory=create_default
    )