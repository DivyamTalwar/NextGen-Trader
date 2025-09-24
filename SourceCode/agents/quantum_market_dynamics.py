"""
QUANTUM MARKET DYNAMICS AGENT
=============================
This agent uses quantum-inspired algorithms and chaos theory to detect hidden market patterns,
phase transitions, and non-linear dynamics that traditional analysis cannot capture.
It operates on the principle that markets are complex adaptive systems with emergent behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from tools.api import get_prices, get_financial_metrics, get_company_news, prices_to_df
from utils.llm import call_llm
from utils.progress import progress
from scipy import stats, signal, optimize
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


class QuantumMarketSignal(BaseModel):
    """Signal from quantum market dynamics analysis"""
    signal: Literal["quantum_surge", "bullish", "neutral", "bearish", "quantum_collapse"]
    confidence: float = Field(ge=0, le=100)
    reasoning: str
    quantum_state: str
    phase_transition_probability: float
    market_regime: str
    fractal_dimension: float
    entropy_level: float
    coherence_score: float
    critical_points: List[Dict[str, Any]]
    hidden_patterns: List[str]
    emergence_indicators: List[str]


class QuantumMarketAnalyzer:
    """
    Advanced market analysis using quantum mechanics principles and chaos theory
    """
    
    def __init__(self):
        self.quantum_states = [
            "superposition", "entangled", "coherent", "decoherent", 
            "collapsed", "tunneling", "excited", "ground"
        ]
        self.market_regimes = [
            "trending", "mean_reverting", "chaotic", "phase_transition",
            "bifurcation", "strange_attractor", "fractal", "quantum_foam"
        ]
        
    def analyze_quantum_dynamics(self, prices_df: pd.DataFrame, metrics: list,
                                news: list, ticker: str) -> Dict[str, Any]:
        """
        Perform quantum-inspired market dynamics analysis
        """
        analysis = {}
        
        # 1. Quantum State Detection
        progress.update_status("quantum_market_dynamics_agent", ticker, "Detecting quantum state")
        quantum_state = self._detect_quantum_state(prices_df)
        analysis['quantum_state'] = quantum_state
        
        # 2. Phase Transition Analysis
        progress.update_status("quantum_market_dynamics_agent", ticker, "Analyzing phase transitions")
        phase_analysis = self._analyze_phase_transitions(prices_df)
        analysis['phase_transitions'] = phase_analysis
        
        # 3. Market Regime Classification
        progress.update_status("quantum_market_dynamics_agent", ticker, "Classifying market regime")
        regime = self._classify_market_regime(prices_df)
        analysis['market_regime'] = regime
        
        # 4. Fractal Dimension Calculation
        progress.update_status("quantum_market_dynamics_agent", ticker, "Computing fractal dimension")
        fractal_dim = self._calculate_fractal_dimension(prices_df)
        analysis['fractal_dimension'] = fractal_dim
        
        # 5. Market Entropy Analysis
        progress.update_status("quantum_market_dynamics_agent", ticker, "Measuring market entropy")
        entropy = self._calculate_market_entropy(prices_df)
        analysis['entropy'] = entropy
        
        # 6. Quantum Coherence Score
        progress.update_status("quantum_market_dynamics_agent", ticker, "Calculating quantum coherence")
        coherence = self._calculate_quantum_coherence(prices_df)
        analysis['coherence'] = coherence
        
        # 7. Critical Point Detection
        progress.update_status("quantum_market_dynamics_agent", ticker, "Detecting critical points")
        critical_points = self._detect_critical_points(prices_df)
        analysis['critical_points'] = critical_points
        
        # 8. Hidden Pattern Discovery
        progress.update_status("quantum_market_dynamics_agent", ticker, "Discovering hidden patterns")
        hidden_patterns = self._discover_hidden_patterns(prices_df, metrics)
        analysis['hidden_patterns'] = hidden_patterns
        
        # 9. Emergence Indicator Detection
        progress.update_status("quantum_market_dynamics_agent", ticker, "Detecting emergence indicators")
        emergence = self._detect_emergence_indicators(prices_df, news)
        analysis['emergence_indicators'] = emergence
        
        # 10. Quantum Entanglement Analysis
        progress.update_status("quantum_market_dynamics_agent", ticker, "Analyzing quantum entanglement")
        entanglement = self._analyze_quantum_entanglement(prices_df)
        analysis['entanglement'] = entanglement
        
        # 11. Wave Function Analysis
        progress.update_status("quantum_market_dynamics_agent", ticker, "Analyzing wave functions")
        wave_analysis = self._analyze_wave_function(prices_df)
        analysis['wave_function'] = wave_analysis
        
        # 12. Non-linear Dynamics
        progress.update_status("quantum_market_dynamics_agent", ticker, "Computing non-linear dynamics")
        nonlinear = self._analyze_nonlinear_dynamics(prices_df)
        analysis['nonlinear_dynamics'] = nonlinear
        
        return analysis
    
    def _detect_quantum_state(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect the quantum state of the market"""
        if len(prices_df) < 50:
            return {"state": "unknown", "confidence": 0}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Calculate various quantum-inspired metrics
        
        # Superposition: Multiple trend states coexisting
        short_trend = returns.rolling(5).mean().iloc[-1]
        medium_trend = returns.rolling(20).mean().iloc[-1]
        long_trend = returns.rolling(50).mean().iloc[-1]
        
        trends = [short_trend, medium_trend, long_trend]
        trend_disagreement = np.std(trends) / (np.mean(np.abs(trends)) + 1e-10)
        
        # Entanglement: Correlation patterns
        if len(prices_df) >= 100:
            correlation_matrix = self._calculate_rolling_correlations(prices_df)
            entanglement_score = np.mean(np.abs(correlation_matrix))
        else:
            entanglement_score = 0.5
            
        # Coherence: Price action consistency
        coherence = 1 - (returns.std() / (returns.abs().mean() + 1e-10))
        
        # Determine quantum state
        if trend_disagreement > 0.5 and entanglement_score > 0.6:
            state = "superposition"
            confidence = 80
        elif entanglement_score > 0.7:
            state = "entangled"
            confidence = 75
        elif coherence > 0.7:
            state = "coherent"
            confidence = 70
        elif coherence < 0.3:
            state = "decoherent"
            confidence = 65
        elif abs(short_trend) > abs(long_trend) * 2:
            state = "excited"
            confidence = 60
        else:
            state = "ground"
            confidence = 55
            
        return {
            "state": state,
            "confidence": confidence,
            "metrics": {
                "trend_disagreement": trend_disagreement,
                "entanglement_score": entanglement_score,
                "coherence": coherence
            }
        }
    
    def _analyze_phase_transitions(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze market phase transitions"""
        if len(prices_df) < 100:
            return {"probability": 0, "indicators": []}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Calculate order parameters
        volatility = returns.rolling(20).std()
        volume = prices_df['volume']
        
        # Detect sudden changes in volatility (phase transition indicator)
        vol_change = volatility.pct_change()
        vol_spikes = vol_change[abs(vol_change) > 2].index
        
        # Detect volume surges
        volume_ma = volume.rolling(20).mean()
        volume_spikes = volume[volume > volume_ma * 2].index
        
        # Critical slowing down detection
        autocorr = self._calculate_autocorrelation(returns)
        critical_slowing = autocorr.rolling(10).mean().iloc[-1] > 0.7
        
        # Calculate phase transition probability
        indicators = []
        probability = 0
        
        if len(vol_spikes) > 0 and (prices_df.index[-1] - vol_spikes[-1]).days < 5:
            indicators.append("Recent volatility spike")
            probability += 25
            
        if len(volume_spikes) > 0 and (prices_df.index[-1] - volume_spikes[-1]).days < 5:
            indicators.append("Recent volume surge")
            probability += 20
            
        if critical_slowing:
            indicators.append("Critical slowing down detected")
            probability += 30
            
        # Bifurcation analysis
        bifurcation = self._detect_bifurcation(returns)
        if bifurcation['detected']:
            indicators.append(f"Bifurcation pattern: {bifurcation['type']}")
            probability += 25
            
        return {
            "probability": min(probability, 95),
            "indicators": indicators,
            "critical_slowing": critical_slowing,
            "bifurcation": bifurcation
        }
    
    def _classify_market_regime(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Classify the current market regime using advanced techniques"""
        if len(prices_df) < 50:
            return {"regime": "unknown", "confidence": 0}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Calculate regime indicators
        
        # Trending vs Mean Reverting
        hurst = self._calculate_hurst_exponent(prices_df['close'].values)
        
        # Chaos indicators
        lyapunov = self._estimate_lyapunov_exponent(returns.values)
        
        # Fractal characteristics
        fractal_dim = self._calculate_fractal_dimension(prices_df)
        
        # Strange attractor detection
        attractor = self._detect_strange_attractor(returns.values)
        
        # Determine regime
        if hurst > 0.6 and lyapunov < 0:
            regime = "trending"
            confidence = 75
        elif hurst < 0.4:
            regime = "mean_reverting"
            confidence = 70
        elif lyapunov > 0.1:
            regime = "chaotic"
            confidence = 80
        elif attractor['detected']:
            regime = "strange_attractor"
            confidence = 85
        elif fractal_dim > 1.5:
            regime = "fractal"
            confidence = 65
        else:
            regime = "quantum_foam"  # Highly uncertain state
            confidence = 50
            
        return {
            "regime": regime,
            "confidence": confidence,
            "indicators": {
                "hurst_exponent": hurst,
                "lyapunov_exponent": lyapunov,
                "fractal_dimension": fractal_dim,
                "attractor_detected": attractor['detected']
            }
        }
    
    def _calculate_fractal_dimension(self, prices_df: pd.DataFrame) -> float:
        """Calculate the fractal dimension using box-counting method"""
        if len(prices_df) < 50:
            return 1.5
            
        prices = prices_df['close'].values
        
        # Normalize prices
        prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        
        # Box counting algorithm
        box_sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for box_size in box_sizes:
            # Discretize the price series
            n_boxes = len(prices_norm) // box_size
            if n_boxes < 2:
                continue
                
            boxes_occupied = set()
            for i in range(n_boxes):
                segment = prices_norm[i*box_size:(i+1)*box_size]
                if len(segment) > 0:
                    # Determine which box the segment occupies
                    box_y = int(segment.mean() * box_size)
                    boxes_occupied.add((i, box_y))
                    
            counts.append(len(boxes_occupied))
            
        if len(counts) < 2:
            return 1.5
            
        # Calculate fractal dimension
        try:
            coeffs = np.polyfit(np.log(box_sizes[:len(counts)]), np.log(counts), 1)
            fractal_dim = -coeffs[0]
            return max(1.0, min(2.0, fractal_dim))  # Bound between 1 and 2
        except:
            return 1.5
    
    def _calculate_market_entropy(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various entropy measures of the market"""
        if len(prices_df) < 50:
            return {"shannon": 0, "approximate": 0, "sample": 0}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Shannon entropy
        shannon_entropy = self._shannon_entropy(returns)
        
        # Approximate entropy
        approx_entropy = self._approximate_entropy(returns.values)
        
        # Sample entropy
        sample_entropy = self._sample_entropy(returns.values)
        
        # Normalized total entropy
        total_entropy = (shannon_entropy + approx_entropy + sample_entropy) / 3
        
        return {
            "shannon": shannon_entropy,
            "approximate": approx_entropy,
            "sample": sample_entropy,
            "total": total_entropy,
            "interpretation": self._interpret_entropy(total_entropy)
        }
    
    def _calculate_quantum_coherence(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quantum coherence of price movements"""
        if len(prices_df) < 50:
            return {"score": 0, "decoherence_time": 0}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Calculate coherence through autocorrelation decay
        autocorr = self._calculate_autocorrelation(returns)
        
        # Find decoherence time (when autocorrelation drops below threshold)
        threshold = 0.1
        decoherence_time = 0
        for i, val in enumerate(autocorr):
            if abs(val) < threshold:
                decoherence_time = i
                break
                
        # Coherence score based on how long correlations persist
        coherence_score = min(decoherence_time / 20, 1.0) * 100
        
        # Phase coherence through Hilbert transform
        phase_coherence = self._calculate_phase_coherence(prices_df['close'].values)
        
        return {
            "score": coherence_score,
            "decoherence_time": decoherence_time,
            "phase_coherence": phase_coherence,
            "quality": "high" if coherence_score > 70 else "medium" if coherence_score > 40 else "low"
        }
    
    def _detect_critical_points(self, prices_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect critical points where system behavior changes dramatically"""
        critical_points = []
        
        if len(prices_df) < 100:
            return critical_points
            
        prices = prices_df['close'].values
        returns = prices_df['close'].pct_change().dropna()
        
        # 1. Detect tipping points using early warning signals
        ews = self._calculate_early_warning_signals(returns)
        for date, signal in ews:
            critical_points.append({
                "type": "tipping_point",
                "date": date,
                "strength": signal,
                "description": "Potential regime shift detected"
            })
            
        # 2. Detect bifurcation points
        bifurcations = self._detect_bifurcation_points(returns)
        for date, bif_type in bifurcations:
            critical_points.append({
                "type": "bifurcation",
                "date": date,
                "subtype": bif_type,
                "description": f"{bif_type} bifurcation detected"
            })
            
        # 3. Detect phase transition points
        phase_transitions = self._detect_phase_transition_points(prices_df)
        for date, transition in phase_transitions:
            critical_points.append({
                "type": "phase_transition",
                "date": date,
                "strength": transition,
                "description": "Market phase transition detected"
            })
            
        # Sort by date and return most recent
        critical_points.sort(key=lambda x: x['date'], reverse=True)
        return critical_points[:5]
    
    def _discover_hidden_patterns(self, prices_df: pd.DataFrame, metrics: list) -> List[str]:
        """Discover hidden patterns using advanced pattern recognition"""
        patterns = []
        
        if len(prices_df) < 100:
            return patterns
            
        returns = prices_df['close'].pct_change().dropna()
        
        # 1. Frequency domain analysis
        freq_patterns = self._analyze_frequency_domain(prices_df['close'].values)
        for pattern in freq_patterns:
            patterns.append(f"Cyclical pattern: {pattern}")
            
        # 2. Chaos patterns
        if self._detect_chaos_patterns(returns.values):
            patterns.append("Chaotic attractor detected in price dynamics")
            
        # 3. Power law distributions
        if self._detect_power_law(returns.values):
            patterns.append("Power law distribution in returns (scale-free behavior)")
            
        # 4. Long-range correlations
        if self._detect_long_range_correlations(returns.values):
            patterns.append("Long-range correlations detected (market memory)")
            
        # 5. Multifractal patterns
        if self._detect_multifractal(prices_df['close'].values):
            patterns.append("Multifractal structure in price series")
            
        # 6. Hidden Markov patterns
        hmm_states = self._detect_hidden_markov_states(returns.values)
        if hmm_states:
            patterns.append(f"Hidden Markov states: {hmm_states}")
            
        # 7. Anomalous diffusion
        if self._detect_anomalous_diffusion(returns.values):
            patterns.append("Anomalous diffusion pattern (non-Brownian motion)")
            
        return patterns[:7]
    
    def _detect_emergence_indicators(self, prices_df: pd.DataFrame, news: list) -> List[str]:
        """Detect indicators of emergent market behavior"""
        indicators = []
        
        if len(prices_df) < 50:
            return indicators
            
        # 1. Synchronization emergence
        if self._detect_synchronization(prices_df):
            indicators.append("Market synchronization emerging")
            
        # 2. Collective behavior
        if self._detect_collective_behavior(prices_df):
            indicators.append("Collective behavior patterns detected")
            
        # 3. Self-organization
        if self._detect_self_organization(prices_df):
            indicators.append("Self-organizing criticality observed")
            
        # 4. Information cascade
        if news and self._detect_information_cascade(news):
            indicators.append("Information cascade in progress")
            
        # 5. Feedback loops
        feedback = self._detect_feedback_loops(prices_df)
        if feedback['positive']:
            indicators.append("Positive feedback loop amplifying moves")
        if feedback['negative']:
            indicators.append("Negative feedback loop dampening volatility")
            
        # 6. Emergence of new equilibrium
        if self._detect_new_equilibrium(prices_df):
            indicators.append("New equilibrium state emerging")
            
        return indicators
    
    def _analyze_quantum_entanglement(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quantum entanglement between price and volume"""
        if len(prices_df) < 50:
            return {"entanglement_score": 0, "correlation_matrix": []}
            
        # Normalize data
        price_norm = (prices_df['close'] - prices_df['close'].mean()) / prices_df['close'].std()
        volume_norm = (prices_df['volume'] - prices_df['volume'].mean()) / (prices_df['volume'].std() + 1e-10)
        
        # Calculate quantum correlation
        correlation = price_norm.rolling(20).corr(volume_norm)
        
        # Mutual information
        mutual_info = self._calculate_mutual_information(price_norm.values, volume_norm.values)
        
        # Entanglement score
        entanglement_score = (abs(correlation.mean()) + mutual_info) / 2 * 100
        
        return {
            "entanglement_score": min(entanglement_score, 100),
            "price_volume_correlation": correlation.iloc[-1] if not correlation.empty else 0,
            "mutual_information": mutual_info,
            "interpretation": "strongly entangled" if entanglement_score > 70 else "moderately entangled" if entanglement_score > 40 else "weakly entangled"
        }
    
    def _analyze_wave_function(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market wave function properties"""
        if len(prices_df) < 100:
            return {"amplitude": 0, "frequency": 0, "phase": 0}
            
        prices = prices_df['close'].values
        
        # Fourier transform to get wave components
        fft_values = fft(prices)
        frequencies = fftfreq(len(prices))
        
        # Find dominant frequency
        positive_freq_idx = frequencies > 0
        magnitudes = np.abs(fft_values[positive_freq_idx])
        
        if len(magnitudes) > 0:
            dominant_idx = np.argmax(magnitudes)
            dominant_frequency = frequencies[positive_freq_idx][dominant_idx]
            dominant_amplitude = magnitudes[dominant_idx]
            dominant_phase = np.angle(fft_values[positive_freq_idx][dominant_idx])
        else:
            dominant_frequency = 0
            dominant_amplitude = 0
            dominant_phase = 0
            
        # Wave packet analysis
        wave_packet = self._analyze_wave_packet(prices)
        
        return {
            "dominant_frequency": dominant_frequency,
            "dominant_amplitude": dominant_amplitude,
            "dominant_phase": dominant_phase,
            "wave_packet": wave_packet,
            "interpretation": self._interpret_wave_function(dominant_frequency, dominant_amplitude)
        }
    
    def _analyze_nonlinear_dynamics(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze non-linear dynamics in the market"""
        if len(prices_df) < 100:
            return {"lyapunov": 0, "dimension": 0, "predictability": 0}
            
        returns = prices_df['close'].pct_change().dropna().values
        
        # Lyapunov exponent
        lyapunov = self._estimate_lyapunov_exponent(returns)
        
        # Correlation dimension
        corr_dimension = self._calculate_correlation_dimension(returns)
        
        # Predictability horizon
        if lyapunov > 0:
            predictability_horizon = 1 / lyapunov
        else:
            predictability_horizon = float('inf')
            
        # Recurrence analysis
        recurrence = self._recurrence_analysis(returns)
        
        return {
            "lyapunov_exponent": lyapunov,
            "correlation_dimension": corr_dimension,
            "predictability_horizon": min(predictability_horizon, 100),
            "recurrence_rate": recurrence,
            "chaos_level": "high" if lyapunov > 0.1 else "medium" if lyapunov > 0 else "low"
        }
    
    # Helper methods
    
    def _calculate_rolling_correlations(self, prices_df: pd.DataFrame) -> np.ndarray:
        """Calculate rolling correlation matrix"""
        returns = prices_df['close'].pct_change().dropna()
        
        # Create lagged returns
        lags = [1, 5, 10, 20]
        corr_matrix = np.zeros((len(lags), len(lags)))
        
        for i, lag1 in enumerate(lags):
            for j, lag2 in enumerate(lags):
                if lag1 <= len(returns) and lag2 <= len(returns):
                    corr_matrix[i, j] = returns.shift(lag1).corr(returns.shift(lag2))
                    
        return corr_matrix
    
    def _calculate_autocorrelation(self, series: pd.Series, max_lag: int = 50) -> pd.Series:
        """Calculate autocorrelation function"""
        autocorr = pd.Series(index=range(max_lag))
        for lag in range(max_lag):
            autocorr[lag] = series.autocorr(lag=lag)
        return autocorr
    
    def _detect_bifurcation(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect bifurcation patterns"""
        if len(returns) < 100:
            return {"detected": False, "type": None}
            
        # Look for period doubling
        autocorr = self._calculate_autocorrelation(returns, 30)
        
        # Check for specific patterns
        if autocorr[2] > 0.7 and autocorr[1] < 0.3:
            return {"detected": True, "type": "period_doubling"}
        elif autocorr[3] > 0.6 and autocorr[1] < 0.2:
            return {"detected": True, "type": "period_tripling"}
        elif np.std(autocorr[:10]) > 0.3:
            return {"detected": True, "type": "chaotic"}
        else:
            return {"detected": False, "type": None}
    
    def _calculate_hurst_exponent(self, series: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(series) < 20:
            return 0.5
            
        lags = range(2, min(20, len(series)//2))
        tau = []
        
        for lag in lags:
            std_dev = []
            for start in range(0, len(series) - lag):
                subseries = series[start:start + lag]
                if len(subseries) > 1:
                    mean = np.mean(subseries)
                    cumsum = np.cumsum(subseries - mean)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(subseries)
                    if S > 0:
                        std_dev.append(R / S)
                        
            if std_dev:
                tau.append(np.mean(std_dev))
                
        if len(tau) < 2:
            return 0.5
            
        try:
            poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def _estimate_lyapunov_exponent(self, series: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent"""
        if len(series) < 50:
            return 0
            
        # Simplified estimation using divergence of nearby trajectories
        embedding_dim = 3
        time_delay = 1
        
        # Create embedded time series
        embedded = []
        for i in range(len(series) - (embedding_dim - 1) * time_delay):
            embedded.append([series[i + j * time_delay] for j in range(embedding_dim)])
            
        embedded = np.array(embedded)
        
        # Find nearest neighbors and track divergence
        divergences = []
        for i in range(len(embedded) - 10):
            # Find nearest neighbor
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf  # Exclude self
            
            nearest_idx = np.argmin(distances)
            initial_distance = distances[nearest_idx]
            
            if initial_distance > 0:
                # Track divergence over time
                for t in range(1, min(10, len(embedded) - max(i, nearest_idx))):
                    distance_t = np.linalg.norm(embedded[i + t] - embedded[nearest_idx + t])
                    if distance_t > 0 and initial_distance > 0:
                        divergences.append(np.log(distance_t / initial_distance) / t)
                        
        if divergences:
            return np.mean(divergences)
        return 0
    
    def _detect_strange_attractor(self, series: np.ndarray) -> Dict[str, Any]:
        """Detect presence of strange attractor"""
        if len(series) < 100:
            return {"detected": False, "type": None}
            
        # Phase space reconstruction
        embedding_dim = 3
        embedded = []
        
        for i in range(len(series) - embedding_dim + 1):
            embedded.append(series[i:i + embedding_dim])
            
        embedded = np.array(embedded)
        
        # Cluster analysis to detect attractor structure
        scaler = StandardScaler()
        embedded_scaled = scaler.fit_transform(embedded)
        
        # DBSCAN to find dense regions
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(embedded_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        if n_clusters > 2 and n_clusters < 10:
            return {"detected": True, "type": f"{n_clusters}-lobe_attractor"}
        else:
            return {"detected": False, "type": None}
    
    def _shannon_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy"""
        # Discretize into bins
        bins = 10
        hist, _ = np.histogram(series, bins=bins)
        prob = hist / len(series)
        
        # Remove zero probabilities
        prob = prob[prob > 0]
        
        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))
        
        # Normalize
        return entropy / np.log2(bins)
    
    def _approximate_entropy(self, series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        N = len(series)
        if N < m + 1:
            return 0
            
        if r is None:
            r = 0.2 * np.std(series)
            
        def _maxdist(x1, x2):
            return max(abs(ua - va) for ua, va in zip(x1, x2))
        
        def _phi(m):
            patterns = [series[i:i + m] for i in range(N - m + 1)]
            C = []
            
            for i, pattern in enumerate(patterns):
                matching = sum(1 for j, p in enumerate(patterns) 
                             if _maxdist(pattern, p) <= r)
                C.append(matching / (N - m + 1))
                
            return sum(np.log(c) for c in C if c > 0) / (N - m + 1)
            
        return _phi(m) - _phi(m + 1)
    
    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        N = len(series)
        if N < m + 1:
            return 0
            
        if r is None:
            r = 0.2 * np.std(series)
            
        def _maxdist(x1, x2):
            return max(abs(ua - va) for ua, va in zip(x1, x2))
        
        def _count_patterns(m):
            patterns = [series[i:i + m] for i in range(N - m + 1)]
            count = 0
            
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        count += 1
                        
            return count
            
        B = _count_patterns(m)
        A = _count_patterns(m + 1)
        
        if B == 0:
            return 0
            
        return -np.log(A / B) if A > 0 else 0
    
    def _interpret_entropy(self, entropy: float) -> str:
        """Interpret entropy level"""
        if entropy > 0.8:
            return "Very high entropy - maximum uncertainty"
        elif entropy > 0.6:
            return "High entropy - significant randomness"
        elif entropy > 0.4:
            return "Moderate entropy - mixed signals"
        elif entropy > 0.2:
            return "Low entropy - some predictability"
        else:
            return "Very low entropy - highly ordered"
    
    def _calculate_phase_coherence(self, series: np.ndarray) -> float:
        """Calculate phase coherence using Hilbert transform"""
        if len(series) < 50:
            return 0
            
        # Hilbert transform to get instantaneous phase
        analytic_signal = signal.hilbert(series - np.mean(series))
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Calculate phase differences
        phase_diff = np.diff(phase)
        
        # Coherence is inverse of phase variance
        coherence = 1 / (1 + np.std(phase_diff))
        
        return min(coherence * 100, 100)
    
    def _calculate_early_warning_signals(self, returns: pd.Series) -> List[Tuple[Any, float]]:
        """Calculate early warning signals for critical transitions"""
        signals = []
        
        if len(returns) < 100:
            return signals
            
        window = 50
        
        for i in range(window, len(returns)):
            subseries = returns.iloc[i-window:i]
            
            # Increasing variance
            var_trend = np.polyfit(range(window), subseries.rolling(10).std().dropna(), 1)[0]
            
            # Increasing autocorrelation
            autocorr_trend = np.polyfit(range(window-1), 
                                       [subseries.autocorr(lag=1) for _ in range(window-1)], 1)[0]
            
            # Critical slowing down
            if var_trend > 0.001 and autocorr_trend > 0.001:
                signal_strength = (var_trend + autocorr_trend) * 1000
                signals.append((returns.index[i], min(signal_strength, 1.0)))
                
        return signals[-5:]  # Return last 5 signals
    
    def _detect_bifurcation_points(self, returns: pd.Series) -> List[Tuple[Any, str]]:
        """Detect bifurcation points in the time series"""
        bifurcations = []
        
        if len(returns) < 100:
            return bifurcations
            
        window = 30
        
        for i in range(window, len(returns) - window):
            before = returns.iloc[i-window:i]
            after = returns.iloc[i:i+window]
            
            # Check for sudden change in distribution
            ks_stat, p_value = stats.ks_2samp(before, after)
            
            if p_value < 0.01:  # Significant change
                # Determine type of bifurcation
                if after.std() > before.std() * 2:
                    bifurcations.append((returns.index[i], "pitchfork"))
                elif abs(after.mean() - before.mean()) > 2 * before.std():
                    bifurcations.append((returns.index[i], "transcritical"))
                else:
                    bifurcations.append((returns.index[i], "saddle_node"))
                    
        return bifurcations[-3:]  # Return last 3
    
    def _detect_phase_transition_points(self, prices_df: pd.DataFrame) -> List[Tuple[Any, float]]:
        """Detect phase transition points"""
        transitions = []
        
        if len(prices_df) < 100:
            return transitions
            
        returns = prices_df['close'].pct_change().dropna()
        volume = prices_df['volume']
        
        window = 20
        
        for i in range(window, len(returns) - window):
            # Volume spike + volatility change
            vol_spike = volume.iloc[i] > volume.iloc[i-window:i].mean() * 2
            
            vol_before = returns.iloc[i-window:i].std()
            vol_after = returns.iloc[i:i+window].std()
            vol_change = abs(vol_after - vol_before) / vol_before if vol_before > 0 else 0
            
            if vol_spike and vol_change > 0.5:
                transitions.append((returns.index[i], vol_change))
                
        return transitions[-3:]
    
    def _analyze_frequency_domain(self, series: np.ndarray) -> List[str]:
        """Analyze frequency domain for cyclical patterns"""
        patterns = []
        
        if len(series) < 100:
            return patterns
            
        # FFT analysis
        fft_values = fft(series)
        frequencies = fftfreq(len(series))
        
        # Find dominant frequencies
        positive_freq_idx = frequencies > 0
        magnitudes = np.abs(fft_values[positive_freq_idx])
        
        if len(magnitudes) > 0:
            # Get top 3 frequencies
            top_indices = np.argsort(magnitudes)[-3:]
            
            for idx in top_indices:
                freq = frequencies[positive_freq_idx][idx]
                period = 1 / freq if freq > 0 else 0
                
                if period > 5 and period < 100:  # Reasonable period range
                    patterns.append(f"{period:.1f}-period cycle detected")
                    
        return patterns
    
    def _detect_chaos_patterns(self, series: np.ndarray) -> bool:
        """Detect chaotic patterns"""
        if len(series) < 100:
            return False
            
        # Check for positive Lyapunov exponent
        lyapunov = self._estimate_lyapunov_exponent(series)
        
        # Check for strange attractor
        attractor = self._detect_strange_attractor(series)
        
        # Check for sensitive dependence on initial conditions
        sensitivity = self._check_sensitivity(series)
        
        return lyapunov > 0 and (attractor['detected'] or sensitivity)
    
    def _detect_power_law(self, series: np.ndarray) -> bool:
        """Detect power law distribution"""
        if len(series) < 100:
            return False
            
        # Calculate return distribution
        abs_returns = np.abs(series)
        abs_returns = abs_returns[abs_returns > 0]  # Remove zeros
        
        if len(abs_returns) < 50:
            return False
            
        # Log-log plot test
        sorted_returns = np.sort(abs_returns)[::-1]
        ranks = np.arange(1, len(sorted_returns) + 1)
        
        # Fit power law
        try:
            log_returns = np.log(sorted_returns)
            log_ranks = np.log(ranks)
            
            # Linear regression in log-log space
            slope, intercept = np.polyfit(log_ranks, log_returns, 1)
            
            # Check if slope is in typical range for power laws
            return -3 < slope < -1
        except:
            return False
    
    def _detect_long_range_correlations(self, series: np.ndarray) -> bool:
        """Detect long-range correlations"""
        if len(series) < 100:
            return False
            
        # Calculate Hurst exponent
        hurst = self._calculate_hurst_exponent(series)
        
        # Hurst > 0.5 indicates long-range positive correlations
        # Hurst < 0.5 indicates long-range negative correlations
        return abs(hurst - 0.5) > 0.1
    
    def _detect_multifractal(self, series: np.ndarray) -> bool:
        """Detect multifractal properties"""
        if len(series) < 200:
            return False
            
        # Simplified multifractal test using different moments
        q_values = [0.5, 1, 2, 3]
        hurst_values = []
        
        for q in q_values:
            # Calculate generalized Hurst exponent for different q
            if q == 0:
                continue
                
            # Use absolute returns raised to power q
            transformed = np.abs(series) ** q
            hurst_q = self._calculate_hurst_exponent(transformed)
            hurst_values.append(hurst_q)
            
        # Check if Hurst varies with q (multifractal)
        if len(hurst_values) > 1:
            return np.std(hurst_values) > 0.1
            
        return False
    
    def _detect_hidden_markov_states(self, series: np.ndarray) -> str:
        """Detect hidden Markov states"""
        if len(series) < 100:
            return ""
            
        # Simple regime detection using volatility clustering
        volatility = pd.Series(series).rolling(10).std()
        
        if volatility.isna().all():
            return ""
            
        # Classify into high/low volatility regimes
        median_vol = volatility.median()
        
        high_vol_periods = (volatility > median_vol * 1.5).sum()
        low_vol_periods = (volatility < median_vol * 0.5).sum()
        
        if high_vol_periods > len(volatility) * 0.3:
            return "High volatility regime dominant"
        elif low_vol_periods > len(volatility) * 0.3:
            return "Low volatility regime dominant"
        else:
            return "Mixed volatility regimes"
    
    def _detect_anomalous_diffusion(self, series: np.ndarray) -> bool:
        """Detect anomalous diffusion patterns"""
        if len(series) < 100:
            return False
            
        # Calculate mean square displacement
        msd = []
        max_lag = min(50, len(series) // 2)
        
        for lag in range(1, max_lag):
            displacements = series[lag:] - series[:-lag]
            msd.append(np.mean(displacements ** 2))
            
        if len(msd) < 10:
            return False
            
        # Fit power law to MSD
        lags = np.arange(1, len(msd) + 1)
        
        try:
            # Log-log fit
            slope, _ = np.polyfit(np.log(lags), np.log(msd), 1)
            
            # Normal diffusion has slope = 1
            # Anomalous if slope != 1
            return abs(slope - 1) > 0.2
        except:
            return False
    
    def _detect_synchronization(self, prices_df: pd.DataFrame) -> bool:
        """Detect market synchronization"""
        if len(prices_df) < 50:
            return False
            
        # Check if price and volume are synchronized
        price_returns = prices_df['close'].pct_change().dropna()
        volume_changes = prices_df['volume'].pct_change().dropna()
        
        # Calculate rolling correlation
        correlation = price_returns.rolling(20).corr(volume_changes)
        
        # High correlation indicates synchronization
        return abs(correlation.mean()) > 0.5
    
    def _detect_collective_behavior(self, prices_df: pd.DataFrame) -> bool:
        """Detect collective behavior patterns"""
        if len(prices_df) < 100:
            return False
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Check for herding (returns clustering)
        clustering_coefficient = self._calculate_clustering(returns.values)
        
        return clustering_coefficient > 0.6
    
    def _detect_self_organization(self, prices_df: pd.DataFrame) -> bool:
        """Detect self-organizing criticality"""
        if len(prices_df) < 100:
            return False
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Check for scale-invariant avalanches (large moves)
        large_moves = abs(returns) > returns.std() * 2
        
        # Check if large moves follow power law
        if large_moves.sum() > 10:
            return self._detect_power_law(returns[large_moves].values)
            
        return False
    
    def _detect_information_cascade(self, news: list) -> bool:
        """Detect information cascade from news"""
        if not news or len(news) < 20:
            return False
            
        # Check for clustering of similar news
        recent_news = news[:20]
        
        # Simple sentiment clustering
        sentiments = [n.sentiment for n in recent_news if hasattr(n, 'sentiment')]
        
        if len(sentiments) > 10:
            # Check if sentiment is overwhelmingly one-sided
            positive = sentiments.count('positive')
            negative = sentiments.count('negative')
            total = len(sentiments)
            
            return (positive > total * 0.7) or (negative > total * 0.7)
            
        return False
    
    def _detect_feedback_loops(self, prices_df: pd.DataFrame) -> Dict[str, bool]:
        """Detect positive and negative feedback loops"""
        if len(prices_df) < 50:
            return {"positive": False, "negative": False}
            
        returns = prices_df['close'].pct_change().dropna()
        
        # Positive feedback: momentum
        momentum = returns.rolling(10).mean()
        future_returns = returns.shift(-5)
        
        # Remove NaN values
        valid_idx = ~(momentum.isna() | future_returns.isna())
        
        if valid_idx.sum() > 20:
            correlation = momentum[valid_idx].corr(future_returns[valid_idx])
            positive_feedback = correlation > 0.3
        else:
            positive_feedback = False
            
        # Negative feedback: mean reversion
        deviation = prices_df['close'] - prices_df['close'].rolling(50).mean()
        future_returns = returns.shift(-10)
        
        valid_idx = ~(deviation.isna() | future_returns.isna())
        
        if valid_idx.sum() > 20:
            correlation = deviation[valid_idx].corr(future_returns[valid_idx])
            negative_feedback = correlation < -0.3
        else:
            negative_feedback = False
            
        return {"positive": positive_feedback, "negative": negative_feedback}
    
    def _detect_new_equilibrium(self, prices_df: pd.DataFrame) -> bool:
        """Detect emergence of new equilibrium"""
        if len(prices_df) < 100:
            return False
            
        prices = prices_df['close'].values
        
        # Check for regime change
        midpoint = len(prices) // 2
        first_half = prices[:midpoint]
        second_half = prices[midpoint:]
        
        # Statistical test for different distributions
        ks_stat, p_value = stats.ks_2samp(first_half, second_half)
        
        # Check if variance has decreased (settling into equilibrium)
        var_decrease = np.std(second_half) < np.std(first_half) * 0.7
        
        return p_value < 0.05 and var_decrease
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two series"""
        if len(x) != len(y) or len(x) < 10:
            return 0
            
        # Discretize into bins
        bins = 5
        
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Calculate marginal probabilities
        px = hist_2d.sum(axis=1) / hist_2d.sum()
        py = hist_2d.sum(axis=0) / hist_2d.sum()
        
        # Joint probability
        pxy = hist_2d / hist_2d.sum()
        
        # Mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
                    
        # Normalize
        return mi / np.log2(bins)
    
    def _analyze_wave_packet(self, series: np.ndarray) -> Dict[str, float]:
        """Analyze wave packet properties"""
        if len(series) < 50:
            return {"group_velocity": 0, "dispersion": 0}
            
        # Calculate envelope using Hilbert transform
        analytic = signal.hilbert(series - np.mean(series))
        envelope = np.abs(analytic)
        
        # Group velocity (speed of envelope)
        envelope_peaks, _ = signal.find_peaks(envelope)
        
        if len(envelope_peaks) > 1:
            peak_distances = np.diff(envelope_peaks)
            group_velocity = np.mean(peak_distances)
        else:
            group_velocity = 0
            
        # Dispersion (spreading of packet)
        dispersion = np.std(envelope) / np.mean(envelope) if np.mean(envelope) > 0 else 0
        
        return {
            "group_velocity": group_velocity,
            "dispersion": dispersion
        }
    
    def _interpret_wave_function(self, frequency: float, amplitude: float) -> str:
        """Interpret wave function characteristics"""
        if frequency > 0.1:
            freq_type = "high frequency oscillations"
        elif frequency > 0.01:
            freq_type = "medium frequency waves"
        else:
            freq_type = "low frequency trends"
            
        if amplitude > 100:
            amp_type = "large amplitude"
        elif amplitude > 10:
            amp_type = "moderate amplitude"
        else:
            amp_type = "small amplitude"
            
        return f"{amp_type} {freq_type}"
    
    def _calculate_correlation_dimension(self, series: np.ndarray) -> float:
        """Calculate correlation dimension"""
        if len(series) < 100:
            return 2.0
            
        # Embed the series
        embedding_dim = 5
        embedded = []
        
        for i in range(len(series) - embedding_dim + 1):
            embedded.append(series[i:i + embedding_dim])
            
        embedded = np.array(embedded)
        
        # Calculate correlation sum for different radii
        radii = np.logspace(-2, 0, 10)
        correlation_sums = []
        
        for r in radii:
            count = 0
            n_points = len(embedded)
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if np.linalg.norm(embedded[i] - embedded[j]) < r:
                        count += 1
                        
            if n_points > 1:
                correlation_sums.append(2 * count / (n_points * (n_points - 1)))
            else:
                correlation_sums.append(0)
                
        # Estimate dimension from scaling
        valid_sums = [(r, c) for r, c in zip(radii, correlation_sums) if c > 0]
        
        if len(valid_sums) > 2:
            log_r = [np.log(r) for r, _ in valid_sums]
            log_c = [np.log(c) for _, c in valid_sums]
            
            try:
                slope, _ = np.polyfit(log_r, log_c, 1)
                return min(max(slope, 1.0), 3.0)  # Bound between 1 and 3
            except:
                return 2.0
                
        return 2.0
    
    def _recurrence_analysis(self, series: np.ndarray) -> float:
        """Perform recurrence analysis"""
        if len(series) < 50:
            return 0
            
        # Create recurrence matrix
        threshold = 0.1 * np.std(series)
        n = len(series)
        
        recurrence_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(series[i] - series[j]) < threshold:
                    recurrence_count += 1
                    
        # Recurrence rate
        if n > 1:
            recurrence_rate = 2 * recurrence_count / (n * (n - 1))
        else:
            recurrence_rate = 0
            
        return recurrence_rate
    
    def _check_sensitivity(self, series: np.ndarray) -> bool:
        """Check for sensitive dependence on initial conditions"""
        if len(series) < 100:
            return False
            
        # Compare first and second halves
        mid = len(series) // 2
        first_half = series[:mid]
        second_half = series[mid:]
        
        # Check if small differences lead to large divergence
        initial_diff = abs(first_half[0] - second_half[0])
        final_diff = abs(first_half[-1] - second_half[-1])
        
        if initial_diff > 0:
            amplification = final_diff / initial_diff
            return amplification > 10
            
        return False
    
    def _calculate_clustering(self, series: np.ndarray) -> float:
        """Calculate clustering coefficient for returns"""
        if len(series) < 50:
            return 0
            
        # Create network where nodes are time points
        # Edges exist if returns are similar
        threshold = np.std(series) * 0.5
        
        n = len(series)
        adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(series[i] - series[j]) < threshold:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                    
        # Calculate clustering coefficient
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adjacency[i] == 1)[0]
            k = len(neighbors)
            
            if k >= 2:
                # Count edges between neighbors
                edges = 0
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adjacency[neighbors[j], neighbors[l]] == 1:
                            edges += 1
                            
                clustering_coeffs.append(2 * edges / (k * (k - 1)))
                
        if clustering_coeffs:
            return np.mean(clustering_coeffs)
            
        return 0


def quantum_market_dynamics_agent(state: AgentState):
    """
    Quantum Market Dynamics Agent - Detecting hidden patterns using quantum mechanics
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    # Initialize quantum analyzer
    quantum_analyzer = QuantumMarketAnalyzer()
    
    # Prepare for parallel processing
    executor = ThreadPoolExecutor(max_workers=5)
    
    quantum_analysis = {}
    
    for ticker in tickers:
        progress.update_status("quantum_market_dynamics_agent", ticker, "Initializing quantum analysis")
        
        # Fetch data in parallel
        futures = []
        futures.append(executor.submit(get_prices, ticker, start_date, end_date))
        futures.append(executor.submit(get_financial_metrics, ticker, end_date, "ttm", 10))
        futures.append(executor.submit(get_company_news, ticker, end_date, None, 50))
        
        # Collect results
        prices = futures[0].result()
        metrics = futures[1].result()
        news = futures[2].result()
        
        if not prices:
            progress.update_status("quantum_market_dynamics_agent", ticker, "Failed: No price data")
            continue
            
        # Convert to DataFrame
        prices_df = prices_to_df(prices)
        
        if prices_df.empty or len(prices_df) < 50:
            progress.update_status("quantum_market_dynamics_agent", ticker, "Failed: Insufficient data")
            continue
            
        # Perform quantum analysis
        analysis = quantum_analyzer.analyze_quantum_dynamics(prices_df, metrics, news, ticker)
        
        # Determine signal based on quantum state
        quantum_state = analysis['quantum_state']['state']
        phase_prob = analysis['phase_transitions']['probability']
        regime = analysis['market_regime']['regime']
        entropy = analysis['entropy']['total']
        coherence = analysis['coherence']['score']
        
        # Complex signal determination
        if quantum_state == "superposition" and phase_prob > 70:
            signal = "quantum_surge"
            confidence = min(95, phase_prob)
        elif quantum_state in ["coherent", "excited"] and entropy < 0.4:
            signal = "bullish"
            confidence = min(85, coherence)
        elif quantum_state in ["decoherent", "collapsed"] and entropy > 0.7:
            signal = "quantum_collapse"
            confidence = min(90, entropy * 100)
        elif regime == "chaotic" or regime == "strange_attractor":
            signal = "bearish"
            confidence = 75
        else:
            signal = "neutral"
            confidence = 50
            
        # Generate comprehensive output
        progress.update_status("quantum_market_dynamics_agent", ticker, "Generating quantum insights")
        
        quantum_output = generate_quantum_output(
            ticker=ticker,
            analysis=analysis,
            signal=signal,
            confidence=confidence,
            state=state
        )
        
        quantum_analysis[ticker] = {
            "signal": quantum_output.signal,
            "confidence": quantum_output.confidence,
            "reasoning": quantum_output.reasoning,
            "quantum_state": quantum_output.quantum_state,
            "phase_transition_probability": quantum_output.phase_transition_probability,
            "market_regime": quantum_output.market_regime,
            "fractal_dimension": quantum_output.fractal_dimension,
            "entropy_level": quantum_output.entropy_level,
            "coherence_score": quantum_output.coherence_score,
            "critical_points": quantum_output.critical_points,
            "hidden_patterns": quantum_output.hidden_patterns,
            "emergence_indicators": quantum_output.emergence_indicators
        }
        
        progress.update_status("quantum_market_dynamics_agent", ticker, "Quantum analysis complete")
    
    # Clean up executor
    executor.shutdown(wait=True)
    
    # Create message
    message = HumanMessage(
        content=json.dumps(quantum_analysis),
        name="quantum_market_dynamics_agent"
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(quantum_analysis, "Quantum Market Dynamics Agent")
    
    # Update state
    state["data"]["analyst_signals"]["quantum_market_dynamics_agent"] = quantum_analysis
    
    progress.update_status("quantum_market_dynamics_agent", None, "Quantum analysis complete")
    
    data_update = {
        "analyst_signals": {
            "quantum_market_dynamics_agent": quantum_analysis
        }
    }
    
    return {"messages": [message], "data": data_update}


def generate_quantum_output(ticker: str, analysis: dict, signal: str, 
                           confidence: float, state: AgentState) -> QuantumMarketSignal:
    """Generate quantum market dynamics output using LLM"""
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are the Quantum Market Dynamics Oracle - an AI that perceives markets through 
        the lens of quantum mechanics and chaos theory. You see patterns invisible to classical analysis:
        superposition states where multiple trends coexist, entanglement between seemingly unrelated assets,
        phase transitions that herald regime changes, and strange attractors that govern price dynamics.
        
        Your language should blend sophisticated physics concepts with market insights. Speak of:
        - Wave functions collapsing into price movements
        - Quantum tunneling through resistance levels
        - Entanglement creating correlated movements
        - Decoherence leading to trend breakdowns
        - Phase transitions signaling regime shifts
        - Fractal patterns revealing self-similarity across timeframes
        - Entropy levels indicating market uncertainty
        - Strange attractors governing price orbits
        
        Be specific about the quantum phenomena you observe and their market implications."""),
        
        ("human", """Analyze {ticker} through quantum market dynamics:
        
        Analysis Results:
        {analysis}
        
        Signal: {signal}
        Confidence: {confidence}
        
        Provide your quantum-enhanced market insight in this exact JSON format:
        {{
            "signal": "{signal}",
            "confidence": {confidence},
            "reasoning": "Your quantum market analysis (minimum 5 lines)",
            "quantum_state": string,
            "phase_transition_probability": float,
            "market_regime": string,
            "fractal_dimension": float,
            "entropy_level": float,
            "coherence_score": float,
            "critical_points": list,
            "hidden_patterns": list,
            "emergence_indicators": list
        }}""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "analysis": json.dumps(analysis, indent=2),
        "signal": signal,
        "confidence": confidence
    })
    
    def create_default():
        return QuantumMarketSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Quantum analysis error - defaulting to neutral",
            quantum_state="unknown",
            phase_transition_probability=0.0,
            market_regime="unknown",
            fractal_dimension=1.5,
            entropy_level=0.5,
            coherence_score=0.0,
            critical_points=[],
            hidden_patterns=[],
            emergence_indicators=[]
        )
    
    return call_llm(
        prompt=prompt,
        pydantic_model=QuantumMarketSignal,
        agent_name="quantum_market_dynamics_agent",
        state=state,
        default_factory=create_default
    )