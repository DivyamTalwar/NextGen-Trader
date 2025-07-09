class Cache:
    """In-memory cache for API responses. Acts as a simple key-value store."""

    def __init__(self):
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._line_items_cache: dict[str, list[dict[str, any]]] = {} # <-- ADD THIS LINE
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}

    def get_line_items(self, key: str) -> list[dict[str, any]] | None:
        """Get cached line items by its specific key."""
        return self._line_items_cache.get(key)

    def set_line_items(self, key: str, data: list[dict[str, any]]):
        """Set line items in the cache for a specific key."""
        self._line_items_cache[key] = data

    def get_prices(self, key: str) -> list[dict[str, any]] | None:
        """Get cached price data by its specific key."""
        return self._prices_cache.get(key)

    def set_prices(self, key: str, data: list[dict[str, any]]):
        """Set price data in the cache for a specific key."""
        self._prices_cache[key] = data

    def get_financial_metrics(self, key: str) -> list[dict[str, any]] | None:
        """Get cached financial metrics by its specific key."""
        return self._financial_metrics_cache.get(key)

    def set_financial_metrics(self, key: str, data: list[dict[str, any]]):
        """Set financial metrics in the cache for a specific key."""
        self._financial_metrics_cache[key] = data
        
    def get_insider_trades(self, key: str) -> list[dict[str, any]] | None:
        return self._insider_trades_cache.get(key)

    def set_insider_trades(self, key: str, data: list[dict[str, any]]):
        self._insider_trades_cache[key] = data
        
    def get_company_news(self, key: str) -> list[dict[str, any]] | None:
        return self._company_news_cache.get(key)

    def set_company_news(self, key: str, data: list[dict[str, any]]):
        self._company_news_cache[key] = data


_cache = Cache()

def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache