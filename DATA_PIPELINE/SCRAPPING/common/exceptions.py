class ScrapingError(Exception): pass
class RateLimitError(ScrapingError): pass
class ValidationError(ScrapingError): pass
