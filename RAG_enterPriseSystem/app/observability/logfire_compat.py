"""
Optional Logfire adapter.

The production stack should install and configure logfire, but local demos and
CI smoke tests should not fail just because observability is unavailable.
"""

from contextlib import contextmanager
import logging
import os


logger = logging.getLogger("ip_sakti")

try:
    if os.getenv("IP_SAKTI_DISABLE_LOGFIRE", "").lower() in {"1", "true", "yes"}:
        _logfire = None
    else:
        import logfire as _logfire
except ModuleNotFoundError:
    _logfire = None


@contextmanager
def _noop_span(*args, **kwargs):
    class _NoopSpan:
        def set_attribute(self, *args, **kwargs):
            return None

    yield _NoopSpan()


class _LogfireCompat:
    def configure(self, *args, **kwargs):
        if _logfire:
            token = kwargs.get("token")
            if not token or str(token).startswith("your_"):
                kwargs["token"] = None
                kwargs.setdefault("send_to_logfire", False)
                kwargs.setdefault("console", False)
            return _logfire.configure(*args, **kwargs)
        logging.basicConfig(level=logging.INFO)
        return None

    def instrument_requests(self, *args, **kwargs):
        if _logfire and hasattr(_logfire, "instrument_requests"):
            return _logfire.instrument_requests(*args, **kwargs)
        return None

    @staticmethod
    def _safe_msg(msg):
        if isinstance(msg, str) and "{" in msg and "}" in msg:
            return msg.replace("{", "{{").replace("}", "}}")
        return msg

    def span(self, *args, **kwargs):
        if _logfire:
            return _logfire.span(*args, **kwargs)
        return _noop_span(*args, **kwargs)

    def info(self, message, *args, **kwargs):
        if _logfire:
            return _logfire.info(self._safe_msg(message), *args, **kwargs)
        logger.info(message)
        return None

    def warning(self, message, *args, **kwargs):
        safe = self._safe_msg(message)
        if _logfire and hasattr(_logfire, "warning"):
            return _logfire.warning(safe, *args, **kwargs)
        if _logfire and hasattr(_logfire, "warn"):
            return _logfire.warn(safe, *args, **kwargs)
        logger.warning(message)
        return None

    def warn(self, message, *args, **kwargs):
        return self.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        if _logfire:
            return _logfire.error(self._safe_msg(message), *args, **kwargs)
        logger.error(message)
        return None

    def exception(self, message, *args, **kwargs):
        if _logfire:
            return _logfire.exception(self._safe_msg(message), *args, **kwargs)
        logger.exception(message)
        return None


logfire = _LogfireCompat()
