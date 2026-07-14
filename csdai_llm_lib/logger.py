import logging
import sys
from json import dumps
from dataclasses import dataclass

from google.cloud.logging import Client as CloudLoggingClient, handlers as gcp_logging_handlers
from httplib2 import Http

from .base_env import get_base_settings

class GoogleChatLoggingHandler(logging.Handler):

    def __init__(self, chat_webhook:str, level: int = 0) -> None:
        super().__init__(level)
        self.__webhook_url = chat_webhook
        self.__http = Http()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.__http.request(
                uri=self.__webhook_url,
                method="POST",
                headers={"Content-Type": "application/json; charset=UTF-8"},
                body=dumps({"text": self.format(record)})
            )
        except Exception:
            self.handleError(record)
@dataclass
class LLMLogger:
    logger_name: str = "llm_chain"
    level: str = "debug"
    job_id: str = "test"


    def __post_init__(self):
        self.__log_level_mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        self._logger = logging.getLogger(self.logger_name)
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
        self._logger.setLevel(self.logging_level)
        self._logger.propagate = False
        self._default_format = logging.Formatter(
            fmt="%(levelname)8s - %(asctime)s:" + self.logger_name.upper() + \
                " (job_id: " + self.job_id + "), %(filename)s " + \
                ":: %(funcName)s, line %(lineno)04d `%(message)s`",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._default_format)
        console_handler.setLevel(self.logging_level)
        self._logger.addHandler(console_handler)

    @property
    def logging_format(self):
        return self._default_format

    @logging_format.setter
    def logging_format(self, new_format: str, **kwargs):
        self._default_format = logging.Formatter(
            fmt=new_format,
            **kwargs
        )

    @property
    def logging_level(self) -> int:
        return self.__log_level_mapping.get(self.level, logging.DEBUG)

    def logger(self, chat_webhook: str | None = None, cloud_logging_project_id: str | None = None) -> logging.Logger:

        if chat_webhook is not None:
            for handler in self._logger.handlers:
                if isinstance(handler, GoogleChatLoggingHandler):
                    self._logger.removeHandler(handler)

            chat_handler = GoogleChatLoggingHandler(chat_webhook)
            chat_handler.setFormatter(self._default_format)
            chat_handler.setLevel(logging.ERROR)
            self._logger.addHandler(chat_handler)
        if cloud_logging_project_id is not None:
            for handler in self._logger.handlers:
                if isinstance(handler, gcp_logging_handlers.CloudLoggingHandler):
                    self._logger.removeHandler(handler)
            cloud_handler = gcp_logging_handlers.CloudLoggingHandler(CloudLoggingClient(project=get_base_settings().project_id))
            cloud_handler.setFormatter(self._default_format)
            cloud_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(cloud_handler)

        return self._logger

    def close(self) -> None:
        """Flush and close all handlers.

        The CloudLoggingHandler ships entries via a background thread; if the
        interpreter exits before that thread drains its queue, google-cloud-logging
        warns that entries may be lost. Calling this before the program ends
        flushes and closes the handlers deterministically.
        """
        for handler in list(self._logger.handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:  # pylint: disable = broad-except
                pass
            self._logger.removeHandler(handler)
