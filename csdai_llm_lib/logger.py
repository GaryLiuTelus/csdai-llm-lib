import io
import logging
import queue
import sys
import threading
import time
from contextlib import redirect_stderr, suppress
from dataclasses import dataclass
from json import dumps

from google.cloud.logging import Client as CloudLoggingClient
from google.cloud.logging import handlers as gcp_logging_handlers
from httplib2 import Http


class GoogleChatLoggingHandler(logging.Handler):
    """Post log records to a Google Chat webhook without ever blocking the caller.

    `emit` only enqueues; a daemon thread does the HTTP calls, with a request
    timeout and a fresh connection each time (a kept-alive one goes stale and
    fails with `SSL: UNEXPECTED_EOF_WHILE_READING`). Records are dropped when the
    queue is full, and delivery failures are reported on stderr at most a few
    times, so a slow or failing webhook can neither stall the worker threads nor
    abort the job.
    """

    _STOP = object()
    MAX_QUEUED = 500
    REQUEST_TIMEOUT_SECONDS = 10
    MIN_INTERVAL_SECONDS = 1.0  # Google Chat allows about one message per second
    MAX_FAILURE_REPORTS = 3

    def __init__(self, chat_webhook: str, level: int = 0) -> None:
        super().__init__(level)
        self.__webhook_url = chat_webhook
        self.__queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUED)
        self.__failures = 0
        self.__worker = threading.Thread(target=self.__run, daemon=True)
        self.__worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except (TypeError, ValueError, KeyError):  # bad %-args or format field
            self.handleError(record)
            return
        with suppress(queue.Full):  # drop the record rather than block the caller
            self.__queue.put_nowait(text)

    def __send(self, text: str) -> None:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                Http(timeout=self.REQUEST_TIMEOUT_SECONDS).request(
                    uri=self.__webhook_url,
                    method="POST",
                    headers={"Content-Type": "application/json; charset=UTF-8"},
                    body=dumps({"text": text}),
                )
                return
            except Exception as e:  # noqa: BLE001
                last_error = e
        self.__failures += 1
        if self.__failures <= self.MAX_FAILURE_REPORTS:
            print(f"Failed to send chat notification: {last_error!r}", file=sys.stderr)

    def __run(self) -> None:
        while True:
            item = self.__queue.get()
            if item is self._STOP:
                return
            self.__send(item)
            time.sleep(self.MIN_INTERVAL_SECONDS)

    def close(self) -> None:
        # Give queued messages a bounded time to go out, then stop the worker.
        try:
            self.__queue.put(self._STOP, timeout=1)
            self.__worker.join(timeout=30)
        finally:
            super().close()


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
            "critical": logging.CRITICAL,
        }
        self._logger = logging.getLogger(self.logger_name)
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
        self._logger.setLevel(self.logging_level)
        self._logger.propagate = False
        self._default_format = logging.Formatter(
            fmt="%(levelname)8s - %(asctime)s:"
            + self.logger_name.upper()
            + " (job_id: "
            + self.job_id
            + "), %(filename)s "
            + ":: %(funcName)s, line %(lineno)04d `%(message)s`",
            datefmt="%Y-%m-%d %H:%M:%S",
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
        self._default_format = logging.Formatter(fmt=new_format, **kwargs)

    @property
    def logging_level(self) -> int:
        return self.__log_level_mapping.get(self.level, logging.DEBUG)

    def logger(
        self,
        chat_webhook: str | None = None,
        cloud_logging_project_id: str | None = None,
    ) -> logging.Logger:

        if chat_webhook is not None:
            for handler in list(self._logger.handlers):
                if isinstance(handler, GoogleChatLoggingHandler):
                    self._logger.removeHandler(handler)

            chat_handler = GoogleChatLoggingHandler(chat_webhook)
            chat_handler.setFormatter(self._default_format)
            chat_handler.setLevel(logging.ERROR)
            self._logger.addHandler(chat_handler)
        if cloud_logging_project_id is not None:
            for handler in list(self._logger.handlers):
                if isinstance(handler, gcp_logging_handlers.CloudLoggingHandler):
                    self._logger.removeHandler(handler)
            cloud_handler = gcp_logging_handlers.CloudLoggingHandler(
                CloudLoggingClient(project=cloud_logging_project_id)
            )
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

        On shutdown the handler prints progress lines ("Waiting up to 5
        seconds.", "Sent all pending logs.") to stderr, which Cloud Batch
        records as ERROR. They are swallowed here; only a genuine
        "Failed to send ..." line is passed on to stderr.
        """
        captured = io.StringIO()
        with redirect_stderr(captured):
            for handler in list(self._logger.handlers):
                try:
                    handler.flush()
                    handler.close()
                except Exception:  # noqa: BLE001, S110
                    pass
                self._logger.removeHandler(handler)
        for line in captured.getvalue().splitlines():
            if "Failed to send" in line:
                print(line, file=sys.stderr)
