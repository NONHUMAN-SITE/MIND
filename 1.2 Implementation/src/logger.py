import logging
from datetime import datetime

class Logger:

    COLORS = {
        "INFO": "\033[32m",     # Verde
        "WARNING": "\033[33m",  # Amarillo
        "ERROR": "\033[31m",    # Rojo
        "RESET": "\033[0m",     # Restablecer color
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Configurar el handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        self.logger.addHandler(console_handler)

    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message):
        formatted_message = f"{self._get_timestamp()} - {self.COLORS['INFO']}INFO: {message}{self.COLORS['RESET']}"
        print(formatted_message)

    def error(self, message):
        formatted_message = f"{self._get_timestamp()} - {self.COLORS['ERROR']}ERROR: {message}{self.COLORS['RESET']}"
        print(formatted_message)

    def warning(self, message):
        formatted_message = f"{self._get_timestamp()} - {self.COLORS['WARNING']}WARNING: {message}{self.COLORS['RESET']}"
        print(formatted_message)

logger = Logger()
