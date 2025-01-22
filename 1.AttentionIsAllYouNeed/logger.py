import logging

class Logger:

    COLORS = {
        "INFO": "\033[32m",     # Verde
        "WARNING": "\033[33m",  # Amarillo
        "ERROR": "\033[31m",    # Rojo
        "RESET": "\033[0m",     # Restablecer color
    }

    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) 
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message):
        print(f"{self.COLORS['INFO']}INFO: {message}{self.COLORS['RESET']}")
        self.logger.info(message)

    def error(self, message):
        print(f"{self.COLORS['ERROR']}ERROR: {message}{self.COLORS['RESET']}")
        self.logger.error(message)

    def warning(self, message):
        print(f"{self.COLORS['WARNING']}WARNING: {message}{self.COLORS['RESET']}")
        self.logger.warning(message)

logger = Logger("logger.log")
