from utils.log import configure_logging
from dotenv import load_dotenv, find_dotenv

configure_logging()
load_dotenv(find_dotenv())