from flask import Flask
from message_length_estimator import message_lengths_words, message_length_char

app = Flask(__name__)

from applic import run
