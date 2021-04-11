#from .models.message_length_estimator import message_lengths_words, message_length_char
from flask import Flask

app = Flask(__name__, template_folder='applic/templates')

from applic import run
