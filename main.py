import nltk
from flask import Flask,jsonify,request
app = Flask(__name__)
import random


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

@app.route("/",methods=['GET', 'POST'])
def greeting():
	if request.method == 'GET':
		sentence = request.args.get('q')
		for word in sentence.split():
			if word.lower() in GREETING_INPUTS:
				return jsonify(random.choice(GREETING_RESPONSES))