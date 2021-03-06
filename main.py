import nltk
from flask import Flask, jsonify, request, Response
import random
import numpy as np
import string
from flask_cors import CORS
from camera import Camera


app = Flask(__name__)
CORS(app)

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
try:
    f = open('chatbot.txt', 'r', errors='ignore')
except:
    f = open('/home/Yajana/Python-Backend/chatbot.txt', 'r', errors='ignore')

raw=f.read()
raw=raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words



lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generating response
def response(user_response):
    robo_response=''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

@app.route("/gethint.php",methods=['GET', 'POST'])
def getHint():
	global word_tokens
	global sent_tokens
	if request.method == 'GET':
		user_response = str(request.args.get('q'))
		for word in user_response.split():
			if word.lower() in GREETING_INPUTS:
				return jsonify(random.choice(GREETING_RESPONSES))

			else:
				sent_tokens.append(user_response)
				word_tokens=word_tokens+nltk.word_tokenize(user_response)
				final_words=list(set(word_tokens))
				resp = response(user_response)
				sent_tokens.remove(user_response)
				return jsonify(resp)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

   
