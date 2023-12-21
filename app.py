from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import cv2
from tensorflow.keras.preprocessing import image
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np



model_path1 = os.path.join(os.path.dirname(__file__), 'jalan.h5')
#modeldeteksi = load_model(model_path1)






app = Flask(__name__)
app.static_folder = 'static'

leafCascade = cv2.CascadeClassifier("static/src/cascade3.xml")
models = load_model(model_path1)
label= ['jalan retak','rusak_kecil','rusak_parah','rusak_sedang']




import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model/models.h5')
import json
import random
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl','rb'))
classes = pickle.load(open('model/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/chatbot", methods=['GET'])
def home_chatbot():
    if request.method == 'POST':
        file = request.files['file']   
    else:
        return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)





@app.route("/")
def home():
    return render_template("index.html")


#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            classes=models.predict(img) 
            index = np.argmax(classes)
            return render_template('predict.html', road = label[index],prob=round(classes[0][index]*100,2), user_image = file_path)
        else:
            return render_template('index.html')


def detect_leaf(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    leaf = leafCascade.detectMultiScale(
        gray,
        scaleFactor = 1.05,
        minNeighbors = 6,
        minSize = (128,128),
        maxSize = (800,800)
    )
    
    for (x, y, w, h) in leaf:
        load = frame[y:y+h, x:x+w]
        load = cv2.resize(load, (128,128))
        z = tf.keras.utils.img_to_array(load)
        z = np.expand_dims(z, axis=0)
        images = np.vstack([z])
        classes = models.predict(images)
        index = np.argmax(classes)
        cv2.putText(frame, label[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
    return frame

def gen_frames(): 
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_leaf(frame)
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect')
def detector():
    return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)