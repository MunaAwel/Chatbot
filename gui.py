import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model
model = load_model('Train_chatbot.h5')
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import random
import json
import tkinter as tk
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


# Error handling technique when opening files
try:
    with open(r"C:\Users\ERC\Downloads\intents.json") as json_file:
        intents = json.load(json_file)
        print("File successfully opened and loaded.")
    # Further operations with 'intents' data can be placed here
except IOError:
    print("Error: Unable to open file or file does not exist.")
except json.JSONDecodeError:
    print("Error: JSON decoding error. File may not be valid JSON.")

#the first step is to tokenize the words
# once the words are tokenized we need to find the base root word for each tokenized
def clean_up_sentence(sentence):
    sentence_words = nltk.word.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_detail= True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_detail:
                    print(f"Found in bad : {word}")
    return (np.array(bag))

def predict_clasess(sentence):
    results = []
    x = bag_of_words(sentence, words,show_detail = False)
    res = model.predict(np.array(x))[0]
    error_threshold = 0
    for i,r in enumerate(res):
        if r > 0.25:
            results.append([i,r])
        results.sort(key=lambda a : a[1], reverse = True)
        return_list =[]
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intent = intents_json['intents']
    for i in list_of_intent:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


root = tk.Tk()
root.title("Muna's Chatbot")
root.geometry("400x500")
root.resizable(width = False, height = False)

mychatbot = tk.Text(root,bd = "0", bg = "white", height = "8", width = "50", font = ("Arial", 18))
mychatbot.config(state ="disabled")

scrollbar = tk.Scrollbar(root, command =mychatbot.yview, cursor="heart")
mychatbot['yscrollcommand'] = scrollbar.set

sendButton = tk.Button(root, font=("Arial", 15, "bold"), text ="Send", width = 15, height = 4,
                       bd = 0, bg = "pink", fg = "blue" )

entryBox = tk.Text(root, font = ("Arial",15), bd = 0, bg = "white", width ="29", height = "5")

scrollbar.place(x = 376, y = 6, height = 386)
mychatbot.place(x=6, y=6, height = 386, width = 370)
entryBox.place(x=128, y=401, height=90, width=265)
sendButton.place(x=6, y=401, height=90)
root.mainloop()