from main import cleaner 
import pickle
import random
import json
with open('data.json','r') as f:
    data=json.load(f)

def response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't get that."
    

with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('indexes.pkl', 'rb') as f:
    index = pickle.load(f)
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


def output(textin):
    res = model.named_steps['cleaner'](textin)
    vector = model.named_steps['cv'].transform([res])
    prediction = model.named_steps['modelR'].predict(vector.toarray().reshape(1, -1))
    return index[prediction[0]]


print("💬 This is your chatbot. Type 'exit' to stop.")

while True:
    print("👤 You: ", end="")
    
    textin = input()
    if textin.lower() == 'exit':
        break
    result = output(textin)
    response_text = response(result)
    print("🤖 Bot: ",response_text)
    
    
    
    


