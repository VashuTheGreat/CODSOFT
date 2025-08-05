import pickle
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda




model = load_model('imgcaption1.keras')
model_feature=load_model('model.h5')

with open('tokenizer.pkl','rb') as f:

    tokenizer=pickle.load(f)

def generate_features(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)
    
    feature = model_feature.predict(image, verbose=0)
    return feature.reshape((feature.shape[0], -1))



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = '<start>'
    image=generate_features(image)
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)[0]
        
        yhat = model.predict([image, np.array([sequence])], verbose=0) 
        yhat = np.argmax(yhat)
        
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break

        in_text += ' ' + word
        if word == '<end>':
            break

    return in_text.replace('<start>', '').replace('<end>', '').strip()


prediction_text=predict_caption(model=model,image='./data/Images/41999070_838089137e.jpg',tokenizer=tokenizer,max_length=35)
result = prediction_text.split("end", 1)[0]
print(result)

