import tensorflow as tf
from model import CNN_Encoder, RNN_Decoder
# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
from PIL import Image
from tqdm import tqdm
import pickle
from utils import load_image
#from helper import standardize, evaluate, plot_attention

#importing all the helper fxn from helper.py which we will create later

import streamlit as st

import os

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(style="darkgrid")

sns.set()

from PIL import Image

st.title('Image Caption Generator')

image_model = tf.keras.applications.InceptionV3(include_top=False)
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")



# Max word count for a caption.
max_length = 50 ## NEED
attention_features_shape = 64 ## NEED
# Use the top 5000 words for a vocabulary.
vocabulary_size = 70000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)
# Learn the vocabulary from the caption data.
#tokenizer.adapt(caption_dataset)


from_disk = pickle.load(open("tv_layer.pkl", "rb"))
tokenizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
tokenizer.set_weights(from_disk['weights'])

# Create mappings for words to indices and indicies to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

# Feel free to change these parameters according to your system's configuration

embedding_dim = 256
units = 512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
attention_features_shape = 64 ## NEED



encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

optimizer = tf.keras.optimizers.Adam()

checkpoint_path = "checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()



def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0

uploaded_file = st.file_uploader("Upload Image")
# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)

        #image_urll ='https://tensorflow.org/images/surf.jpg'
        #image_extension = image_urll[-4:]
        image_path = os.path.join('static/images',uploaded_file.name)#tf.keras.utils.get_file('image'+image_extension, origin=image_urll)
        #print(image_path)
        result, attention_plot = evaluate(image_path)
        print('Prediction Caption:', ' '.join(result))
        fig1=plot_attention(image_path, result, attention_plot)
        # opening the image
        
        result2, attention_plot2 = evaluate(image_path)
        result3, attention_plot3 = evaluate(image_path)
        result4, attention_plot4 = evaluate(image_path)
        result5, attention_plot5 = evaluate(image_path)

        #os.remove('static/images/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        st.subheader('5 Predicted Captions for the above image :-')
        st.write("__________________________________________________________________________________")
        
        st.pyplot(fig1)
        st.write(' '.join(result))
        st.write(' '.join(result2))
        st.write(' '.join(result3))
        st.write(' '.join(result4))
        st.write(' '.join(result5))
        st.markdown('***')
        st.markdown("Thanks for going through this mini-analysis with us. Cheers!")