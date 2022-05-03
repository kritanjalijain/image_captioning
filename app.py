import tensorflow as tf
from model import CNN_Encoder, RNN_Decoder
#from model.py import model architecture
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from PIL import Image
import urllib.request
import pickle
from utils import load_image
#from helper import standardize, evaluate, plot_attention
#importing all the helper fxn from helper.py which we will create later
import requests
from streamlit_lottie import st_lottie
import streamlit as st
import hydralit_components as hc
import time
import seaborn as sns


sns.set_theme(style="darkgrid")


sns.set()



 


def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
        
lottie_load = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_nncar5qq.json')

with st.spinner('Loading...'):
    #st_lottie(lottie_load, speed=1, height=500, key="initial")
    #time.sleep(7)

    with hc.HyLoader('',hc.Loaders.pulse_bars,):
    #time.sleep(7)
        time.sleep(7)

# a dedicated single loader

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
# Use the top 70000 words for a vocabulary.
vocabulary_size = 70000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)

# import learnt vocab
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

#print('before loading', encoder.weights)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()


#predict function
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

def app():
    st.title('Auto Image Caption Generator')
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
        
    #image_path =''
    lottie_att = load_lottieurl('https://assets10.lottiefiles.com/private_files/lf30_jynxrd4m.json')
    st_lottie(lottie_att, speed=1, height=380)
    st.subheader("Generate captions of images automatically!")
    st.markdown("Hey there! Welcome to Auto Image Captioning App. This app uses (and never keeps or stores!) the image you want to analyze and generate 5 most suitable captions for the same along with attention plot.")
    
    st.write("__________________________________________________________________________________")
    # Radio Buttons
    st.markdown(" **To begin, let's select the type of fetching you want to conduct. You can either fetch an image via url or search upload an image from your local device to predict captions 👇.** ")
    st.write("")
    stauses = st.radio('Select the mode of fetching',("Fetch image via url","Upload image from local device"))
    if stauses == 'Fetch image via url':
        st.success("Enter Url")
        image_url = st.text_input("Copy paste URL of image")
        if image_url is not "":
            urllib.request.urlretrieve(image_url, "temp.png")
            st.image(image_url)
            result, attention_plot = evaluate("temp.png")
            print('Prediction Caption:', ' '.join(result))
            fig1=plot_attention("temp.png", result, attention_plot)
                    # opening the image
                    
            result2, attention_plot2 = evaluate("temp.png")
            result3, attention_plot3 = evaluate("temp.png")
            result4, attention_plot4 = evaluate("temp.png")
            result5, attention_plot5 = evaluate("temp.png")

            #os.remove('temp.png')

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
        #print(image_path)
    elif stauses == 'Upload image from local device':
        st.success("Upload Image")
        st_lottie(load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_2oranrew.json'), speed=1, height=280)
        uploaded_file = st.file_uploader("Upload Image")
        # text over upload button "Upload Image"

        if uploaded_file is not None:

            if save_uploaded_file(uploaded_file): 

                # display the image

                display_image = Image.open(uploaded_file)

                st.image(display_image)

                image_path = os.path.join('static/images',uploaded_file.name)
                result, attention_plot = evaluate(image_path)
                print('Prediction Caption:', ' '.join(result))
                fig1=plot_attention(image_path, result, attention_plot)
                        # opening the image
                        
                result2, attention_plot2 = evaluate(image_path)
                result3, attention_plot3 = evaluate(image_path)
                result4, attention_plot4 = evaluate(image_path)
                result5, attention_plot5 = evaluate(image_path)


                st.set_option('deprecation.showPyplotGlobalUse', False)
                        
                st.subheader('5 Predicted Captions for the above image :-')
                st.write("__________________________________________________________________________________")
                        
                st.pyplot(fig1)
                st.write(' '.join(result))
                st.write(' '.join(result2))
                st.write(' '.join(result3))
                st.write(' '.join(result4))
                st.write(' '.join(result5))
                print('after loading', encoder.weights)
                os.remove('static/images/'+uploaded_file.name) # deleting uploaded saved picture after prediction
                #print(image_path)
    else:
        st.warning("Choose an option")

    st.markdown('***')
    st.markdown("Thanks for going through this mini-analysis with us. Cheers!")
    

if __name__ == "__main__":
	app()