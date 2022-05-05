# Auto Image Captioning using Attention Mechanism

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/kritanjalijain/image_captioning/blob/master/test_img_captioning.ipynb)

## Project Description
The project aims to develop a system that automatically generates rich, accurate and human-like description of the objects in the image and the relationships between those objects. The system uses a convolutional neural network (CNN) to extract features from an image, integrates the features with an attention mechanism, and creates captions using a recurrent neural network (RNN). To encode an image into a feature vector as graphical attributes, we employed GoogleNet (Inception V3) pre-trained convolutional neural network. Following that, a GRU (Gated Recurrent Unit) is chosen as the decoder to construct the descriptive sentence. In order to increase performance, we merge the Bahdanau attention model with GRU to allow learning to be focused on a specific portion of the image. The user interface for the system is provided via a web app made using the Streamlit framework allowing the user to generate not one but five captions either by uploading the image or through the image URL. On the MS COCO dataset, the results achieve competitive performance against state-of-the-art approaches.

## Pipeline

![](https://github.com/kritanjalijain/image_captioning/blob/master/images/flow.png)

### Web Application Landing Page

* The home page the user is greeted with after opening the web-app via its URL. The below page explains the process through which users can generate captions for images they feed in. 

![](https://github.com/kritanjalijain/image_captioning/blob/master/images/2.png)
#### Fig. 2- Landing page



* Next, the user can scroll through the page to proceed the mode through which they desire to feed in the photo; either by entering a URL or by uploading it from local device.



### Fetching by an Image URL

* The user can enter the url of the image in the search box after selecting the ‘Fetch image via url’ mode of fetching. 

The figure below shows the initial page setup the user is greeted with.
 
* For instance, in the below figure, the user wishes to generate captions for the image url ‘https://images.fineartamerica.com/images-medium-large-5/cows-in-a-field-with-one-cow-staring-at-john-short--design-pics.jpg’.  Upon pressing ‘Enter’, the request is confirmed and the processing begins. 
 
![](https://github.com/kritanjalijain/image_captioning/blob/master/images/4.png) 
#### Fig. 3


### Caption generation and attention plot visualization


* After pressing enter, the image is fetched from the URL and displayed on the webpage. The results include 5 predicted captions for the given image along with the attention plot which visualizes the parts of the image the attention mechanism focused on during processing the corresponding word of the sentence.


![](https://github.com/kritanjalijain/image_captioning/blob/master/images/5.png)  
#### Fig. 4

* Thus, for the above input image of a few cows grazing in a field, 5 captions are generated which are quite suitable although most of the sentences grasp the gist of the image accurately and highlight important features of the image by using keywords/phrases like ‘group of cows’ , ‘graze the distant field’, ‘grass’; the grammar is not absolutely perfect. The model can be trained further to improve performance and by increasing dataset (which I couldn't due to GPU limitations :/ ). 

### Feeding images via local device

* Optionally, the user can perform a similar analysis by uploading an image from his local computer/mobile after selecting the ‘Upload image from local device’ mode of fetching. 

![](https://github.com/kritanjalijain/image_captioning/blob/master/images/6.png) 
#### Fig. 5

* The above figure displays the upload image option now display. The user can click on ‘Browse files’ to select an image up to the size of 200mb.
 
![](https://github.com/kritanjalijain/image_captioning/blob/master/images/7.png) 
![](https://github.com/kritanjalijain/image_captioning/blob/master/images/8.png) 
 
#### Fig. 6

* Here, we have taken an example of a female tennis player on court. The 5 predicted captions and attention plot for the first one is displayed. 




## Built With
* Python 
* Tensorflow
* Sreamlit


## Setup and Installation
* Clone the repository 
``` 
git clone https://github.com/kritanjalijain/image_captioning.git
```
* Change to working directory
```
cd image_captioning
```
* Install all dependencies (preferrably in a virtual env)
```
pip install -r requirements.txt
```

* Upload weights of trained model in 'checkpoints' folder. 

* Run the app in terminal 
```
streamlit run app.py
```

* Open the local host port the server is using in a browser by clicking on the link displayed in the terminal in case it does not automatically pop-up


## References

*  Show, Attend and Tell: Neural Image Caption Generation with Visual Attention https://arxiv.org/abs/1502.03044
* https://www.tensorflow.org/tutorials/text/image_captioning
* https://docs.streamlit.io/

