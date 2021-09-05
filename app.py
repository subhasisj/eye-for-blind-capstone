import io
import pickle
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from gtts import gTTS
from PIL import Image, ImageOps
import os
import glob
import time

# Hyperparams
embedding_dim = 256
units = 512
vocab_size = 5001  # top 5,000 words +1
attention_features_shape = 64


try:
    os.mkdir("temp")
except:
    pass


def preprocess_image(img):
    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, (299, 299))
    # img = img.resize((299, 299))
    # img = tf.keras.preprocessing.image.img_to_array(img)

    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


# @st.cache
def initialize_InceptionV3_image_features_extract_model():
    """
    :return: InceptionV3 model instance
    """
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output  # shape [batch_size, 8, 8, 2048]

    return tf.keras.Model(new_input, hidden_layer)


class Encoder(tf.keras.Model):
    def __init__(self, embed_dim=256):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(
            embedding_dim
        )  # build your Dense layer with relu activation

    def call(self, features):
        features = self.dense(
            features
        )  # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = tf.nn.relu(features)
        return features


class Attention_model(tf.keras.Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # build your Dense layer
        self.W2 = tf.keras.layers.Dense(units)  # build your Dense layer
        self.V = tf.keras.layers.Dense(1)  # build your final Dense layer with unit 1
        self.units = units

    def call(self, features, hidden):
        # features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(
            hidden, 1
        )  # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )
        score = self.V(
            attention_hidden_layer
        )  # build your score funciton to shape: (batch_size, 8*8, units)
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # extract your attention weights with shape: (batch_size, 8*8, 1)
        context_vector = (
            attention_weights * features
        )  # shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = tf.reduce_sum(
            context_vector, axis=1
        )  # reduce the shape to (batch_size, embedding_dim)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(
            self.units
        )  # iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(
            vocab_size, embedding_dim
        )  # build your Embedding layer
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.d1 = tf.keras.layers.Dense(self.units)  # build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)  # build your Dense layer

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(
            features, hidden
        )  # create your context vector & attention weights from attention model
        embed = self.embed(
            x
        )  # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat(
            [tf.expand_dims(context_vector, 1), embed], axis=-1
        )  # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output, state = self.gru(
            embed
        )  # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(
            output, (-1, output.shape[2])
        )  # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output)  # shape : (batch_size * max_length, vocab_size)

        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def get_caption_tokenizer(caption_tokenizer_path="./checkpoints/tokenizer.pkl"):
    with open(caption_tokenizer_path, "rb") as f:
        return pickle.load(f)


# @st.cache
def load_model(checkpoint_path="./checkpoints/train", vocab_size=5001):
    tokenizer = get_caption_tokenizer()
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    encoder = Encoder()
    decoder = Decoder(embed_dim=dec_input, units=units, vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    return (
        encoder,
        decoder,
        tokenizer,
        initialize_InceptionV3_image_features_extract_model(),
    )


encoder, decoder, tokenizer, image_model = load_model()


def predictions_from_model(image, max_sequence_len=35, attention_features_shape=64):
    attention_plot = np.zeros((max_sequence_len, attention_features_shape))

    hidden = decoder.init_state(batch_size=1)

    # temp_input = tf.expand_dims(
    #     preprocess_image(image)[0], 0
    # )  # process the input image to desired format before extracting features

    # img_tensor_val = image_model(
    #     temp_input
    # )  # Extract features using our feature extraction model

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.keras.applications.inception_v3.preprocess_input(input_arr)
    input_arr = tf.image.resize(input_arr, size=(299, 299))
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    img_tensor_val = image_model.predict(input_arr)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
    )

    features = encoder(
        img_tensor_val
    )  # extract the features by passing the input to encoder

    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result = []

    for i in range(max_sequence_len):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden
        )  # get the output from decoder

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][
            0
        ].numpy()  # extract the predicted id(embedded value) which carries the max value
        result.append(tokenizer.index_word[predicted_id])
        # map the id to the word from tokenizer and append the value to the result list

        if tokenizer.index_word[predicted_id] == "<end>":
            return result, attention_plot, predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[: len(result), :]
    return result, attention_plot, predictions


def generate_captions(test_image):
    # Testing on test image
    openImg = test_image
    # print(test_image)
    # real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
    result, attention_plot, pred_test = predictions_from_model(test_image)
    pred_caption = " ".join(result).rsplit(" ", 1)[0]

    st.markdown(f"## Generated Caption:")
    st.success(pred_caption)

    return pred_caption
    # plot_attmap(result, attention_plot, test_image)
    # plt.imshow(load_image(test_image)[0])


# https://github.com/android-iceland/streamlit-text-to-speech/blob/main/app.py
def convert_text_to_speech(text, output_language="en", tld="co.in"):

    tts = gTTS(text, lang=output_language, tld=tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name


def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)


# https://github.com/zarif101/image_captioning_ai/blob/master/main.py
# https://github.com/sankalp1999/Image_Captioning_With_Attention/blob/main/streamlit_app.py


def main():
    st.title("Eye for the Blind - An Image Caption Generator")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

    if st.button("Generate captions!"):
        pred_caption = generate_captions(image)

        # convert text to speech
        result = convert_text_to_speech(pred_caption)
        audio_file = open(f"temp/{result}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.markdown(f"## Your audio:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)


#     im = Image.open(openImg)
#     width, height = im.size
#     print(width,height)
#     div=3
#     if width > 3000:
#         div=10
#     im = im.resize((width//div, height//div))

# #    newsize = (200, 200)
#     #openImg = openImg.resize(200,200)
#     #return Image.open(openImg).resize(newsize)
#     return im


if __name__ == "__main__":
    main()
    remove_files(7)
