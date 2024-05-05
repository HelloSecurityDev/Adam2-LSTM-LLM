import os
import numpy as np
import tkinter as tk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and preprocess the dataset
def load_dataset():
    dataset = tf.keras.utils.get_file(fname="cornell_movie_dialogs.zip",
                                      origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
                                      extract=True)
    dataset_folder = os.path.join(os.path.dirname(dataset), "cornell movie-dialogs corpus")
    conversations_file = os.path.join(dataset_folder, "movie_lines.txt")
    conversations = open(conversations_file, encoding='utf-8', errors='ignore').read().split('\n')
    conversations = [conv.split(' +++$+++ ')[-1] for conv in conversations]
    return conversations

# Step 2: Data augmentation (simple shuffle)
def augment_data(data):
    np.random.shuffle(data)
    return data

# Step 3: Define and train the LSTM model
def build_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 256, input_length=max_length-1))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 4: Generate text using the trained model
def generate_text(model, tokenizer, max_length, seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Step 5: GUI Interface for interacting with the model
def generate_response(input_text):
    global model, tokenizer, max_length
    response = generate_text(model, tokenizer, max_length, input_text, 50)
    return response

def send_message(event=None):
    input_text = entry.get()
    response = generate_response(input_text)
    conversation.config(state=tk.NORMAL)
    conversation.insert(tk.END, f"User: {input_text}\n")
    conversation.insert(tk.END, f"AI: {response}\n\n")
    conversation.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

def GUI_interface():
    global model, tokenizer, max_length

    root = tk.Tk()
    root.title("AI Chat Interface")
    root.geometry("600x400")

    conversation = tk.Text(root, wrap="word")
    conversation.config(state=tk.DISABLED)
    conversation.pack(expand=True, fill="both")

    entry = tk.Entry(root)
    entry.pack(side="bottom", fill="x")
    entry.bind("<Return>", send_message)

    send_button = tk.Button(root, text="Send", command=send_message)
    send_button.pack(side="bottom")

    root.mainloop()

def main():
    # Step 1: Load and preprocess the dataset
    data = load_dataset()
    augmented_data = augment_data(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(augmented_data)
    vocab_size = len(tokenizer.word_index) + 1
    input_sequences = tokenizer.texts_to_sequences(augmented_data)
    max_length = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length, padding='pre'))

    # Step 2: Define training and evaluation data
    X_train = input_sequences[:, :-1]
    y_train = input_sequences[:, -1]

    # Step 3: Define and train the LSTM model
    model = build_model(vocab_size, max_length)
    model.fit(X_train, y_train, epochs=10, batch_size=128)

    # Step 4: Generate text using the trained model
    print(generate_text(model, tokenizer, max_length, "I", 50))

    # Step 5: GUI Interface for interacting with the model
    GUI_interface()

if __name__ == "__main__":
    main()
