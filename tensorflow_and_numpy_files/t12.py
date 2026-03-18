import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.tokenize import word_tokenize
import spacy
from spacy import displacy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

# Create the main window
root = tk.Tk()
root.title("YUI Chat")

# Create a text box to display the conversation
conversation_text = scrolledtext.ScrolledText(root, height=20, width=80)
conversation_text.pack()

# Create a text entry box for user input
input_entry = tk.Text(root, height=5, width=40)
input_entry.pack()

# Define the interaction function
def interact(input_text):
    # Tokenize the input text
    tokens = word_tokenize(input_text)

    # Process the tokens using spaCy
    doc = spacy.tokens.Doc(tokens)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Generate a response using the YUI model
    input_seq = np.array([[1.0] * len(entities)])
    output = model.predict(input_seq)

    # Return the response
    return output[0][0]

# Load the YUI model
model = Sequential()
model.add(LSTM(128, input_shape=(1000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('yui_model.h5')

# Define the send button function
def send_button_clicked():
    user_input = input_entry.get("1.0", "end-1c")
    response = interact(user_input)
    conversation_text.insert("end", "You: " + user_input + "\n")
    conversation_text.insert("end", "YUI: " + str(response) + "\n")
    input_entry.delete("1.0", "end")

# Create the send button
send_button = tk.Button(root, text="Send", command=send_button_clicked)
send_button.pack()

# Start the Tkinter event loop
root.mainloop()
