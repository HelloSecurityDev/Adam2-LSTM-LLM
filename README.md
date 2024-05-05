# Adam2-LSTM-LLM
a python based Long Short-Term Memory (LSTM) Language Model (LLM) named Adam2

The Adam2 LSTM LLM (Adam2 Long Short-Term Memory Language Model), created by [Adam Rivers](https://abtzpro.github.io) and [Hello Security](https://hellosecurityllc.github.io), is an advanced natural language processing (NLP) script written in Python. It is designed to generate human-like text based on input prompts, leveraging cutting-edge deep learning algorithms and the Long Short-Term Memory (LSTM) neural network architecture.

![Adam2 Logo](https://github.com/HelloSecurityDev/Adam2-LSTM-LLM/blob/main/Adam2-LSTM-LLM.png)

## Functionality:
This script provides a sophisticated solution for generating contextually relevant responses to input prompts. It excels in mimicking human-like dialogue and can be used for various applications, including conversational agents, content generation, and language learning.

## Dataset:
The Adam2 LSTM LLM script is trained on the Cornell Movie Dialogs Corpus dataset, which contains a large collection of movie dialogues. This dataset provides diverse and rich conversational data, enabling the model to learn and reproduce natural language patterns effectively.

## Training Process:
The script undergoes several steps during the training process:
1. **Dataset Loading and Preprocessing:** The dataset is downloaded and preprocessed to extract conversational text data. This involves tokenization, data augmentation, and sequence padding to prepare the data for training.
2. **Model Architecture Definition:** The LSTM neural network architecture is defined using TensorFlow and Keras. It consists of an embedding layer, followed by two LSTM layers, and a dense output layer. This architecture is optimized for sequence prediction tasks.
3. **Model Training:** The model is trained on the augmented dataset using the Adam optimizer and sparse categorical crossentropy loss function. Training involves multiple epochs to learn the intricate patterns and structure of human dialogue.
4. **Text Generation:** Once trained, the model can generate text based on input prompts. It predicts the next word in the sequence using the trained LSTM model and repeats the process to generate coherent and contextually appropriate responses.

## GUI Interface:
The script provides a graphical user interface (GUI) for easy interaction with the model. Users can input prompts via the GUI, and the generated responses are displayed in real-time. This intuitive interface enhances user experience and facilitates seamless communication with the AI model.

## Algorithm Overview:
The Adam2 LSTM LLM script utilizes several key algorithms and techniques:
- **Long Short-Term Memory (LSTM):** LSTM is a type of recurrent neural network (RNN) architecture that is well-suited for sequence prediction tasks. It maintains a memory state over time, allowing it to capture long-range dependencies in the input data. The LSTM model used in this script learns to predict the next word in a sequence of text based on the previous words.
- **Tokenization and Embedding:** Text data is tokenized into numerical sequences, with each word represented by a unique integer. The embedding layer converts these integer sequences into dense vectors, where words with similar meanings are mapped to nearby points in the embedding space.

## Use Case Scenarios:
The Adam2 LSTM LLM script can be applied in various use case scenarios:
- **Conversational Agents:** The generated text can be used to power chatbots and virtual assistants, providing natural and contextually relevant responses to user queries.
- **Content Generation:** Writers and content creators can use the script to generate dialogue snippets, story outlines, or creative writing prompts.
- **Language Learning:** The generated text can serve as practice material for language learners, helping them improve their comprehension and writing skills.

## Requirements
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- tkinter
- numpy

## Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/HelloSecurityDev/Adam2-LSTM-LLM
   ```
   then
   ```bash
   cd Adam2-LSTM-LLM
   ```
3. Install the required packages:
   ```bash
   pip install tensorflow tkinter numpy

4. Launch the script:
   ```bash
   python Adam2.py

## Contributing
Contributions to the project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Authors & Developers
Adam2 LSTM LLM AI was developed by [Hello Security](https://hellosecurityllc.github.io) and [Adam Rivers](https://abtzpro.github.io)

## Notes & Disclaimers
Adam2 is still in active development and changes are being made every day. there are bound to be errors, quirks, and unexpected bugs. Please report any issues or bugs experienced in the correct fashion.
