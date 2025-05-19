# **Aim**:Character-Level Transliteration using RNN-based Seq2Seq Models on the [Dakshina Dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar) .

This dataset consists of 12 Indian languages. Since my mother tongue is Bengali, I have chosen English–Bengali transliteration for this project.
# **Key Steps to Achieve the Aim:**
[The code is organized and explained step-by-step in the Jupyter Notebook. Each step is numbered for clarity. In this README file, I have described the implementation by referring to the corresponding step numbers from the notebook. This makes it easier to follow the logic and match the explanation with the actual code.]

**Step 1:**

I have loaded the file paths for the training, validation, and test datasets.

**Step 2:**

load_and_prepare_data function is responsible for loading and preprocessing the transliteration data from a specified file path. It follows a systematic approach:

- The transliteration data is read from a TSV file into a pandas DataFrame. Any unnecessary columns (such as the third dummy column) and missing values are removed.
- All input and target words are explicitly converted to string format, and the index of the DataFrame is reset.
- The maximum lengths of the input (Roman script) and target (Bengali script) words are calculated to enable uniform padding.
- Character-level vocabularies are created for both the input and target languages. These vocabularies include three special tokens: <pad> (padding), <sos> (start of sequence), and <eos> (end of sequence).
- Each word is encoded into a sequence of token IDs based on the corresponding vocabulary:
- Input words are padded to the maximum input length.
- Target words are padded to the maximum target length, with additional padding tokens added as needed.
- The encoded sequences are converted into PyTorch tensors.

These tensors are wrapped into a TensorDataset and a DataLoader, making them ready for use in training or evaluation.

- The function returns the following:
- The constructed TensorDataset,
- The DataLoader for batching,
- The input and target vocabularies (as dictionaries),
- The maximum lengths of the input and target sequences.

**Step 3:**

 Loaded traiining data from traiining data path.

**Step 4:**

load_and_prepare_test_valid_data loads and preprocesses the validation or test transliteration data from a specified file path. It accepts existing input and target vocabularies as well as maximum sequence lengths, allowing consistent encoding between training and evaluation phases.

The process involves:

- Reading the TSV file into a pandas DataFrame and cleaning it by removing unnecessary columns and missing values.
- Converting all words to strings and resetting the DataFrame index.
- If vocabularies are not provided, building character-level vocabularies for both input (Roman script) and target (Bengali script) languages, including special 
  tokens <pad>, <sos>, and <eos>.
- Determining the maximum lengths of input and target sequences if not provided.
- Encoding each input word into a fixed-length sequence of token IDs padded to the maximum input length.
- Encoding each target word as a sequence with <sos> and <eos> tokens added at the beginning and end respectively, then padded to the maximum target length plus 2 
  (for these special tokens).
- Converting all sequences into PyTorch tensors.
- Wrapping the tensors into a TensorDataset and DataLoader with the specified batch size for evaluation purposes.

The function returns the prepared dataset and dataloader along with the vocabularies and maximum sequence lengths, ensuring compatibility with the training setup.

**step 5:**

  Loaded validation and test data from validation and test data path.
  I used a different function to load the validation and test data because there were two Bengali characters missing from their vocabularies. Using the same 
  function as for training data could have caused issues due to these missing characters, so the separate function ensures proper handling of the vocabularies and 
  consistent encoding across datasets.

**Step 6:**

- TextEncoder class:

  Defines the encoder module with an embedding layer and an RNN (LSTM/GRU/RNN). It supports multiple layers, dropout, and optional bidirectionality. Takes input 
  sequences and outputs hidden states.

- Decoder class:

  Defines the decoder module with an embedding layer, an RNN, and a linear layer to map hidden states to output vocabulary scores. It processes one step at a 
  time, 
  taking previous token and hidden state as input.

- Seq2SeqModel class:

  Combines the encoder and decoder into a sequence-to-sequence model. Handles initialization of decoder hidden states, runs teacher forcing during training, and 
  supports beam search decoding for inference.

- _init_decoder_hidden:  Adapts encoder hidden states to decoder's format, handling bidirectional RNNs and LSTM's cell states.

- _merge_bidirectional:  Combines forward and backward hidden states by summation.

- _pad_or_trim:  Adjusts hidden state tensor size to match decoder layers.

- beam_search_decode:  Implements beam search decoding for generating sequences with better search than greedy decoding.

- train_model function:

    Trains the Seq2Seq model for one epoch over the data loader. Performs forward pass, computes loss, backpropagation, and optimizer step. Returns updated model 
    and average training loss.

- evaluate_model function:

    Evaluates the model on validation/test data without teacher forcing. Computes loss and token-level accuracy excluding padding tokens. Returns average loss and 
    accuracy percentage.

**Step 7:**

 - W&B Setup Code:

     - Imports necessary libraries: wandb for experiment tracking, numpy and random for computations, and getpass for secure input.
     - Uses getpass to securely prompt the user to enter their Weights & Biases (W&B) API key without showing it on the screen.
     - Logs into W&B using the provided API key to enable experiment tracking and logging.


 **Step 8:**

- W&B Hyperparameter Sweep Configuration:

     - Defines a Bayesian optimization sweep to find the best model hyperparameters.
     - The goal is to maximize the validation accuracy (val_accuracy).
     - Specifies ranges/values for key hyperparameters such as embedding size, dropout rate, number of encoder and decoder layers, hidden layer size, RNN cell 
       type, bidirectionality, batch size, number of epochs, learning rate, and beam search width.
     - Creates a sweep on the W&B platform under the project named 'DL_Translation', returning a sweep_id to identify this sweep.


**Step 9:**

- In the main function:
    - Initializes a W&B run with current hyperparameters.
    - Builds the Seq2Seq model using those hyperparameters.
    - Sets loss function (CrossEntropy) and Adam optimizer.
    - Loads training and validation data loaders.
    - Runs training and validation for a fixed number of epochs.
    - Logs training loss, validation loss, and validation accuracy to W&B each epoch.
    - Prints progress to the console.

The W&B agent runs this function multiple times with different hyperparameters to optimize the model.


**Step 10:**

   Here, I am building the architecture of my best model obtained from the W&B sweep. This model is a Seq2Seq neural network with an encoder and decoder, using 
   the hyperparameters that gave the best performance during the W&B optimization. The encoder processes the input sequence with a specified number of layers, 
   hidden size, embedding size, cell type (like GRU, LSTM, or RNN), and can be bidirectional. The decoder generates the output sequence using similar settings. 
   Dropout is applied for regularization, and beam search decoding is included to improve inference results. This architecture reflects the optimized 
   configuration from the W&B runs.

**Step 11:**

   Here, I trained my best model for up to 35 epochs, printing the training loss and accuracy, as well as the validation loss and accuracy, after each epoch.

**Step 12:**

   I evaluated my model on the test dataset.

**Step 13:**

- Function run_inference:

    - Sets the model to evaluation mode to disable dropout and other training-specific behaviors.
    - Iterates over the test dataloader without computing gradients.
    - For each batch:
       - Moves input (latin) and target (devanagari) tensors to the specified device (CPU/GPU).
       - Runs the model to get output predictions with teacher forcing turned off (teacher_forcing_ratio=0).
       - Converts target tensors to NumPy arrays and collects them as actual ground truth.
       - Extracts the predicted token indices from the model output by applying argmax on the appropriate dimension.
       - Converts both inputs and predicted outputs back to NumPy arrays.
       - Appends tuples of input and predicted outputs to the predictions list.
    - Returns the lists of predictions and actual targets for further analysis.

- Vocab dictionaries:

    - latin_idx2token: A dictionary mapping integer indices back to their corresponding Latin characters from the test input vocabulary.
    - bangla_idx2token: A dictionary mapping integer indices back to their corresponding Bengali characters from the test target vocabulary.

**Step 14:**

- Function decode_sequence:

    - Takes a list of token indices (indices), a dictionary mapping indices to tokens (idx2token), and the target vocabulary dictionary (target_vocab).

    - Filters out special tokens like padding (<pad>), start-of-sequence (<sos>), and end-of-sequence (<eos>) from the indices.

    - Converts the remaining valid indices back to their corresponding characters using idx2token.

    - Joins these characters into a single string.

   - Returns the decoded string representing the original text sequence.

**Step 15:**

- Function process_output_indices:

    - Takes a list of token indices (indices), a dictionary mapping indices to tokens (idx2token), and the target vocabulary dictionary (target_vocab).
    - Iterates through each index in the sequence.
    - Stops decoding when it encounters the end-of-sequence token (<eos>).
    - Skips special tokens like padding (<pad>) and start-of-sequence (<sos>).
    - Converts each valid index to its corresponding character using idx2token.
    - Concatenates these characters to form the decoded string.
    - Returns the decoded text representing the sequence up to the <eos> token.

**Step 16:**

  This section of code performs inference using the best trained Seq2Seq model and then visualizes the model's transliteration performance at the character level.

# **Result:**

 **Model Training and Evaluation Summary:**

After conducting approximately 15 hyperparameter sweeps, we identified a model configuration that achieved the highest training accuracy of 86.33%(tokenwise) within 5 epochs.
 - embed_size = 128
 - hidden_size = 512
 - encoder_layers = 3
 - decoder_layers = 2
 - cell_type = 'lstm'
 - batch_size = 16
 - num_epochs = 35
 - dropout = 0.2
 - beam_width= 5
 - learning_rate = 0.0001
 - bidirectional = True

Subsequently, we extended the training of this model to 35 epochs, observing the following performance metrics(Step 12 in the notebook):

 - Peak validation accuracy: 47.38% (achieved at epoch 35)
 - Final test accuracy: 47.74% (after 35 epochs) [It is a clear indication of overfitting.]

- After that, I calculated the exact word-level accuracy, which measures how many complete words the model predicted correctly without any character mismatch. The results are as follows:

  -  Prediction Accuracy Summary
      - Total Predictions: 9228
         - ✅ 100% Match: 2024
         - ✅ Above 75%: 1601
         - ✅ Above 50%: 1580
         - ✅ Above 25%: 1869
         - ❌ ≤ 25% Match: 2154
 So, based on this, the exact word-level accuracy is approximately 21.9% (2024 out of 9228 words were perfectly predicted).








