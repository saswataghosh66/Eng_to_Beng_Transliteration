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

Decoder class:
Defines the decoder module with an embedding layer, an RNN, and a linear layer to map hidden states to output vocabulary scores. It processes one step at a time, taking previous token and hidden state as input.

Seq2SeqModel class:
Combines the encoder and decoder into a sequence-to-sequence model. Handles initialization of decoder hidden states, runs teacher forcing during training, and supports beam search decoding for inference.

_init_decoder_hidden: Adapts encoder hidden states to decoder's format, handling bidirectional RNNs and LSTM's cell states.

_merge_bidirectional: Combines forward and backward hidden states by summation.

_pad_or_trim: Adjusts hidden state tensor size to match decoder layers.

beam_search_decode: Implements beam search decoding for generating sequences with better search than greedy decoding.

train_model function:
Trains the Seq2Seq model for one epoch over the data loader. Performs forward pass, computes loss, backpropagation, and optimizer step. Returns updated model and average training loss.

evaluate_model function:
Evaluates the model on validation/test data without teacher forcing. Computes loss and token-level accuracy excluding padding tokens. Returns average loss and accuracy percentage.
 



