# **Aim**:Character-Level Transliteration using RNN-based Seq2Seq Models on the [Dakshina Dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar) .

This dataset consists of 12 Indian languages. Since my mother tongue is Bengali, I have chosen English–Bengali transliteration for this project.
# **Key Steps to Achieve the Aim:**
[The code is organized and explained step-by-step in the Jupyter Notebook. Each step is numbered for clarity. In this README file, I have described the implementation by referring to the corresponding step numbers from the notebook. This makes it easier to follow the logic and match the explanation with the actual code.]

**Step 1:**

I have loaded the file paths for the training, validation, and test datasets.

**Step 2:**

# This function loads and preprocesses the transliteration data from a given file path.
# It performs the following steps:
# 1. Reads the TSV file into a pandas DataFrame and removes any unnecessary columns or missing values.
# 2. Converts all words to strings and resets the DataFrame index.
# 3. Computes the maximum lengths of input and target words for padding.
# 4. Builds character-level vocabularies for both the input (Roman script) and target (Bengali script),
#    including special tokens: <pad>, <sos>, and <eos>.
# 5. Encodes each input and target word into a fixed-length sequence of token IDs.
#    - Input sequences are padded to max_input_len.
#    - Target sequences are padded to max_target_len and prefixed with a <pad> token.
# 6. Converts the sequences into PyTorch tensors.
# 7. Wraps them in a TensorDataset and DataLoader for use in training or evaluation.

# The function returns:
# - The TensorDataset,
# - The DataLoader,
# - The input and target vocabularies,
# - The maximum input and target sequence lengths.



