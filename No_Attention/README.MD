# **Aim**:Character-Level Transliteration using RNN-based Seq2Seq Models on the [Dakshina Dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar) .

This dataset consists of 12 Indian languages. Since my mother tongue is Bengali, I have chosen English–Bengali transliteration for this project.
# **Key Steps to Achieve the Aim:**
[The code is organized and explained step-by-step in the Jupyter Notebook. Each step is numbered for clarity. In this README file, I have described the implementation by referring to the corresponding step numbers from the notebook. This makes it easier to follow the logic and match the explanation with the actual code.]

**Step 1:**

I have loaded the file paths for the training, validation, and test datasets.

**Step 2:**

This function is responsible for loading and preprocessing the transliteration data from a specified file path. It follows a systematic approach:

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



