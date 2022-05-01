# Deep Learning HomeWork 3

## RMM & Transformer

### Difference between preprocessing text and image 

- we need to `tokenize the text` when we proprocess the text data, because the text can not be easily converted to the number(or in other words, the tensor). we can not simply use the alphabet to represent a word. because every word has a meaning and the meaning is not relevant to the order of the letters. AS a result, we use a number(id) to represent a word, which is called `tokenize`.
- when we preprocess the image data. we nearly do not change the form of the picture. we just need to `tranform` the IMG file into a `tensor`, sometimes we need to `flatten the tensor`, `normalize the tensor`, `image augmentation` or some special technologies.
- we need to add a sign to represent `the start/end of the sentence` (`<eos>`) to let the AI know the where the sentence is becuase the sentences often have `different lengths`. while the image can be easily converted to the same-size ones.

