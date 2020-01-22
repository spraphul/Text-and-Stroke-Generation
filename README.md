# Text-and-Stroke-Generation

Based on the research paper [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)

#### Key includes:

1. Mixed Density Network implementation(two normal and one bernoulli)
2. Encoder Decoder LSTM Architecture 
3. Attention Layer 
4. Probabilistic Loss Function as discussed in the paper


#### Dataset Format:
1. Strokes Data : A series of vectors of length=3. First coordinate is a boolean value which tells whether the pen lifts in the air or not. The next two coordinates are the relative differences in x and y coordinates.

2. Text Data : String
