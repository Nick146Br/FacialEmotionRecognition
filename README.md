# FacialEmotionRecognition
A Graph Neural Network model to identify facial emotions

# Pre-Processing
From each image was extracted 26 facial points, and each of these points constituted a graph. The edges for this graph was created using the delaunay technique.

# Model
The model used was a graph neural network, which exchanges messages between the nodes and has the hability to learn in a non-euclidean space. More specifically in this project we used
a variant of GCN (Graph Convolution Networks) known as SSGC (Simple Spectral Graph Convolution).