# Cross-Modal-Representation-Learning
 
Most real-world problems are characterized by data simultaneously collected from several sensors. For
instance, an activity can be recorded by a video camera with image and audio sensors. Web pages contain
text, images, audio clips, tables, all of which describe a related concept in that document. Image collections
often contain tags or even complete captions written in natural text that describe the content of those images.
In machine learning, multi-view analysis refers to the setting where data about a single
concept comes in multiple views (image, text, audio signal, graph, table, . . . ).
This project explore models and methods related to cross-modal representation learning, where
the goal is to learn a common representation from multiple views. Learning this
representation is essential for several key tasks:
1. **Cross-modal retrieval**: Given a query in one view (e.g., text), retrieve similar instances from the gallery
in another view (e.g., images).
2. **Cross-modal translation**: Given an instance in one view (e.g., image), reconstruct that instance in
another view (e.g., produce a text caption for the given image).
3. **Cross-modal alignment**: Given different views of an instance, align the subsets of features in the two
views that correspond to each other.

## Part 1 : Linear-Multi-View Representation Learning
We use the Recipe 1 Million Dataset which consists of approximately 1 Million text recipes with titles, instructions and ingredients in English. The images dataset containing 800K recipe images was downloaded from im2recipe webpage. We convert the text data into feature vectors of size 768 using Bert. We extract features of 4 different types of text - title, ingredients, instructions and all of them concatenated and study their correlation with images. We train a Canonical Correlation Analysis (CCA) model on the extracted features of the training text-image pairs. CCA is used to find the latent space where the objective is to maximize the correlation between a linear combination of text features and linear combination of image features.

First Header                    | R @ 1  | R @ 5 | R @ 10|
--------------------------------| ------ |------ |-------|
**Generate Features(dim = 50)** |        |       |       |
Title                           | 3.29   | 11.46 | 18.19 |
Ingredients                     | 4.13   |14.15  | 21.43 | 
Instructions                    | 4.38   |15.15  | 22.39 |
All                             | **4.9**|16.53  | 25.34 | 
**Given Features(dim = 50)**    |        |       |       |
Title                           | 12.79  |36.02  | 49.42 | 
Ingredients                     | 12.8   |36.65  | 50.49 |
Instructions                    | 23.29  |52.5   | 65.85 | 
All                             | 34.9   |68.5   | 80.4  |
**Given Features(dim = 768)**   |        |       |       |
Title                           | 20.43  | 38.91 | 44.8  | 
Ingredients                     | 20.45  | 39.76 | 46.06 |
Instructions                    | 35.58  | 57.66 | 64.24 | 
All                             |**55.03**|**77.92**|**82.2**|

From the above results we can see that a concatenation of Title + Ingredients + Instructions finds a lot more correlation with images. In general, it can also be seen that instructions and ingredients find a lot more correlation than title owing to the fact that ingredients and instructions contains many pseudo labels for it's corresponding images.

<p float="left">
  <img src="https://github.com/AninditaChavan/Cross-Modal-Representation-Learning/assets/20729102/c5f3723c-d138-4a9d-bbb5-0d437b213dae" width="300" />
  <img src="https://github.com/AninditaChavan/Cross-Modal-Representation-Learning/assets/20729102/8d389610-fc7b-4fd7-939f-f5d7ba892f44" width="300" /> 
  <img src="https://github.com/AninditaChavan/Cross-Modal-Representation-Learning/assets/20729102/dd847d25-05d8-412d-a714-0a871d620da2" width="300" />
</p>

## Part 2 : Non-Linear-Multi-View Representation Learning
## Part 3 : Multi-View Model for Video-Text alignment
