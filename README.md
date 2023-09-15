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
1. **Cross-modal retrieval**: given a query in one view (e.g., text), retrieve similar instances from the gallery
in another view (e.g., images).
2. **Cross-modal translation**: given an instance in one view (e.g., image), reconstruct that instance in
another view (e.g., produce a text caption for the given image).
3. **Cross-modal alignment**: given different views of an instance, align the subsets of features in the two
views that correspond to each other.
