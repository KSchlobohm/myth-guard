# myth-guard
A non-production learning project that explores different approaches to object detection.

## Glossary

- *Classification*: the process of categorizing and labeling visual data, such as images or video frames, into predefined classes based on their visual content

- *COCO dataset*: Common Objects in Context dataset, a large-scale object detection, segmentation, and captioning dataset. COCO has 80 object categories that cover common objects found in everyday life, such as people, animals, vehicles, and household items. The dataset consists of 328,000 images and 2.5 million labeled instances.

- *Object detection*: identifying and locating multiple objects within an image or video frame by drawing bounding boxes around each detected object and classifying them into predefined categories.

- *Ontology*: an Ontology defines how your Base Model is prompted, what your Dataset will describe, and what your Target Model will predict. A simple Ontology is the CaptionOntology which prompts a Base Model with text captions and maps them to class names. Other Ontologies may, for instance, use a CLIP vector or example images instead of a text caption.

- *Segmentation*: partitioning an image into multiple segments or regions, each corresponding to different objects or parts of objects, to precisely delineate their boundaries and classify each pixel accordingly