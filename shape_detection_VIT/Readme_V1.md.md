# shape detection on a VIT architcuture 

model used : https://huggingface.co/Engineer-Eslam/shape-recognizer-color 
trained on dataset gen-rated on following rules: 
Each image is generated within the following parameters :

1. A fixed size of 200x200 pixel
2. A random background colour
3. A random filling colour of each shape
4. A random rotation angle between -180° and 180°
5. A random position inside of the containing image
6. A random perimeter

model results : 
  • Accuracy: 99.17%
  • Precision: 0.9923
  • Recall: 0.9917
  • F1 Score: 0.9917

=> model file in models folder 
=> code file in code folder (model + testing) 

further improvements : 
1. more realtistic image geneartion methods 
2. better tranning timming 
3. Using GPU to train faster and on more data 