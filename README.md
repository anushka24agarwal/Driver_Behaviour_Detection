# Driver Behavior Detection Model

This project focuses on classifying driver behavior into six categories: Safe Driving, Talking on the Phone, Texting on the Phone, Turning, Yawning, and Engaging in Other Activities. We utilize deep learning techniques to create an image classification model based on the DenseNet architecture.

## Dataset

The dataset consists of images collected from drivers engaged in various activities while driving. The data is organized into six categories:

1. **Safe Driving**
2. **Talking**
3. **Texting on the Phone**
4. **Turning**
5. **Yawning**
6. **Engaging in Other Activities**

Each image represents a snapshot of a driver's behavior during a specific activity. The dataset was divided into training and validation sets for model evaluation.

## Model Architecture

The model uses the **DenseNet** pre-trained on **ImageNet** architecture for image classification. DenseNet is known for its dense connections between layers, which helps in improving gradient flow and feature reuse.

### Custom Layers

1. **Base Model: DenseNet121**
   - DenseNet121, pre-trained on ImageNet, was used as the base model. We removed the top classification layer to tailor it for our task of classifying six specific driver behaviors.
   
2. **GlobalAveragePooling2D Layer**
   - A GlobalAveragePooling2D layer is used to reduce the spatial dimensions of the model's output, converting the features into a fixed-size vector while retaining important information for classification.

3. **Dense Layer (256 units, ReLU activation)**
   - A dense layer with 256 units and **ReLU** activation was added to further extract features from the global pooled output and introduce non-linearity to the model.

4. **Dropout Layer (rate: 0.5)**
   - A **Dropout** layer with a dropout rate of 50% was added to prevent overfitting during training, helping the model generalize better by randomly setting half of the features to zero.

5. **Final Dense Layer (6 units, Softmax activation)**
   - The final layer is a **Dense** layer with 6 units (one for each class) and a **Softmax** activation function to output probabilities for the 6 behavior categories. The class with the highest probability is selected as the model's prediction.

### Model Performance

The model achieved the following performance:

- **Training Accuracy**: 97.21%
- **Training Loss**: 0.0884
- **Validation Accuracy**: 98.82%
- **Validation Loss**: 0.0449
- **Test Accuracy**: 98.42%
- **Test Loss**: 0.0400

The high accuracy across training, validation, and test sets demonstrates the model's ability to classify driver behavior with great precision.

## Frameworks and Libraries Used

- **TensorFlow** / **Keras**: For building, training, and evaluating the deep learning model.
- **OpenCV**: For image preprocessing and augmentation.
- **NumPy**: For numerical operations on the data.
- **Matplotlib**: For visualizing the training process and model performance.
- **Pandas**: For handling dataset and labels.

### Output

![Sleepy](path_to_image_1.png)
*A person yawning while driving car captured as sleepy*

### Training and Evaluation Process

![Turning](path_to_image_2.png)
*A person turning and talking while driving car captured as turning.*
