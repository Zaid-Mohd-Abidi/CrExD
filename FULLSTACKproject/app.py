from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define data paths
train_data_dir = "/home/zaidabidi/Downloads/train"
test_data_dir = "/home/zaidabidi/Downloads/test"

# Define image data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Batch size for the generators
batch_size = 32

# Create the train and test generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

app = Flask(__name__)

# Load the trained model
model = load_model("/home/zaidabidi/model.h5")

# Initialize the LIME explainer
explainer = lime_image.LimeImageExplainer()

# Define the number of superpixels to use in the explanation (adjust this if needed)
num_superpixels = 250

# Define class labels for the model's predictions
class_labels = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Function to predict classes for a batch of images using the model
def predict_classes(images):
    predictions = model.predict(images)
    return predictions

@app.route('/hi')
def hello():
    print("hi")
    return "hi"

@app.route('/get_prediction/<path:urlpath>')
def get_prediction(urlpath):
    path = "/home/zaidabidi/Downloads/test/" + urlpath
    print(urlpath)
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    img_array = img_array.reshape((1,) + img_array.shape)

    # Get the model's prediction for the image
    prediction_model = model.predict(img_array)
    class_index = prediction_model.argmax()
    prediction_label = class_labels[class_index]

    # Explain the model's prediction for the selected image using LIME
    explanation = explainer.explain_instance(
        img_array[0],
        predict_classes,
        top_labels=5,
        hide_color=0,
        num_samples=700,
        num_features=num_superpixels
    )

    # Get the explanation for the top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_superpixels,
        hide_rest=False
    )

    # Save the LIME explanation image
    lime_explanation_path = "lime_explanation.png"
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.savefig(lime_explanation_path)
    plt.close()

    data_drift_batches = 20  # Number of batches to monitor for data drift
    data_p_values = []
    for _ in range(data_drift_batches):
        batch_images, _ = next(train_generator)
        flattened_batch_images = np.ravel(batch_images)
        _, p_value = ks_2samp(flattened_batch_images, np.ravel(train_generator[0][0]))
        data_p_values.append(p_value)

    data_drift_p_value = np.mean(data_p_values)
    #print(f'Train Generator Data drift p-value: {data_drift_p_value}')

    # Data drift monitoring for test generator
    test_data_drift_batches = 20  # Number of batches to monitor for data drift
    test_data_p_values = []
    for _ in range(test_data_drift_batches):
        batch_images, _ = next(test_generator)
        flattened_batch_images = np.ravel(batch_images)
        _, p_value = ks_2samp(flattened_batch_images, np.ravel(test_generator[0][0]))
        test_data_p_values.append(p_value)

    test_data_drift_p_value = np.mean(test_data_p_values)
    #print(f'Test Generator Data drift p-value: {test_data_drift_p_value}')

    # Concept drift monitoring
    concept_drift_batches = 20  # Number of batches to monitor for concept drift
    concept_predictions = []
    for _ in range(concept_drift_batches):
        batch_images, _ = next(train_generator)
        batch_predictions = model.predict(batch_images)
        concept_predictions.extend(np.argmax(batch_predictions, axis=1))

    true_labels = train_generator.classes[:len(concept_predictions)]
    true_labels = np.reshape(true_labels, (-1, 1))

    concept_drift_score = accuracy_score(true_labels, concept_predictions)
    #print(f'Concept drift score: {concept_drift_score}')

    # Plotting data drift comparison
    data_drift_plot_path = "data__.png"
    plt.plot(range(data_drift_batches), data_p_values, label='Train Generator Data Drift')
    plt.plot(range(test_data_drift_batches), test_data_p_values, label='Test Generator Data Drift')
    plt.xlabel('Batches')
    plt.ylabel('P-value')
    plt.legend()
    plt.savefig(data_drift_plot_path)
    #plt.show()
    plt.close()

    return render_template('in.html', user_image=urlpath, prediction=prediction_label, lime_explanation=lime_explanation_path,tr_drift=data_drift_p_value,ts_drift=test_data_drift_p_value,cs_drift=concept_drift_score,drift__path=data_drift_plot_path)

if __name__ == "__main__":
    print("Starting python flask server for image prediction")
    app.run()


