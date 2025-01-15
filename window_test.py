import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import home_made
from my_descent_td1 import GradientDescent


# Function to update the canvas with the mouse
def paint(event):
    # Get the coordinates of the mouse click within the pixel grid
    col = event.x // pixel_size
    row = event.y // pixel_size

    # Make sure the drawn pixel is within bounds
    if 0 <= col < width and 0 <= row < height:
        canvas.create_rectangle(col * pixel_size, row * pixel_size,
                                (col + 1) * pixel_size, (row + 1) * pixel_size, fill='white', outline='white')
        # Draw on the image (set pixel to white)
        image.putpixel((col, row), 255)

# Function to clear the canvas and reset the image


def clear_canvas():
    canvas.delete("all")
    global image
    # Reset the image to black (0 = black)
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    # Recreate the grid of pixels on the canvas
    for i in range(width):
        for j in range(height):
            canvas.create_rectangle(i * pixel_size, j * pixel_size,
                                    (i + 1) * pixel_size, (j + 1) * pixel_size, fill='black', outline='gray')

# Function to predict the drawn digit


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_digit():
    # Preprocess the image (normalize)
    img_array = np.array(image)  # Convert to numpy array
    img_array = img_array.flatten()  # Flatten the 8x8 array to 1D

    # Normalize the data (matching preprocessing from training)
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    # Reshape to match the model's input shape
    img_array = img_array.reshape(1, -1)

    # Use the trained sklearn model to make a prediction
    sklearn_prediction = model_sklearn.predict(img_array)[0]

    # Now, make a prediction using your own GradientDescent model (user's model)
    # Use the same `theta_optimized` that was trained earlier
    # Apply sigmoid to the result
    user_prediction_prob = sigmoid(
        np.dot(img_array, home_made.theta_optimized))
    user_prediction = (user_prediction_prob >= 0.5).astype(
        int)  # Binary classification threshold at 0.5
    user_prediction = user_prediction[0]  # Get the final prediction

    # Display both predictions
    result_label.config(text=f"Your Prediction: {
                        user_prediction}\nSklearn Prediction: {sklearn_prediction}")


# Parameters for the pixel canvas
width, height = 8, 8  # Size of the image (8x8 pixels)
pixel_size = 40  # Size of each "pixel" on the canvas

# Create a blank image for drawing (initially black)
image = Image.new('L', (width, height), 0)  # 'L' mode for grayscale, 0 = black
draw = ImageDraw.Draw(image)

# Creating the main window
root = tk.Tk()
root.title("Draw a Digit")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=width * pixel_size,
                   height=height * pixel_size, bg='black')
canvas.pack()

# Bind the mouse events to the canvas
canvas.bind("<B1-Motion>", paint)

# Add a button to clear the canvas
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Add a button to predict the digit
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

# Label to display the prediction result
result_label = tk.Label(
    root, text="Your Prediction Digit: \nSklearn Prediction", font=("Arial bold", 16))
result_label.pack(side="left", padx=0)

# Initialize the trained model
# Load digits dataset from sklearn
digits = datasets.load_digits()
X, y = digits.data, digits.target
# Scaling the data to match input range
X_scaled = StandardScaler().fit_transform(X)
model_sklearn = LogisticRegression(max_iter=1000)
model_sklearn.fit(X_scaled, y)  # Train the logistic regression model

# Initialize the grid of pixels on the canvas
clear_canvas()

# Run the Tkinter event loop
root.mainloop()
# pew
