import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Frame, Menu, Canvas, Toplevel
from PIL import Image, ImageTk
import requests
import os
import torch
import json
import cv2  # OpenCV library
from PIL import Image, ImageTk
from ultralytics import YOLO  # YOLO model import

# URL of the Flask server
FLASK_URL = "http://127.0.0.1:5000/predict"
model = YOLO('best.pt')  # Load your custom YOLO model here

# List to store past predictions
past_predictions = []

# Settings
app_resolution = (800, 600)
selected_model = 'one'

# Function to save past predictions to a file
def save_predictions():
    with open('past_predictions.json', 'w') as f:
        json.dump(past_predictions, f, indent=4)

# Function to load past predictions from a file
def load_predictions():
    global past_predictions
    if os.path.exists('past_predictions.json'):
        with open('past_predictions.json', 'r') as f:
            past_predictions = json.load(f)

# Function to upload and classify an image
# Function to upload and classify an image
def upload_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

    # Prepare the image for sending to the server
    with open(file_path, 'rb') as img_file:
        files = {'file': img_file}
        data = {'model': selected_model}
        response = requests.post(FLASK_URL, files=files, data=data)

    if response.status_code == 200:
        predictions = response.json()
        show_popup(predictions, selected_model)
        past_predictions.append({'file_path': file_path, 'predictions': predictions, 'model': selected_model})
        save_predictions()
    else:
        messagebox.showerror("Error", "Failed to get prediction from the server.")


# Function to start webcam capture and perform YOLO prediction
def start_webcam():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        messagebox.showerror("Error", "Couldn't open the webcam.")
        return

    # Create a window for webcam output in tkinter
    webcam_window = tk.Toplevel(root)
    webcam_window.title("Webcam YOLO Prediction")

    # Create a label for displaying the webcam frame in the GUI
    webcam_label = tk.Label(webcam_window)
    webcam_label.pack()

    # Capture frames from the webcam in a loop
    def capture_frame():
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to grab frame.")
            cap.release()
            return

        # Run inference using the YOLO model
        results = model(frame)  # Perform inference on the current frame

        # Get bounding boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        # Draw bounding boxes and labels
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {int(class_id)}, Conf: {confidence:.2f}",
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to a format suitable for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        # Update the Tkinter window with the latest frame
        webcam_label.config(image=img_tk)
        webcam_label.image = img_tk

        # Continue capturing frames
        webcam_window.after(10, capture_frame)

    # Start capturing frames
    capture_frame()

# Function to show a popup window with the predicted classes and probabilities
def show_popup(predictions, model_name):
    message = f"Using Model: {model_name}\n\nThe predicted classes are:\n"
    for pred in predictions:
        label = pred['label']
        score = pred['score'] * 100  # Convert to percentage
        message += f"{score:.2f}% {label}\n"
    
    messagebox.showinfo("Prediction Result", message)

# Function to show past predictions in the main window
def show_past_predictions():
    for widget in root.winfo_children():
        widget.destroy()

    past_frame = tk.Frame(root)
    past_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(past_frame)
    scrollbar = tk.Scrollbar(past_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for prediction in past_predictions:
        img = Image.open(prediction['file_path'])
        img = img.resize((100, 100))
        img = ImageTk.PhotoImage(img)
        img_label = tk.Label(scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        img_label.pack(pady=5)

        model_name = prediction.get('model', 'Unknown')
        tk.Label(scrollable_frame, text=f"Model: {model_name}").pack()

        for pred in prediction['predictions']:
            label = pred['label']
            score = pred['score'] * 100  # Convert to percentage
            tk.Label(scrollable_frame, text=f"{score:.2f}% {label}").pack()

        tk.Label(scrollable_frame, text="").pack()  # Add empty space between entries

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    back_button = tk.Button(past_frame, text="Back to Main Menu", command=show_main_menu, width=20, height=2)
    back_button.pack(pady=10)

# Function to center the window on the screen
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f'{width}x{height}+{x}+{y}')

# Function to show the main menu
def show_main_menu():
    for widget in root.winfo_children():
        widget.destroy()

    main_menu = tk.Frame(root)
    main_menu.pack()

    new_prediction_button = tk.Button(main_menu, text="Make a New Prediction", command=show_new_prediction, width=20, height=2)
    new_prediction_button.pack(pady=10)

    past_predictions_button = tk.Button(main_menu, text="Past Predictions", command=show_past_predictions, width=20, height=2)
    past_predictions_button.pack(pady=10)

    options_button = tk.Button(main_menu, text="Options", command=show_options_menu, width=20, height=2)
    options_button.pack(pady=10)

    webcam_button = tk.Button(main_menu, text="Webcam Prediction", command=start_webcam, width=20, height=2)
    webcam_button.pack(pady=10)

    quit_button = tk.Button(main_menu, text="Quit", command=root.quit, width=20, height=2)
    quit_button.pack(pady=10)

# Function to show the new prediction interface
def show_new_prediction():
    for widget in root.winfo_children():
        widget.destroy()

    prediction_frame = tk.Frame(root)
    prediction_frame.pack()

    upload_button = tk.Button(prediction_frame, text="Upload Image", command=upload_file, width=20, height=2)
    upload_button.pack(pady=20)
    global panel
    panel = tk.Label(prediction_frame)
    panel.pack(pady=20)

    back_button = tk.Button(prediction_frame, text="Back to Main Menu", command=show_main_menu, width=20, height=2)
    back_button.pack(pady=10)

# Function to show the options menu
def show_options_menu():
    for widget in root.winfo_children():
        widget.destroy()

    options_frame = tk.Frame(root)
    options_frame.pack()

    # Model selection buttons
    global selected_model
    tk.Label(options_frame, text="Select Model:", font=("Arial", 14)).pack(pady=10)
    
    one_button = tk.Button(options_frame, text="Model 1", command=lambda: set_model('one'), width=20, height=2)
    one_button.pack(pady=5)

    two_button = tk.Button(options_frame, text="Model 2", command=lambda: set_model('two'), width=20, height=2)
    two_button.pack(pady=5)

    back_button = tk.Button(options_frame, text="Back to Main Menu", command=show_main_menu, width=20, height=2)
    back_button.pack(pady=20)

# Function to set the selected model
def set_model(model_name):
    global selected_model
    selected_model = model_name

# Initialize the tkinter root window
root = tk.Tk()
root.title("Animal Classification")
center_window(root, app_resolution[0], app_resolution[1])

# Start the main menu
show_main_menu()

# Run the tkinter event loop
root.mainloop()
