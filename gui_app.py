import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Frame, Menu, Canvas, Toplevel
from PIL import Image, ImageTk
import requests
import os
import json

# URL of the Flask server
FLASK_URL = "http://127.0.0.1:5000/predict"

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

    canvas = Canvas(past_frame)
    scrollbar = Scrollbar(past_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

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
    options_window = Toplevel(root)
    options_window.title("Options")
    center_window(options_window, 300, 300)

    model_label = tk.Label(options_window, text="Select Model:")
    model_label.pack(pady=5)
    model_var = tk.StringVar(value=selected_model)
    model_one = tk.Radiobutton(options_window, text="Model One", variable=model_var, value="one")
    model_one.pack(anchor='w')
    model_two = tk.Radiobutton(options_window, text="Model Two", variable=model_var, value="two")
    model_two.pack(anchor='w')
    model_three = tk.Radiobutton(options_window, text="Model Three", variable=model_var, value="three")
    model_three.pack(anchor='w')

    def save_options():
        global selected_model
        selected_model = model_var.get()
        options_window.destroy()

    save_button = tk.Button(options_window, text="Save", command=save_options, width=20, height=2)
    save_button.pack(pady=10)

    quit_button = tk.Button(options_window, text="Quit", command=options_window.destroy, width=20, height=2)
    quit_button.pack(pady=10)

# Create the main application window
root = tk.Tk()
root.title("Animal Classifier")

# Set the size of the window and center it
center_window(root, app_resolution[0], app_resolution[1])

# Load past predictions from the file
load_predictions()

# Add a menu bar
menubar = Menu(root)
options_menu = Menu(menubar, tearoff=0)
options_menu.add_command(label="Options", command=show_options_menu)
menubar.add_cascade(label="Menu", menu=options_menu)
root.config(menu=menubar)

# Show the main menu
show_main_menu()

# Start the Tkinter event loop
root.mainloop()
