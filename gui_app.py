import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image, ImageTk

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with open(file_path, 'rb') as f:
                response = requests.post('http://127.0.0.1:5000/predict', files={'file': f})
                response.raise_for_status()
                predictions = response.json()
                if 'error' in predictions:
                    raise ValueError(predictions['error'])
                
                result_text.delete('1.0', tk.END)
                for pred in predictions:
                    result_text.insert(tk.END, f"{pred['label']} ({pred['score']:.2f})\n")
    
            # Display the image
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img)
            img_label.image = img
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {e}")
        except ValueError as e:
            messagebox.showerror("Error", f"Server error: {e}")

app = tk.Tk()
app.title('Animal Identifier')

upload_button = tk.Button(app, text='Upload Image', command=upload_file)
upload_button.pack(pady=10)

img_label = tk.Label(app)
img_label.pack(pady=10)

result_text = tk.Text(app, height=10, width=50)
result_text.pack(pady=10)

app.mainloop()