import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import time
import pyttsx3
import joblib
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Initialize TTS Engine
tts_engine = pyttsx3.init()

# Global variables
sentence = ""
capture_flag = False
cap = None
selected_language = "Word Gestures"
model = None
LABELS = []
is_capturing = False
confidence_threshold = 0.7  # Confidence threshold for predictions

# Colors and theme settings
BACKGROUND_COLOR = "#F0F2F5"
PRIMARY_COLOR = "#4267B2"
SECONDARY_COLOR = "#E4E6EB"
ACCENT_COLOR = "#1877F2"
TEXT_COLOR = "#050505"
BUTTON_TEXT_COLOR = "#FFFFFF"

# Load default ASL model
def load_selected_model(lang):
    global model, LABELS
    try:
        if lang == "ASL":
            model = load_model("model/asl_cnn_model.h5")
            LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        elif lang == "ISL":
            model = load_model("model/isl_cnn_model.h5")
            LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        elif lang == "BSL (One Hand)":
            model = joblib.load("model/bsl_one_hand_model.pkl")
            LABELS = model.classes_.tolist()
        elif lang == "BSL (Two Hands)":
            model = joblib.load("model/bsl_two_hand_model.pkl")
            LABELS = model.classes_.tolist()
        elif lang == "Word Gestures":
            model = load_model("model/word_gesture_model.h5")
            LABELS = ['Cool', 'Good Bye', 'Good Luck', 'Got it', 'Hi', 'I Love You', 'Love', 'Okay', 'Perfect', 'Salute', 'Stop', 'Strong', 'Thumbs up', 'Victory', 'Wait', 'Wait a minute', 'Walk']
        status_label.config(text=f"{lang} model loaded successfully", fg="green")
    except Exception as e:
        status_label.config(text=f"Error loading model: {str(e)}", fg="red")
        print(f"Error loading model: {str(e)}")

# Gesture recognition function
def gesture_recognition(frame):
    global sentence
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    hand_detected = False
    
    if result.multi_hand_landmarks:
        hand_detected = True
        landmarks = []
        x_list = []
        y_list = []

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
            
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                x_list.append(x)
                y_list.append(y)
                landmarks.extend([lm.x, lm.y, lm.z])

        if x_list and y_list:
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            
            # Add padding to the rectangle
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (74, 195, 139), 2)


        label_to_number = {
            'One - 1': '1',
            'Two - 2': '2',
            'Three - 3': '3',
            'Four - 4': '4',
            'Five - 5': '5',
            'Six - 6': '6',
            'Seven - 7': '7',
            'Eight - 8': '8',
            'Nine - 9': '9',
            'Ten - 10': '10',
            'Zero - 0': '0',
            'A - a': 'A',
            'B - b': 'B',
            'C - c': 'C',
            'D - d': 'D',
            'E - e': 'E',
            'F - f': 'F',
            'G - g': 'G',
            'H - h': 'H',
            'I - i': 'I',
            'J - j': 'J',
            'K - k': 'K',
            'L - l': 'L',
            'M - m': 'M',
            'N - n': 'N',
            'O - o': 'O',
            'P - p': 'P',
            'Q - q': 'Q',
            'R - r': 'R',
            'S - s': 'S',
            'T - t': 'T',
            'U - u': 'U',
            'V - v': 'V',
            'W - w': 'W',
            'X - x': 'X',
            'Y - y': 'Y',
            'Z - z': 'Z',
            'space': ' ',
            'nothing': ''
        }

        n_points = len(landmarks)
        prediction_text = ""
        
        if selected_language.startswith("BSL"):
            expected = 63 if selected_language == "BSL (One Hand)" else 126
            if n_points == expected:
                predicted = model.predict([landmarks])[0]
                if auto_append_var.get():
                    mapped = label_to_number.get(predicted, predicted)
                    sentence += mapped
                prediction_text = f"Detected: {predicted}"
        elif selected_language == "Word Gestures":
            if n_points == 63:  # One hand detected
                prediction = model.predict(np.array([landmarks]))[0]
                predicted_index = np.argmax(prediction)
                confidence = prediction[predicted_index]
                
                if confidence >= confidence_threshold:
                    if 0 <= predicted_index < len(LABELS):
                        predicted_label = LABELS[predicted_index]
                        if auto_append_var.get():
                            sentence += predicted_label + " "
                        prediction_text = f"Detected: {predicted_label} ({confidence:.2f})"
                    else:
                        prediction_text = f"Invalid prediction index: {predicted_index}"
                else:
                    prediction_text = f"Low confidence: {confidence:.2f}"
        else:
            if n_points == 63:
                prediction = model.predict(np.array([landmarks]))[0]
                predicted_index = np.argmax(prediction)
                confidence = prediction[predicted_index]
                
                if confidence >= confidence_threshold:
                    if 0 <= predicted_index < len(LABELS):
                        predicted_label = LABELS[predicted_index]
                        mapped = label_to_number.get(predicted_label, predicted_label)
                        if auto_append_var.get():
                            sentence += mapped
                        prediction_text = f"Detected: {predicted_label} ({confidence:.2f})"
                    else:
                        prediction_text = f"Invalid prediction index: {predicted_index}"
                else:
                    prediction_text = f"Low confidence: {confidence:.2f}"
    
    if hand_detected:
        detection_status.config(text="✓ Hand Detected", fg="green")
        if prediction_text:
            prediction_status.config(text=prediction_text)
    else:
        detection_status.config(text="✗ No Hand Detected", fg="red")
        prediction_status.config(text="")
    
    return frame

def toggle_camera():
    global cap, is_capturing
    
    if is_capturing:
        is_capturing = False
        if cap is not None:
            cap.release()
            cap = None
        camera_btn.config(text="Start Camera", bg="#4CAF50")
        video_label.config(image="")
        video_label.image = None
        detection_status.config(text="Camera Off")
        prediction_status.config(text="")
    else:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            is_capturing = True
            camera_btn.config(text="Stop Camera", bg="#F44336")
            capture_video()
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")

last_capture_time = time.time()

def capture_video():
    global sentence, capture_flag, last_capture_time, is_capturing

    if not is_capturing or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Camera disconnected. Trying to reconnect...", fg="orange")
        time.sleep(1)
        if is_capturing:
            root.after(100, capture_video)
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    if current_time - last_capture_time >= 1 and auto_capture_var.get():
        processed_frame = gesture_recognition(frame.copy())
        last_capture_time = current_time
        capture_flag = True
        frame = processed_frame
    elif not auto_capture_var.get():
        processed_frame = gesture_recognition(frame.copy())
        frame = processed_frame

    if capture_flag:
        cv2.putText(frame, "Capturing...", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        capture_flag = False

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.image = imgtk

    if is_capturing:
        root.after(30, capture_video)

def manual_capture():
    global sentence, last_capture_time
    
    if not is_capturing or cap is None:
        messagebox.showinfo("Info", "Please start the camera first")
        return
        
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        gesture_recognition(frame)
        last_capture_time = time.time()
        
        status_label.config(text="Sign captured", fg="green")
        root.after(1000, lambda: status_label.config(text="Ready", fg="black"))

# Update text area
def update_text():
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, sentence)
    char_count.config(text=f"Characters: {len(sentence)}")
    root.after(500, update_text)

# Button Functions
def clear_text():
    global sentence
    sentence = ""
    status_label.config(text="Text cleared", fg="black")

def save_text():
    global sentence
    if sentence.strip() != "":
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"sign_language_{current_time}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(sentence)
                status_label.config(text=f"Saved to {os.path.basename(file_path)}", fg="green")
            except Exception as e:
                status_label.config(text=f"Error saving file: {str(e)}", fg="red")
    else:
        messagebox.showinfo("Info", "Nothing to save")

def speak_text():
    global sentence
    if sentence.strip() != "":
        status_label.config(text="Speaking...", fg="blue")
        
        # Run TTS in a separate thread to prevent UI freezing
        def tts_thread():
            tts_engine.say(sentence)
            tts_engine.runAndWait()
            root.after(0, lambda: status_label.config(text="Done speaking", fg="black"))
            
        Thread(target=tts_thread).start()
    else:
        messagebox.showinfo("Info", "Nothing to speak")

def add_space():
    global sentence
    sentence += " "
    status_label.config(text="Space added", fg="black")

def add_character(char):
    global sentence
    sentence += char
    status_label.config(text=f"Added '{char}'", fg="black")

def quit_app():
    if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
        if cap is not None:
            cap.release()
        root.destroy()

# On language change
def language_changed(event):
    global selected_language, sentence
    selected_language = language_var.get()
    status_label.config(text=f"Changing to {selected_language}...", fg="blue")
    load_selected_model(selected_language)

# Update threshold
def update_threshold(value):
    global confidence_threshold
    confidence_threshold = float(value)
    status_label.config(text=f"Confidence Threshold set to {confidence_threshold:.2f}", fg="black")


# Setup custom title bar
def create_title_bar(root):
    title_bar = tk.Frame(root, bg=PRIMARY_COLOR, height=35)
    title_bar.pack(fill=tk.X)
    
    # Title
    title_label = tk.Label(title_bar, text="Multilingual Sign Language Recognition", 
                          font=("Segoe UI", 12, "bold"), bg=PRIMARY_COLOR, fg="white")
    title_label.pack(side=tk.LEFT, padx=10)
    
    # Close button
    close_button = tk.Button(title_bar, text="×", bg=PRIMARY_COLOR, fg="white", 
                           font=("Arial", 16), bd=0, padx=10, 
                           activebackground="#D32F2F", activeforeground="white",
                           command=quit_app)
    close_button.pack(side=tk.RIGHT)

# Create custom stylish button
def create_styled_button(parent, text, command, width=15, height=2, bg=ACCENT_COLOR, 
                        fg=BUTTON_TEXT_COLOR, font=("Segoe UI", 10)):
    btn = tk.Button(parent, text=text, command=command, width=width, height=height,
                  bg=bg, fg=fg, font=font, bd=0, relief=tk.RAISED,
                  activebackground=PRIMARY_COLOR, activeforeground=BUTTON_TEXT_COLOR,
                  cursor="hand2")
    return btn

# GUI Setup
root = tk.Tk()
root.title("Multilingual Sign Language Recognition")
root.geometry("1000x800")
root.configure(bg=BACKGROUND_COLOR)

# Configure ttk style
style = ttk.Style()
style.theme_use('clam')
style.configure('TCombobox', 
               fieldbackground=SECONDARY_COLOR,
               background=ACCENT_COLOR,
               foreground=TEXT_COLOR,
               arrowcolor=PRIMARY_COLOR)
style.configure('TNotebook',
               background=BACKGROUND_COLOR)
style.configure('TNotebook.Tab',
               background=SECONDARY_COLOR,
               foreground=TEXT_COLOR,
               padding=[10, 5],
               font=('Segoe UI', 10))
style.map('TNotebook.Tab',
         background=[('selected', ACCENT_COLOR)],
         foreground=[('selected', BUTTON_TEXT_COLOR)])

# Main frame
main_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header with app name and logo
header_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
header_frame.pack(fill=tk.X, pady=10)

app_label = tk.Label(header_frame, text="Sign Language Translator", 
                    bg=BACKGROUND_COLOR, fg=PRIMARY_COLOR, 
                    font=("Segoe UI", 24, "bold"))
app_label.pack(pady=10)

# Status indicators
status_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
status_frame.pack(fill=tk.X, pady=5)

status_label = tk.Label(status_frame, text="Ready", bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                      font=("Segoe UI", 10), anchor="w")
status_label.pack(side=tk.LEFT, padx=5)

# MODIFICATION: Move camera button to the right side of status frame
camera_btn = tk.Button(status_frame, text="Start Camera", font=("Segoe UI", 12),
                     bg="#4CAF50", fg="white", bd=0, padx=10, pady=5,
                     command=toggle_camera)
camera_btn.pack(side=tk.RIGHT, padx=5)

detection_status = tk.Label(status_frame, text="Camera Off", bg=BACKGROUND_COLOR, 
                          fg="gray", font=("Segoe UI", 10), anchor="e")
detection_status.pack(side=tk.RIGHT, padx=5)

prediction_status = tk.Label(status_frame, text="", bg=BACKGROUND_COLOR,
                           fg=ACCENT_COLOR, font=("Segoe UI", 10, "bold"), anchor="e")
prediction_status.pack(side=tk.RIGHT, padx=15)

# Options frame (Language and Settings)
options_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
options_frame.pack(fill=tk.X, pady=10)

# Language Selector
language_label = tk.Label(options_frame, text="Select Language:", 
                        bg=BACKGROUND_COLOR, fg=TEXT_COLOR, 
                        font=("Segoe UI", 12))
language_label.pack(side=tk.LEFT, padx=5)

language_var = tk.StringVar()
language_selector = ttk.Combobox(options_frame, textvariable=language_var, 
                               state="readonly", font=("Segoe UI", 12), width=15)
language_selector['values'] = ["Word Gestures", "ASL", "BSL (One Hand)", "BSL (Two Hands)", "ISL"]
language_selector.current(0)
language_selector.bind("<<ComboboxSelected>>", language_changed)
language_selector.pack(side=tk.LEFT, padx=10)

# Auto-append toggle
auto_append_var = tk.BooleanVar(value=True)
auto_append_check = tk.Checkbutton(options_frame, text="Auto-append detected signs", 
                                 variable=auto_append_var, bg=BACKGROUND_COLOR, 
                                 fg=TEXT_COLOR, font=("Segoe UI", 12),
                                 selectcolor=SECONDARY_COLOR, 
                                 activebackground=BACKGROUND_COLOR)
auto_append_check.pack(side=tk.LEFT, padx=20)

# Auto-capture toggle
auto_capture_var = tk.BooleanVar(value=True)
auto_capture_check = tk.Checkbutton(options_frame, text="Auto-capture", 
                                  variable=auto_capture_var, bg=BACKGROUND_COLOR, 
                                  fg=TEXT_COLOR, font=("Segoe UI", 12),
                                  selectcolor=SECONDARY_COLOR, 
                                  activebackground=BACKGROUND_COLOR)
auto_capture_check.pack(side=tk.LEFT, padx=20)

# Video and controls container
content_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Left panel (video)
video_frame = tk.Frame(content_frame, bg="#000000", bd=2, relief=tk.RIDGE)
video_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

video_label = tk.Label(video_frame, bg="black", width=80, height=30)
video_label.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

# Camera controls
camera_controls = tk.Frame(video_frame, bg=BACKGROUND_COLOR, height=50)
camera_controls.pack(fill=tk.X, pady=5)

# MODIFICATION: Removed camera_btn from here

capture_btn = tk.Button(camera_controls, text="Capture Sign", font=("Segoe UI", 12),
                      bg=ACCENT_COLOR, fg="white", bd=0, padx=10, pady=5,
                      command=manual_capture)
capture_btn.pack(side=tk.LEFT, padx=5)

# Threshold slider
threshold_frame = tk.Frame(camera_controls, bg=BACKGROUND_COLOR)
threshold_frame.pack(side=tk.RIGHT, padx=10)

threshold_label = tk.Label(threshold_frame, text="Confidence:", 
                         bg=BACKGROUND_COLOR, fg=TEXT_COLOR, 
                         font=("Segoe UI", 10))
threshold_label.pack(side=tk.LEFT)

threshold_slider = tk.Scale(threshold_frame, from_=0.1, to=1.0, resolution=0.05,
                          orient=tk.HORIZONTAL, length=150, bd=0,
                          sliderlength=20, bg=BACKGROUND_COLOR,
                          highlightthickness=0, command=update_threshold)
threshold_slider.set(confidence_threshold)
threshold_slider.pack(side=tk.LEFT)

threshold_value_label = tk.Label(threshold_frame, text="70%", 
                               bg=BACKGROUND_COLOR, fg=TEXT_COLOR, 
                               font=("Segoe UI", 10), width=4)
threshold_value_label.pack(side=tk.LEFT)

# Right panel (text and controls)
text_panel = ttk.Notebook(content_frame)
text_panel.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

# Text tab
text_tab = tk.Frame(text_panel, bg=BACKGROUND_COLOR)
text_panel.add(text_tab, text="  Text  ")

# Text Area
text_frame = tk.Frame(text_tab, bg=BACKGROUND_COLOR)
text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

text_area = tk.Text(text_frame, height=12, width=40, font=("Segoe UI", 14),
                  bg="white", fg=TEXT_COLOR, wrap=tk.WORD, bd=1,
                  relief=tk.SOLID, padx=10, pady=10)
text_area.pack(fill=tk.BOTH, expand=True)

text_scrollbar = tk.Scrollbar(text_area)
text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=text_scrollbar.set)
text_scrollbar.config(command=text_area.yview)

char_count = tk.Label(text_frame, text="Characters: 0", bg=BACKGROUND_COLOR, 
                    fg=TEXT_COLOR, font=("Segoe UI", 10), anchor="e")
char_count.pack(side=tk.RIGHT, padx=5, pady=3)

# Common characters
common_chars_frame = tk.Frame(text_tab, bg=BACKGROUND_COLOR)
common_chars_frame.pack(fill=tk.X, pady=5)

space_btn = tk.Button(common_chars_frame, text="Space", font=("Segoe UI", 12),
                    bg=SECONDARY_COLOR, fg=TEXT_COLOR, bd=0, padx=15, pady=5,
                    command=add_space)
space_btn.pack(side=tk.LEFT, padx=5)

for char in ['.', ',', '?', '!']:
    char_btn = tk.Button(common_chars_frame, text=char, font=("Segoe UI", 12),
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, bd=0, width=3, pady=5,
                       command=lambda c=char: add_character(c))
    char_btn.pack(side=tk.LEFT, padx=3)

# Text control buttons
text_buttons_frame = tk.Frame(text_tab, bg=BACKGROUND_COLOR)
text_buttons_frame.pack(fill=tk.X, pady=10)

clear_btn = create_styled_button(text_buttons_frame, "Clear All", clear_text, 
                               width=10, bg="#FF5722")
clear_btn.pack(side=tk.LEFT, padx=5)

save_btn = create_styled_button(text_buttons_frame, "Save to File", save_text, 
                              width=10, bg="#4CAF50")
save_btn.pack(side=tk.LEFT, padx=5)

speak_btn = create_styled_button(text_buttons_frame, "Speak", speak_text, 
                               width=10, bg=ACCENT_COLOR)
speak_btn.pack(side=tk.LEFT, padx=5)

# Help tab
help_tab = tk.Frame(text_panel, bg=BACKGROUND_COLOR)
text_panel.add(help_tab, text="  Help  ")

help_text = tk.Text(help_tab, height=15, width=40, font=("Segoe UI", 12),
                  bg="white", fg=TEXT_COLOR, wrap=tk.WORD, bd=0,
                  padx=10, pady=10)
help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

help_content = """
# Sign Language Translator Help

## Getting Started
1. Select your desired sign language from the dropdown menu
2. Click "Start Camera" to begin capturing
3. Position your hand in front of the camera
4. Signs will be detected automatically if auto-capture is enabled

## Tips
- Keep your hand within frame and well-lit
- Hold each sign steady for better recognition
- Use the confidence slider to adjust detection sensitivity
- Click "Capture Sign" to manually capture when auto-capture is off
- Use the space button to add spaces between words
- Click "Speak" to hear your translated text

## Supported Languages
- ASL (American Sign Language)
- BSL (British Sign Language)
- ISL (Indian Sign Language)
"""

help_text.insert(tk.END, help_content)
help_text.config(state=tk.DISABLED)

# Footer with credits
footer_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR, height=30)
footer_frame.pack(fill=tk.X, pady=5)

credits_label = tk.Label(footer_frame, text="© 2025 Sign Language Translator", 
                       bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 8))
credits_label.pack(side=tk.LEFT, padx=5)

quit_btn = tk.Button(footer_frame, text="Quit App", font=("Segoe UI", 10),
                   bg="#F44336", fg="white", bd=0, padx=10, pady=2,
                   command=quit_app)
quit_btn.pack(side=tk.RIGHT, padx=5)

# Load default model
load_selected_model("Word Gestures")

# Start text update
update_text()

# Set protocol for window close button
root.protocol("WM_DELETE_WINDOW", quit_app)

# Start the application
root.mainloop()