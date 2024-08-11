import joblib
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk, ImageOps
import customtkinter
import tkinter as tk

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')

# Initialize customtkinter
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class DigitRecognizerAppRF(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Handwritten Digit Recognition with Random Forest")
        self.geometry("700x700")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Digit Recognizer RF",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Predict", command=self.predict)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Clear", command=self.clear_canvas)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        # Create main canvas for drawing
        self.canvas_width = 600
        self.canvas_height = 600
        self.canvas = customtkinter.CTkCanvas(self, bg='white', width=self.canvas_width, height=self.canvas_height, borderwidth=2, relief='sunken')
        self.canvas.grid(row=0, column=1, columnspan=3, padx=20, pady=20, sticky="nsew")

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create label for prediction result
        self.result_label = customtkinter.CTkLabel(self, text="Draw digits and click Predict", font=('Arial', 16))
        self.result_label.grid(row=1, column=1, columnspan=3, padx=20, pady=10)

        # Initialize the image object for drawing
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), color='white')
        self.draw = ImageDraw.Draw(self.image)

        # Create an image to display bounding boxes
        self.bounding_boxes_image = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')

        # Initial theme state
        self.is_dark_mode = False

    def paint(self, event):
        # Brush size
        brush_size = 20

        # Get the x and y coordinates of the mouse event
        x, y = event.x, event.y

        # Determine brush color based on theme
        brush_color = 'white' if self.is_dark_mode else 'black'

        # Draw an ellipse at the mouse position on the canvas
        self.canvas.create_oval(x - brush_size // 2, y - brush_size // 2,
                                x + brush_size // 2, y + brush_size // 2,
                                fill=brush_color, outline=brush_color)

        # Draw on the PIL image to match the canvas
        self.draw.ellipse([x - brush_size // 2, y - brush_size // 2,
                           x + brush_size // 2, y + brush_size // 2],
                          fill=brush_color)

    def predict(self):
        # Convert canvas to OpenCV image
        img_cv = np.array(self.image)

        # Convert image to grayscale and invert colors
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare to draw bounding boxes and predictions
        bounding_boxes_img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(bounding_boxes_img)

        # List to hold predictions and bounding boxes
        predictions = []
        prediction_texts = []

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = max(x, 0), max(y, 0), min(x + w, self.canvas_width), min(y + h,
                                                                                  self.canvas_height)  # Adjust bounds

            # Extract digit image
            digit_img = img_cv[y:h, x:w]
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            digit_img = Image.fromarray(digit_img)
            digit_img = ImageOps.invert(digit_img)

            # Resize to model input size while preserving aspect ratio
            digit_img_resized = digit_img.resize((28, 28), Image.LANCZOS)
            digit_img_resized = np.array(digit_img_resized)
            digit_img_resized = digit_img_resized.astype('float32') / 255
            digit_img_resized = digit_img_resized.reshape(1, -1)  # Flatten for Random Forest

            # Predict
            digit_img_resized = scaler.transform(digit_img_resized)  # Apply scaler transformation
            pred = rf_model.predict(digit_img_resized)
            predictions.append((x, y, w, h, pred[0]))

        # Draw bounding boxes and predictions on the bounding boxes image
        for x, y, w, h, pred in predictions:
            predicted_class = int(pred)
            prediction_texts.append(str(predicted_class))
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
            draw.text((x, y - 30), str(predicted_class), fill='red')

        # Update result label with predictions
        if prediction_texts:
            self.result_label.configure(text=f"Predicted digits: {', '.join(prediction_texts)}")
        else:
            self.result_label.configure(text="No digits detected.")

        # Convert bounding boxes image to Tkinter PhotoImage
        bounding_boxes_img_tk = ImageTk.PhotoImage(bounding_boxes_img)

        # Display the updated image with bounding boxes
        self.canvas.create_image(0, 0, anchor=tk.NW, image=bounding_boxes_img_tk, tags="bounding_boxes")
        self.canvas.image = bounding_boxes_img_tk

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), color='white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.configure(text="Draw digits and click Predict")

    def toggle_theme(self):
        if self.is_dark_mode:
            customtkinter.set_appearance_mode("light")
            self.is_dark_mode = False
        else:
            customtkinter.set_appearance_mode("dark")
            self.is_dark_mode = True


if __name__ == "__main__":
    app = DigitRecognizerAppRF()
    app.mainloop()
