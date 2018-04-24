import tkinter as tk
from tkinter import *
from PIL import Image
import io
# from ann import run_ann
from predict_using_trained_model import return_prediction

# Main Window settings.
window = tk.Tk()
window.title("Optical Character Recognition")
window.geometry("640x480")
window.resizable(width=False, height=False)


# Drawing inside the canvas.
lastx, lasty = 0, 0

def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    global lastx, lasty
    canvas1.create_line((lastx, lasty, event.x, event.y), fill="black", width=20)
    lastx, lasty = event.x, event.y

# To clear the canvas after drawing by clicking clear button
def btn_clear():
    canvas1.delete("all")

# Save the image into .jpg extension to pass it to OpenCV
def btn_submit():
    canvas1.update()
    ps = canvas1.postscript(colormode="gray")
    im = Image.open(io.BytesIO(ps.encode('utf-8')))
    im.save("image" + '.jpg')

    print("Image Saved............")
    print("Running prediction.....")

    digit = return_prediction()[0]
    label4.configure(text=digit)


# Left frame and it's widgets.
left_frame = tk.Frame(window, padx=30)
left_frame.grid(row=0, column=0)

label1 = tk.Label(left_frame, text="Press and Draw the digit")
label1.grid(row=0, columnspan=2)

canvas1 = tk.Canvas(left_frame, relief='raised', borderwidth=1, bg="white", height=400)
canvas1.grid(row=1, columnspan=2)
canvas1.bind("<Button-1>", xy)
canvas1.bind("<B1-Motion>", addLine)

btnClear = tk.Button(left_frame, text="CLEAR", command=btn_clear)
btnClear.grid(row=2, column=0, sticky='e')

btnSubmit = tk.Button(left_frame, text="SUBMIT", command=btn_submit)
btnSubmit.grid(row=2, column=1, sticky='w')

# Right Frame and it's widgets.
right_frame = tk.Frame(window, padx=30)
right_frame.grid(row=0, column=1, sticky='n')

label2 = tk.Label(right_frame, text="Prediction")
label2.grid(row=0, columnspan=2)

label3 = tk.Label(right_frame, text="The digit is: ", pady=20)
label3.grid(row=1, columnspan=2)

label4 = tk.Label(right_frame, height=12, width=20)
label4.grid(row=3, column=1)

# label4 = tk.Label(right_frame, text="Confidence: " + str(confidence))
# label4.grid(row=4, columnspan=2)

window.mainloop()
