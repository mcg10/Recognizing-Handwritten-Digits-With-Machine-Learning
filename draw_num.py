import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import messagebox
from scipy import misc, spatial
import PIL
from PIL import ImageDraw, Image
import numpy as np

#enables painting and helps make sure that paint binds to canvas
def enable_paint(e):
    global old_x, old_y
    c.bind('<B1-Motion>', paint)
    old_x, old_y = e.x, e.y

#paint function that updates coordinates a user paints in different places
def paint(e):
    global old_x, old_y
    new_x, new_y = e.x, e.y
    c.create_line((old_x, old_y, new_x, new_y), capstyle=ROUND, width=10)
    draw.line((old_x, old_y, new_x, new_y), fill='black', width=10)
    old_x, old_y = new_x, new_y

#save image, read it, reshape it, and have the model make a prediction
def makeGuess():
    image.save("num.png")
    img = misc.imread("num.png", mode = "L")
    img = np.invert(img)
    img = misc.imresize(img,(28,28))
    img = img.reshape(1,28,28,1)
    ans = str(model.predict(img)[0].tolist().index(1))
    messagebox.showinfo("Prediction", "This is a " + ans)   
   
    
#helper function to clear the canvas
def clear():
    c.delete(ALL)
    draw.rectangle((0, 0, 500, 500), fill="white")

    
#create Tkinter environment, import CNN, and establish some helper variables
root = Tk()
model = load_model("mnist_cnn.h5")
width = 500
height = 500
c = Canvas(root, width=width, height=height, bg='white')
old_x, old_y = None, None



#create a saveable image
image = PIL.Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

#make sure that the paint binds to the canvas
c.bind('<1>', enable_paint)
c.pack(expand=YES, fill=BOTH)

#create a button that actuates the predicition the function
predict_button = Button(text="Predict", command= makeGuess)
predict_button.pack()

#create a button that clears the canvas
clear_button = Button(text = "Clear Canvas", command = clear)
clear_button.pack()

root.mainloop()
