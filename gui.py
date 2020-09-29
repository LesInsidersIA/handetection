#! /usr/bin/python

import tkinter as tk
from PIL import Image, ImageTk

import multiprocessing

class App:
    def __init__(self, master=tk.Tk()):
        self.master = master
        self.fig_size = [449, 570]
        self.frame = tk.Frame(master)
        self.canvas = tk.Canvas(self.frame, width=449, height=570)
        self.canvas.pack()

        self.load_image('images/01.jpg')
        self.image_label = tk.Label(self.canvas, image=self.fig_image)
        self.image_label.pack()

        self.button_left = tk.Button(self.frame, text="EDGED DETECT", command=self.update)
        self.button_left.pack(side="left")

        self.button_right = tk.Button(self.frame, text="SHOW RESULT", command=self.result)
        self.button_right.pack(side="right")

        self.button_top = tk.Button(self.frame, text="HSV THRESHOLDING", command=self.hsv_transform)
        self.button_top.pack(side="top")

        self.frame.bind("q", self.close)
        self.frame.bind("<Escape>", self.close)
        self.frame.pack()
        self.frame.focus_set()

        self.is_active = True

    def load_image(self, filename):
        self.fig_image = ImageTk.PhotoImage(Image.open(filename).resize(self.fig_size, Image.BILINEAR))

    def update(self, *args):
        self.load_image('images/edged_img.jpg')
        self.image_label.config(image=self.fig_image)

    def result(self, *args):
        self.load_image('images/result_img.jpg')
        self.image_label.config(image=self.fig_image)

    def hsv_transform(self, *args):
        self.load_image('images/hsv_img.jpg')
        self.image_label.config(image=self.fig_image)
    
    def close(self, *args):
        print('GUI Closed !')
        self.master.quit()
        self.is_active = False

    def is_closed(self):
        return not self.is_active

    def mainloop(self):
        self.master.mainloop()
        print('mainloop closed...')

if __name__ == '__main__':
    import time
    app = App()
    app.mainloop()