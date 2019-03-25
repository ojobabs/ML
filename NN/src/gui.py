# Project: Using Neural Networks to Recognize Handwritten Digits
# Date: March 8th, 2019
# Author: Oscar Alberto Carreno Gutierrez

# This program is to be used as a front-end for the Neural-Network
# programs for Digit Recognition made for this project delivery as
# a way to interact with the Shell without the use of commands.

# Libraries Utilized for the software
import subprocess as sp     # Required for redirecting stdout to GUI
import sys, inspect         # Required for all methods and functions redirection

from Tkinter import Tk, Text, Label, BOTH, W, N, E, S, END, INSERT, HORIZONTAL, VERTICAL, NONE, StringVar
from ttk import Frame, Button, Style, Scrollbar, Checkbutton
from os.path import join, dirname
from datetime import datetime

mainpath = join(dirname(__file__), "main.py")
pythonpath = "python"

class FrameBuilder(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)        # Initializes the frame
        self.parent = parent                # Assigns the root variable to parent
        self.initUI()                       # Initializes the GUI configuration

    def initUI(self):

        # Configures the GUI
        self.parent.title("Neural Networks for Handwritten Digit Recognition")
        self.style = Style()
        self.pack(fill = BOTH, expand = 1)

        # Creates a grid 5x4 in to which we will place elements.
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 0)
        self.columnconfigure(3, weight = 0)
        self.columnconfigure(4, weight = 0)
        self.columnconfigure(5, weight = 0)
        self.columnconfigure(6, weight = 0)
        self.columnconfigure(7, weight = 0)
        self.columnconfigure(8, weight = 0)
        self.rowconfigure(1, weight = 1)
        self.rowconfigure(2, weight = 0)
        self.rowconfigure(3, weight = 0)
        self.rowconfigure(4, weight = 0)

        # Create explanation text for app
        explanation = """
        "This program is capable of utilizing different learning algorithms as a mean to achieve a +90% accuracy in recognizing MNIST
    	digits based on Neural Networks as a way of detecting the different numbers and classifying them correctly" \n
        Select the checkboxes that you wish to utilize ([ ] Learn from starting random values, [X] Learn from preexisting data / [ ] Do not
        save the results [X] Save the final values of the Weights and Biases), Press the Run button to start the Neural Network\n"""

        self.exp = Label(self, text = explanation)
        self.exp.grid(row = 2,
                      column = 1,
                      columnspan = 4,
                      padx = 2,
                      pady = 2,)

        # Creates the Y scrollbar that is spawned through the whole grid
        yscrollbar = Scrollbar(self, orient = VERTICAL)
        yscrollbar.grid(row = 1,
                        column = 8,
                        sticky = N + S)

        # Creates the text area and spawns it through the whole window
        self.textarea = Text(self,
                             wrap = NONE,
                             bd = 0,
                             yscrollcommand = yscrollbar.set)

        self.textarea.grid(row = 1,
                           column = 1,
                           columnspan = 4,
                           rowspan = 1,
                           padx = 0,
                           sticky = E + W + S + N)

        yscrollbar.config(command = self.textarea.yview)

        # Creates the run button in the lower row
        self.runButton = Button(self, text = "Run Neural Network")
        self.runButton.grid(row = 4,
                            column = 1,
                            padx = 5,
                            pady = 5,
                            sticky = W)
        self.runButton.bind("<ButtonRelease-1>", self.runNN)

        # Creates the variable initialization checkbox in the lower row
        self.initvalVar = StringVar()
        initvalCheck = Checkbutton(self, text = "Learn from Initial Values",
                                   variable = self.initvalVar,
                                   onvalue = "-l",
                                   offvalue = "")
        initvalCheck.grid(row = 4,
                          column = 2,
                          padx = 5,
                          pady = 5)

        # Creates the save variables checkbox in the lower row
        self.saveValVar = StringVar()
        saveVal = Checkbutton(self, text = "Save final Weights & Biases",
                              variable = self.saveValVar,
                              onvalue = "-s",
                              offvalue = "")
        saveVal.grid(row = 4,
                     column = 3,
                     padx = 5,
                     pady = 5)

        # Creates the clear button for the textbox
        self.clearButton = Button(self, text = "Clear")
        self.clearButton.grid(row = 4,
                              column = 4,
                              padx = 5,
                              pady = 5)
        self.clearButton.bind("<ButtonRelease-1>", self.clearText)

        # Defines the tags that are used to colorise the text added to the text widget.
        self.textarea.tag_config("errorstring", foreground = "#CC0000") # Red Color
        self.textarea.tag_config("infostring", foreground = "#008800")  # Green Color

    def tagsForLine(self, line):
        # Returns a tuple of tags to be applied to the line of text when being added
        l = line.lower()
        if "error" in l or "traceback" in l:
            return ("errorstring", )
        return ()

    def addText(self, str, tags=None):
        # Add a line of text to the textWidget.
        self.textarea.insert(INSERT, str, tags or self.tagsForLine(str))
        self.textarea.yview(END)

    def clearText(self, event):
        # Clears all the text from the textbox
        self.textarea.delete("1.0", END)

    def moveCursorToEnd(self):
        # Moves the cursor to the end of the text widget's text
        self.textarea.mark_set("insert", END)

    def runNN(self, event):
        self.moveCursorToEnd()
        self.addText("The Neural Network has Started %s\n" % (str(datetime.now())), ("infostring", ))
        self.addText("*" * 80 + "\n", ("infostring", ))

        cmdlist = filter(lambda x: x if x else None,
            [pythonpath, mainpath, self.initvalVar.get(), self.saveValVar.get()])

        self.addText(" ".join(cmdlist) + "\n", ("infostring", ))

        proc = sp.Popen(cmdlist,
                        stdout = sp.PIPE,
                        stderr = sp.STDOUT,
                        universal_newlines = True)

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            self.addText(line)
            self.textarea.update_idletasks()

        self.addText("Neural Network Finished %s \n" % (str(datetime.now())), ("infostring", ))
        self.addText("*" * 80 + "\n", ("infostring", ))


def main():
    # Main Root Widget
    root = Tk()
    # Main Window Settings
    root.geometry("1000x600+300+300")
    # Frame Builder
    FrameBuilder(root)
    # Main Root Loop
    root.mainloop()

if __name__ == '__main__':
    # Runs the GUI
    main()
