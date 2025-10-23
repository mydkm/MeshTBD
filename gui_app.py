import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("My First GUI")

# Add a label
label = tk.Label(root, text="Mesh Automation!", font=("Arial", 16))
label.pack(padx=20, pady=20)

# Add a button to close the window
button = tk.Button(root, text="Close", command=root.destroy)
button.pack(pady=10)

# Start the event loop (keeps window open)
root.mainloop()