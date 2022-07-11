import tkinter
import tkinter.font
import tkinter.messagebox
import requests

from tkinter import StringVar, scrolledtext
from functools import partial

def change_path(path, transformer_menu, bigram_entry, *args):
    if path.get() == "Perplexity":
        transformer_menu.config(state="disabled")
        bigram_entry.config(state="normal")
    elif path.get() == "Transformers":
        transformer_menu.config(state="active")
        bigram_entry.config(state="disabled")


def send_request(top, input, path, transformer, bigram_factor, *args):
    bigram = 0
    try:
        bigram = float(bigram_factor.get())
    except ValueError:
        tkinter.messagebox.showerror(title="Input error", message="Bigram factor is not a float value")
        render_input(top)
        return

    if bigram < 0 or bigram > 1:
        tkinter.messagebox.showerror(title="Input error", message="Bigram factor is not a value between 0 and 1")
        render_input(top)
        return

    if input.get("1.0",'end-1c') == "":
        tkinter.messagebox.showerror(title="Input error", message="No text was introduced")
        render_input(top)
        return

    data = {
        "sentence": input.get("1.0",'end-1c'),
        "path": path.get().lower(),
        "transformer": transformer.get().lower(),
        "bigram_factor": bigram
    }

    response = requests.post("http://0.0.0.0:5000/simplify", json=data)

    render_result(top, response.json())


def render_result(top, response):
    for widget in top.winfo_children():
        widget.destroy()

    bg_image = tkinter.PhotoImage(file="images/background.png")
    image_label = tkinter.Label(top, image=bg_image)
    image_label.image = bg_image
    image_label.place(x=0, y=0, relwidth=1, relheight=1)

    font = tkinter.font.Font(family="Bookman Old Style", size=20)
    font_bold = tkinter.font.Font(family="Bookman Old Style", size=20, weight="bold")

    original = scrolledtext.ScrolledText(top, width=45, height = 6, font=font, wrap = tkinter.WORD)
    original.place(x=12, y=200)
    original.tag_config("highlighted", foreground="red", font=font_bold)

    simplified = scrolledtext.ScrolledText(top, width=45, height = 6, font=font, wrap = tkinter.WORD)
    simplified.place(x=12, y=400)
    simplified.tag_config("highlighted", foreground="red", font=font_bold)

    original_index = 0
    simplified_index = 0

    for pair in response:
        original.insert(tkinter.END, pair["old"] + " ")
        simplified.insert(tkinter.END, pair["new"] + " ")

        if pair["old"] != pair["new"]:
            original.tag_add("highlighted", "1." + str(original_index), "1." + str(original_index + len(pair["old"])))
            simplified.tag_add("highlighted", "1." + str(simplified_index), "1." + str(simplified_index + len(pair["new"])))

        original_index += len(pair["old"]) + 1
        simplified_index += len(pair["new"]) + 1

    submit_button = tkinter.Button(top, bg="white", font=font_bold, justify="center", text="Back", height=2, 
                                   command=partial(render_input, top))
    submit_button.place(x=280, y=600)
    submit_button.config(width=12)


def render_input(top, *args):
    for widget in top.winfo_children():
        widget.destroy()

    font = tkinter.font.Font(family="Bookman Old Style", size=20)
    font_bold = tkinter.font.Font(family="Bookman Old Style", size=20, weight="bold")

    bg_image = tkinter.PhotoImage(file="images/background.png")
    image_label = tkinter.Label(top, image=bg_image)
    image_label.image = bg_image
    image_label.place(x=0, y=0, relwidth=1, relheight=1)

    input = scrolledtext.ScrolledText(top, width=45, height = 10, font=font, wrap = tkinter.WORD)
    input.place(x=12, y=200)

    path = tkinter.StringVar(top)
    path.set("Perplexity")

    transformer = tkinter.StringVar(top)
    transformer.set("BERT")

    transformer_menu = tkinter.OptionMenu(top, transformer, "BERT", "RoBERTa", "GPT2")
    transformer_menu.config(width=12, bg="white", font=font, justify="center", state="disabled")
    transformer_menu.place(x=25 + 270, y=500)

    bigram_factor = StringVar(top)
    bigram_factor.set("0.00")

    bigram_entry = tkinter.Entry(top, textvariable=bigram_factor)
    bigram_entry.config(width=12, bg="white", font=font, justify="center")
    bigram_entry.place(x=25 + 2 * 270, y=503)

    path_menu = tkinter.OptionMenu(top, path, "Perplexity", "Transformers", command=partial(change_path ,path, transformer_menu, bigram_entry))
    path_menu.config(width=12, bg="white", font=font, justify="center")
    path_menu.place(x=25, y=500)

    submit_button = tkinter.Button(top, bg="white", font=font_bold, justify="center", text="Simplify", height=2, 
                                   command=partial(send_request, top, input, path, transformer, bigram_factor))
    submit_button.place(x=280, y=600)
    submit_button.config(width=12)


if __name__ == "__main__":
    top = tkinter.Tk()
    top.title("SimpLex simplification system")
    top.geometry("800x800")

    top.resizable(False, False)

    render_input(top)

    top.mainloop()
