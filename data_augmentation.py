import numpy as np
import cv2 as cv
import os 
import sys
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich.status import Status
import time
import questionary

console = Console()

def bright(imgs, labels, current_dir, bright_amount):
    '''Aumentar o brilho'''

    bright_amount = int(bright_amount)

    for name, path in imgs:
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img)

        v[v > 255 - bright_amount] = 255
        v[v <= 255 - bright_amount] += bright_amount

        bright = cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR)
        img_path = os.path.join(current_dir, name + '_bright.png')
        cv.imwrite(img_path, bright)

    for name, path in labels:
        with open(path, 'r', encoding='utf-8') as f:
            label_data = f.read()
        label_path = os.path.join(current_dir, name + '_bright.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_data)

def blur(imgs, labels, current_dir, kernell):
    '''Adiciona desfoque'''
    kernell = int(kernell)
    kernell_original = kernell
    for name, path in imgs:
        img = cv.imread(path)
        x, y, _ = img.shape
        kernell = int(kernell*(x+y)/2/640)
        if kernell % 2 == 0:
            kernell += 1
        kernell = (kernell, kernell)
        blurred = cv.GaussianBlur(img, kernell, 0)
        img_path = os.path.join(current_dir, name + '_blurred.png')
        cv.imwrite(img_path, blurred)
        kernell = kernell_original

    for name, path in labels:
        with open(path, 'r', encoding='utf-8') as f:
            label_data = f.read()
        label_path = os.path.join(current_dir, name + '_blurred.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_data)

def noise(imgs, labels, current_dir, noise_amount):
    '''Aplicar ruido gaussiano'''

    noise_amount = float(noise_amount)

    for name, path in imgs:
        img = cv.imread(path)
        x, y, _ = img.shape
        noise = img.astype(np.float32) + np.random.normal(0, noise_amount*(x+y)/2/640, img.shape).astype(np.float32)
        img_path = os.path.join(current_dir, name + '_noise.png')
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        cv.imwrite(img_path, noise)

    for name, path in labels:
        with open(path, 'r', encoding='utf-8') as f:
            label_data = f.read()
        label_path = os.path.join(current_dir, name + '_noise.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_data)

def flip(imgs, labels, current_dir):
    '''Girar 180°'''
    
    for name, path in imgs:
        img = cv.imread(path)
        rotated = cv.rotate(img, cv.ROTATE_180)
        img_path = os.path.join(current_dir, name + '_rotated.png')
        cv.imwrite(img_path, rotated)

    for name, path in labels:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        rotated_lines = []
        for line in lines:
            cls, x, y, w, h = line.strip(). split()
            x_new = 1.0 - float(x)
            y_new = 1.0 - float(y)

            rotated_line = f"{cls} {x_new:.6f} {y_new:.6f} {w} {h}\n"
            rotated_lines.append(rotated_line)

        label_path = os.path.join(current_dir, name + '_rotated.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.writelines(rotated_lines)

def gray_scale(imgs, labels, current_dir):
    '''Colocar em escala de cinza'''
    
    for name, path in imgs:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_path = os.path.join(current_dir, name + '_gray.png')
        cv.imwrite(img_path, gray)

    for name, path in labels:
        with open(path, 'r', encoding='utf-8') as f:
            label_data = f.read()
        label_path = os.path.join(current_dir, name + '_gray.txt')
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_data)

def UI(imgs, labels, imgs_and_labels):
    console.clear()

    option = questionary.checkbox(
        "Which kind of augmentation do you want?",
        choices=[
            "Grayscale",
            "Noise",
            "Bright",
            "Flip 180º",
            "Blur"
        ]
    ).ask()

    parameters_default = {
        "Noise amount": "25",
        "Bright amount": "50",
        "Blur kernell": "5"
    }
    

    while True:
        console.clear()
        current_parameters = [f"{key}: {value}" for key, value in parameters_default.items()]
        current_parameters.append("Confirm and Continue")

        choice = questionary.select(
            "Select a parameter to edit:",
            choices = current_parameters
        ).ask()

        if choice == "Confirm and Continue":
            break
        key = choice.split(":")[0]

        new_value = questionary.text(f"Type the new value for {key}:").ask()
        parameters_default[key] = new_value
    
    with console.status("[bold green]Processing...[/bold green]", spinner="dots"):
    
        if "Grayscale" in option:
            gray_scale(imgs, labels, imgs_and_labels)
        if "Noise" in option:
            noise(imgs, labels, imgs_and_labels, parameters_default["Noise amount"])
        if "Bright" in option:
            bright(imgs, labels, imgs_and_labels, parameters_default["Bright amount"])
        if "Flip 180º" in option:
            flip(imgs, labels, imgs_and_labels)
        if "Blur" in option:
            blur(imgs, labels, imgs_and_labels, parameters_default["Blur kernell"])
    
    console.print("[bold green]Process finished :)[/bold green]")
    return

def menu():
    console.clear()
    console.print(Panel.fit("Data Augmentation Algorithm", subtitle="Main Menu"))
    console.print("[1] Start Process")
    console.print("[2] Default Values")
    console.print("[3] How To Use")
    console.print("[4] Exit")

def data_verification():
    console.clear()
    imgs_and_labels = get_imgs_and_labels_path()
    content = os.listdir(imgs_and_labels)

    imgs = []
    labels = []

    for file in content:
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        full_path = os.path.join(imgs_and_labels, file)

        if ext == '.txt':
            labels.append((name, full_path))
        if ext != '.txt' and ext != '.py':
            imgs.append((name, full_path))
        
    labels.sort(key=lambda x: x[0])
    imgs.sort(key=lambda x: x[0])
    
    img_dict = dict(imgs)
    label_dict = dict(labels)

    all_names = sorted(set(img_dict.keys()) | set(label_dict.keys()))

    table = Table(title="Data Verification")
    table.add_column("Image", justify="center")
    table.add_column("Label", justify="center")

    with Status("Verifying...", spinner="dots") as status:
        for name in all_names:
            time.sleep(0.1)

            img = img_dict.get(name)
            label = label_dict.get(name)

            img_text = Text(img if img else "Missing", style="green" if img and label else "red")
            label_text = Text(label if label else "Missing", style="green" if img and label else "red")

            console.print(f"[bold]Verifying:[/bold] [cyan]{img if img else 'Missing'}[/cyan] / [magenta]{label if label else 'Missing'}[/magenta]")

            table.add_row(img_text, label_text)
            console.print(table)

    name_imgs = set(img_dict.keys())
    name_labels = set(label_dict.keys())

    if name_imgs != name_labels:
        missing_labels = name_imgs - name_labels
        missing_imgs = name_labels - name_imgs

        if missing_labels:
            console.print(f"[red]These images don't have labels:\n" + "\n".join(missing_labels) + "[/red]")
            console.print("Please, fix the missing files. Press Enter to return to menu...")
            input("")
        if missing_imgs:
            console.print(f"[red]These labels don't have images:\n" + "\n".join(missing_imgs) + "[/red]")
            console.print("Please, fix the missing files. Press Enter to return to menu...")
            input("")

        return False, False, False, True

            
    elif len(imgs) != len(labels):
        console.print("\n[red]You have different amount of images and labels.[/red]")
        console.print("Please, fix the missing files. Press Enter to return to menu...")
        input("")
        return False, False, False, True
    
    elif name_imgs == name_labels and name_labels == 0:
        console.print("[red]\nThere is no images or labels in the imgs_and_labels folder[/red]") 
        console.print("Please, fix the missing files. Press Enter to return to menu...")
        input("")
        return False, False, False, True
    
    else:
        return imgs, labels, imgs_and_labels, False
    
def get_imgs_and_labels_path():
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, "imgs_and_labels")
   

def main():
    console.clear()
    
    while True:
        console.clear()
        menu()
        option = Prompt.ask("\nType an option: ")

        if option == "1":
            imgs, labels, imgs_and_labels, erro = data_verification()          
            if erro == False:
                UI(imgs, labels, imgs_and_labels)
            time.sleep(1)

        elif option == "2":
            console.clear()
            console.print(Panel.fit("Data Augmentation Algorithm", subtitle="Default Values"))
            console.print("\nKernell for Blur: 5x5")
            console.print("Noise Percentage: 25%")
            console.print("Bright Level: +50%")
            console.print("\nPress Enter to continue...")
            input("")
            time.sleep(1)

        elif option == "3":
            console.clear()
            #caminho = os.path.join(os.path.dirname(__file__), "README.md")
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            caminho = os.path.join(base_path, "README.md")    
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    markdown = Markdown(f.read())
                    Console().print(markdown)
                    time.sleep(1)
            except FileNotFoundError:
                Console().print("[red]README.md não encontrado.[/red]")
            print("\nPress Enter to continue ...")
            input("")

        elif option == "4":
            console.clear()
            break
        else:
            console.print("[red]Not an option![/red]")
            time.sleep(1)



if __name__ == "__main__":
    main()