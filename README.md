# Data_Augmentation
This repository is made to help perform data augmentation in images for neural network training, such as with YOLO. It automates five types augmentations for your dataset: Grayscale; Noise Addition; Brightness Ajustment; Blur; and 180º Flip. The augmentation process is customizable, meaning you can select which one of these transformations you want to apply in your dataset and manually adjust the parameters of each one. In case of any questions or sujestions, please open an [issue](https://github.com/Ruzkita/Data_Augmentation/issues) and let me know. If you find this project useful, consider giving this repository a star.

## How to use:
There are two ways to use this repository: you can either clone the entire repository and run the script/.exe file, or simply download the executable file for your OS and run it.

### Cloning the repository:
In order to clone this repository, open your terminal (Linux) or powershell (Windows) and run:

```bash
git clone https://github.com/Ruzkita/Data_Augmentation/
```

Then, run:
```bash
pip install -r requirements.txt
```
To install all dependencies required to run the script (you only need to install the dependencies if plan to run the script instead of the executable). After that, you can run the script using your preferred Python IDE, or run the executable file for your OS.

### Downloading the executable:
If you don't want to clone the entire repository, you can simply download the executable file for your system and run it.
- On Windows you can simply double click the file.
- On Linux, in order to run the executable, you need to give it permission. Open the folder with the executable within and run the following command:

```bash
chmod +x ./data_augmentation_for_linux
```

After that, run the file with:

```bash
./data_augmentation_for_linux
```

### Setting your Dataset
To correctly set you dataset, place all your images and labels in the same folder and name it **imgs_and_labels**.
If you run the script/executable for the first time and the **imgs_and_labels** folder doesn't exist, it will be created automatically in the same directory as the executable. After it's created, place your images and labels together inside the folder and proceed with the augmentation.
