# PytorchMNIST-CNN

This is an implementation of a CNN (Convolutional neural network) to classify hand written digits from the MNIST 
dataset in pytorch

This project also contains code to serve the model in simple API built using flask and python

This project was made for learning purposes and to explore CNN's and image classification, I am only learning so there things that can be done in a better way please feel free to leave an issue with any suggestions

# Made Using

- pytorch for Machine Learning
- Flask for the backend
- PIL (Python Imaging Library) for reading in images

# Resources and inspiration

In order to create this application I followed along with 

- [This](https://youtu.be/GIsg-ZUy0MY) Tutorial on YouTube by [FreeCodeCamp](https://www.freecodecamp.org/) about Deep learning in pytorch
- [This](https://youtu.be/bA7-DEtYCNM) Tutorial on YouTube by [Python Engineer](https://www.youtube.com/c/PythonEngineer/featured) about deploying a pytorch model with a flask API

# How to setup the project

In order to use this project locally you will need to have python installed, which you can download from [Here](https://www.python.org/downloads/)

## Installation

After downloading and installing python you need to set up a virtual environment on your machine

For windows users:

```powershell
python -m venv [Name]
```
For Mac users:

```shell
python3 -m venv [Name] 
```
where Name is the name of the environment

Unfortunately I haven't used a linux system 😢

Go into the environment

This is same for Mac and Windows

```powershell
cd [Name]
```

Inside this directory there should be a folder called "venv"

the next step is to clone the project from GitHub

## Running the program

In order to run the program activate the virtual environment, you can activate the environment using

Windows:

```powershell
venv\Scripts\activate.bat
```

Mac:

```shell
source venv/bin/activate
```

your terminal or command prompt should show a

    (venv) 

in front

then you need to install the dependencies for the project

Windows:

```powershell
pip install -r requirements.txt
```

Mac:

```bash
pip3 install -r requirements.txt
```

now you can run the script

Windows:

```powershell
cd app
run.bat
```
> run.bat is a small script to set the environment variables for flask and start up the backend
> run.sh is same as run.bat for mac 

On Mac this process is a tiny bit more complicated

```shell
cd app
chmod 755 run.sh
./run.sh
```

and a server should be up and listening on localhost port 5000

# todo

- Accept base64 images
- Add a some front end for the project
- Deploy it on either Heroku or Azure (TBD)
