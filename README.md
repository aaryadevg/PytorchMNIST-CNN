# PytorchMNIST-CNN

This is an implementation of a CNN (Convolutional neural network) to classify hand written digits from the MNIST 
dataset in pytorch

This project also contains code to serve the model in simple API built using flask and python

## How to setup the project

In order to use this project locally you will need to have python installed, which you can download from [Here](https://www.python.org/downloads/)

### Installation

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

### Running the program

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