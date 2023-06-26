# Fully connected feed-forward neural network with gradient descent and momentum
A simple project that analyzes the error and accuracy of a neural network based on properties such as activation functions, number of neurons, and momentum.
## Local installation
To run the project on your operating system make sure you already have python 3 installed and then install the following packages:
```
   RUN pip install --upgrade pip
   RUN pip install opencv-python-headless
   RUN pip install matplotlib keras tensorflow
```
## Docker
If you have a docker environment:
```
docker build -t nndl .                    # run the build phase 
docker container run --rm -v .:/app nndl  # to test changes
```

## Results
In the local or dockerize environment, project execution creates error graphs for any combination of learning rate, momentum, and number of neurons. These results are stored in the ```results/errors``` folder and any other information, such as parameters and accuracy, are stored as logs in ```events.log``` in the same location.<br>
To change the generated results, change the parameters in the ```properties.ini``` file.