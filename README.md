# Fully connected feed-forward neural network with gradient descent and momentum
## Local installation
To run the project on your operating system make sure you already have python 3 installed and then install the following packages:
```
   RUN pip install --upgrade pip
   RUN pip install opencv-python-headless
   RUN pip install matplotlib keras tensorflow
   RUN pip install pandas
```
## Docker
If you have a docker environment:
```
docker build -t nndl .                    # run the build phase 
docker container run --rm -v .:/app nndl  # to test changes
```

## Results
With local or dockerize, project's execution create a plot of errors for each combination of learning rate, momentum and number of neurons. These results are stored in ```result/errors``` folder and any other information, such as parameters and accuracy, are stored as logs in ```events.log``` in the same path. <br>
To change generated results, change parameters in ```properties.ini``` file.