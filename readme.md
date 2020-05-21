Stock Trend Prediction based on News Title

## Package Installation
1. following the [guide](https://docs.anaconda.com/anaconda/install/) in Anaconda,install the Anaconda first.
2. create the virtual environment by using Anaconda:  
    `conda create -n **env_name** python=3.6.8`
3. activate the created virtual environment and install the required packages:
       
    ```.env
    conda activate **env_name**
    pip install -r requiments.txt
    ```

## Model Training and Inference visualization
Follow the step in [classification.ipynb](classification.ipynb), you will train a stock trend prediction model based on the news.
In the end, you will get two saved model:
1. transform.m: saving the required feature transformation (including tf-idf transform, topic vectorization, and sentiment related feauture generation)
2. rf.m: save the trained random forest model

Then, you can run the following code to access the inference visualization web:
    `python run.py`
    
This flask app will be located in [127.0.0.1:5000](http://127.0.0.1:5000/). In case, the port 5000 is already used by some other app, you can change the port number of flask app in [run.py](run.py).