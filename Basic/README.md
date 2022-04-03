## Run :

`python3 main.py [-c] <config file in json format> `

- Uses code from `src` folder
- Saves plots in `analysis_plots/<run_name>` folder


## Tests : 

* Running individual tests : 
    ```
    python -m unittest -v tests.test_nh
    python -m unittest -v tests.test_leach
    ```

* Running all tests : 
    ```
    python -m unittest discover -v
    ```