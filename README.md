# patch_based_cnn
the  implement  of  Face Anti-Spoofing Using Patch and Depth-Based CNNs

# Introduction
- data
    - The data after dividing into different patchs.
- model
    - the code of net
- output
    - save model and logs
- src
    - main function and other auxiliary functions
- config.py
    - parameter configuration
- utils.py
    - Shared basic functions

# Logic of code
- First run generate.py to divide the living and spoofing img into different patchs 
according to paper and save them as the test set and training set in the data floder.
In this section, users need to prepare their own data or public data sets.We did not upload the data.
You need to modify the path to your own data path, and modify the parameters, run four times to get four dataset: living test, living training, spoofing test, spoofing training.   

- Then train and test.

# Run

- Divide img to different patchs(64 is used in article)
    ```
    python3 data_generate.py
    ```

- Train and test
    ```
    python3 patch_cnn_main.py
    ```


