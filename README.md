# CODE REPRODUCIBILITY FOR KDD 2022
This section is specifically directed at the reviewers for KDD 2022. To replicate our results and use our simulator, follow these simple steps. 
## Quick Version
This list of steps is done assuming you are working on a machine that uses Python or Jupyter Notebook regularly and just want to replicate our results from Section 4.4. 
1. Install Anaconda on the OS of your choice, following the steps for each platform listed [here](https://docs.anaconda.com/anaconda/install/index.html). Anaconda is a widely used Python virtual environment manager. 
2. Download our DTS.yml file. This is the instruction file to have Anaconda create an exact copy of our environment on your machine, **regardless of your OS**. 
3.  Navigate to the directory where you downloaded it and use the following command:
   ```
   conda env create -f DTS.yml
   ```
3. Download or clone our Git repository into a directory of your choice:
   ```
   git clone https://github.com/robinson08/DTS_KDD_Test.git
   ```
5. Activate the newly created Conda environment:
   ```
   conda activate DTS
   ```
5. Open *RUN_SIMULATOR.ipynb* and run all cells. 


## Detailed version
These steps apply if you are installing on a barebones machine OR want to modify the simulator params yourself to test it out.
1. Install Anaconda on the OS of your choice, following the steps for each platform listed [here](https://docs.anaconda.com/anaconda/install/index.html). Anaconda is a widely used Python virtual environment manager. 
2. **If you are installing on a barebones machine**, you will need the following essentials. Most machines already have these installed, but you can check by trying to run these commands. If they are already installed, the system will notify you. If not, the system will install the requirements. These are very general Python requirements, so **chances are that if you have been using Python or Jupyter Notebook regularly, these will already be installed in your system**. Since the following are not Conda-based commands, they are specific to Ubuntu machines. If your OS is not Ubuntu, simply use your equivalent commands:
   ```
   sudo apt-get install python-dev                 # Check if you have your Python header files installed
   pip install --upgrade setuptools                # Check if you have your pip upgrade tools up to date
   sudo apt-get install build-essential            # Get essential files for building wheels when installing packages
   ```
3. Download the DTS.yml file. Navigate to the directory where you downloaded it, and then create an exact copy of our development environment for you to run our simulator and modify its settings. To do so, simply use the following command:
   ```
   conda env create -f DTS.yml
   ```
4. Download or clone our Git repository into a directory of your choice:
   ```
   git clone https://github.com/robinson08/DTS_KDD_Test.git
   ``` 
5. Activate the newly created Conda environment:
   ```
   conda activate DTS
   ```
6. You are now ready to run any of the provided Jupyter Notebook files and run our simulator!
   - If you simply want to run it to replicate our results from Section 4.4, open *RUN_SIMULATOR.ipynb* and click run all cells. 
   - If you want to modify its parameters when running it to compare against a testbed, open *config*. This was used in Sections 4.2 and 4.3. 
   - When running it just to generate data and perform performance analyses, open *config_multiple.ipynb* and then Run All. This was used in Section 4.4.
   
## Adding your own architectures and algorithms
If you want to add an allocation algorithm of your choice, create it in a ipynb file, then do the following:
1. Open *RUN_SIMULATOR.ipynb* and type the following right after the last listed algorithm:
```
% run ALGORITHM_FSL.ipynb              # Last line of called algorithms
% run your_new_algorithm.ipynb        # Add in your own algorithm
```
For your algorithm, don't forget to use the same global variables that were created in *config.ipynb* to represent the available computer resources during each simulation round. These variable names are also listed in a table in the following section. 

2. Add your desired column names for the output CSVs in the file *Logger.ipynb*. 
3. Open *RUN_SIMULTOR.ipynb* and run all the cells!


[comment]: <> (Test comment)

# DNN Distributed Training Simulator Overview
DNN Distributed Training Simulator (DTS) is a tool used to compare the performance of various distributed training architectures for a user-specified Deep Neural Network (DNN). The tool allows the user to modify **any** of the following variables in the simulated network:

| Variable       | Name in Code     | Unit     |
| :----------: | :----------: | :----------: |
| Processor amount | max_processors | None |
 Processing power | C | GFLOP/s |
 Processing power fluctuation | C_variation | % |
 Parallelization factor | parall_factor | % |
 Residual memory | R | GB |
 Residual fraction | E | None |
 Bandwidth | B | GB/s |
 Bandwidth fluctuation | B_variation | % |
 Layer amount | max_layers | None |
 Network size | G_base | GFLOP |
 Memory required | M_base | GB |
 Server network size | G_server | GFLOP |
 Client network size  | G_client | GFLOP |
 Aggregation network size | G_agg | GFLOP |
 Server memory required | M_server | GB |
 Client memory required | M_client | GB |
 Aggregation-server memory required | M_agg | GB |
 Intermediate results | D_client_out | GB |
 Weights matrix | D_weights | GB |
 Individual sample size | batch_individual_file_size | GB |
 Batch size | batch_size | None |
 Batches per client | total_batches_* | None |
 Epochs | epochs | None |
 Maximum calculated paths | max_paths | None |
 Minimum Split | min_split | None |
 Minimum clients | client_amount | None |
 Amount of events | event_amount | None |
 
** *Replace XX with the architecture abbreviation of your choice (PL, FL, SL, PSL, FSL)*

All of these variables can be modified in **config.ipynb**. Based on the user-specified variables (network conditions), the tool will then output the **fastest training time** and the accompanying **network graph** for the specified DNN. 



## Supported Distributed Training Architectures
DTS currently supports 5 distributed training architectures:
1. Pipeline Learning (PL)
2. Federated Learning (FL)
3. Split Learning (SL)
4. Parallel Split Learning (PSL)
5. Federated Split Learning (FSL)
6. More architectures can be added as explained in the previous section. 

## How to Run
DTS can be run in multiple modes:

### Single Simulation Mode
1. Input all parameters listed in the table shown before into the *config.ipynb*. Save the file only, there is no need to run it. If the user wants to know the values of these parameters from a real NN implemented in PyTorch, refer to the Measurement Tools shown in the next section.
2. In *RUN_SIMULATOR.ipynb*, set the MODE variable to "SINGLE".
3. Define the output file name by modifying the *file_name* (without specifying a extension).
4. In *Logger.ipynb* in the method *logger_individual()*, define the output path for the results. All results will be saved with the file name specified in the previous step and will be saved in .CSV format. The user is free to modify the *logger_individual()* method to output the results in another format.
5. Run all cells in *RUN_SIMULATOR.ipynb* and wait for the result. 

### Multiple Simulation Mode
It is possible to run a sequence of multiple events where some variables remain constant while others randomly change from simulation to simulation. This process is ideal when performing a sensitivity analysis. To do so:
1.  Input the necessary parameters into the *config.ipynb* file. Save only, there is no need to run it
2.  In the *config_mode_multiple.ipynb* file, define which variables the user wants to analyze the influence of, by modifying the list *X_axis_variables*. 
3. Modify the *X_dict* dictionary to establish the constant *X* axis values of each of the analyzed variables. The minimum and maximum of this dictionary will also define the range of values a variable can take when it is being used as the Randomized variable. Save the file.
4.  In *Logger.ipynb* in the method *logger_multiple()*, define the output path for the simulation results. 
5.  Set the mode in *RUN_SIMULATOR.ipynb* to MODE == ''MULTIPLE".
6.  Run all cells and wait for the result

### Running Simulations with Parameters from a Real DNN
The procedure for this is almost identical to the two previous sections, with the exception that now the user needs to obtain these parameters from a real DNN. Doing so from just a segment of code in PyTorch is complicated, so we recommend using the measurement tools showcased in the next section. Once these tools have been used to obtain the NN and processing power parameters, simply repeat the processes shown above.


## How to Obtain the Simulator Parameters for your DNN

### MEASUREMENT_TOOL_NN
The purpose of this tool is to obtain the simulator's required NN parameters from an already implemented NN with PyTorch. The steps to use this tool are listed below:
1. Replace the contents in the class Net_Complete() with the contents of the user's NN of choice. This class should contain the entire NN, without splits. 
2. Replace the contents of the classes Net_server() and Net_client() with the splits of the user's NN of choice. As their name suggests, introduce the server-side NN's contents into the Net_server() class, and the client-side NN's contents into the Net_client() class. 
3. Finally, simply run the entirety of the code ("Run all cells") and the printed output will give the user the value of the necessary variables.

### MEASUREMENT_TOOL_POWER
A parameter that is crucial to the adequate functioning of the simulator is the "available processor power" parameter, named ''C" in the code. The steps to obtain this parameter are explained below:
1. Follow the same procedure as with the previous tool by replacing the NN already in the code with the desired NN. In this case, the only class that requires replacement is Net_Complete(). 
2. Modify the *training_data* variable to point towards a sample of the dataset with which the NN will be trained.
3. Modify the variables IMG_SIZE, BATCH_SIZE, and LR (learning rate). This is important, as the tool will execute a simple training scenario on the specified NN and measure the time the machine it took for the machine to finish all calculations. 
4. Replace the value of the variable NN_size with the G_base output from the previous tool.
5. *Optional*: Modify the training sequence outlined in the last cell of the notebook. This sequence will most likely not require any significant modification, as it follows a standard NN training command structure (forward pass, loss, backward pass, weight adjustment). 
6. Run all cells in the code. This will prompt the tool to train Net_Complete() with a batch sample of the input data. 
7. Once the tool is done quick-training the NN, the resulting output is the processing power allocated by the user's machine to the training of this NN, expressed in GFLOP/s. 

