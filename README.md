# NOTE: FOR STUDENTS ONLY
This repository contains additional content intended to have you better understand the following topics, which were not detailed in the repository only containing the implementation files:
1. [PyTorch beginner course](https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh)
2. [PySyft Split NN tutorial](https://notebook.community/OpenMined/PySyft/examples/tutorials/advanced/split_neural_network/Tutorial%201%20-%20SplitNN%20Introduction)
3. **Example of output results in CSV format**. The column names are self-explanatory, but just in case, here is an example with FSL:
   - "Server" cell = processor_0
   - "Edges" cell = processor_1, processor_3, processor_5
   - "Clients" cell = processor_2, processor4, processor_29
   - Then it means that the network configuration that lead to this specific training time uses processor_0 as the aggregation Server, processors 1, 3, and 5 as the Edges, and processors 2, 4, and 29 as the Clients. Treat all of these outputs as lists. If so, then **the i-th client is connected to the i-th edge**, and then all edges are connected to the aggregation server (processor_0). Clients are not linked together. Edges are not linked together.
4. Two files containing the code to **plot the linear plots and boxplots** used in the thesis
5. An Excel file containing all the potential output **error codes**
6. A PRE-PROCESSING folder containing the **code necessary to convert the EchoNet dataset, or any video dataset, to images**. It also includes code to divide these images into labels and their respective folders. You can mofidy this code to work with any video dataset of your choice. **The EchoNet itself is NOT included in this repo**. You must request access from Professor Flavio Esposito to do so!

The rest of the README is the same as the other repo containing only the implementation files.

# DNN Distributed Training Architecture Simulator: DDTAS 
DNN Distributed Training Architecture Simulator (DDTAS) is a tool used to compare the performance of various distributed training architectures for a user-specified Deep Neural Network (DNN). The tool allows the user to modify **any** of the following variables in the simulated network:

| Variable       | Name in Code     | Unit     |
| :----------: | :----------: | :----------: |
| Processor amount | max_processors | None |
 Processing power | C | GFLOP/s |
 Residual memory | R | GB |
 Residual fraction | E | None |
 Bandwidth | B | GB/s |
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
 Batches per client | total_batches_XX* | None |
 Epochs | epochs | None |
 Maximum calculated paths | max_paths | None |
 Minimum Split | min_split | None |
 Minimum clients | client_amount | None |
 Amount of events | event_amount | None |
 
** *Replace XX with the architecture abbreviation of your choice (PL, FL, SL, PSL, FSL)*

All of these variables can be modified in **config.ipynb**. Based on the user-specified variables (network conditions), the tool will then output the **fastest training time** and the accompanying **network graph** for the specified DNN. 



## Supported Distributed Training Architectures
DDTAS currently supports 5 following distributed training architectures:
1. Pipeline Learning (PL)
2. Federated Learning (FL)
3. Split Learning (SL)
4. Parallel Split Learning (PSL)
5. Federated Split Learning (FSL)



## Necessary Libraries
- Most necessary libraries to run DDTAS, including [PyTorch](https://pytorch.org/) and [PySyft](https://blog.openmined.org/tag/pysyft/) 0.2.9, are installed by following the instructions [here](https://github.com/OpenMined/PySyft/tree/PySyft/syft_0.2.x). 
- [Pandas](https://pandas.pydata.org/) is also required to be able to read the outputs of the simulator. 
- In case it is not installed already, install [NumPy](https://numpy.org/).

## Optional Libraries
- To monitor progress in a multi-variate analysis, install [tqdm](https://pypi.org/project/tqdm/). 

## How to Run
DDTAS can be run in multiple modes:

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


## Allocation Algorithms
Each of the individual allocation algorithms can be modified to the user's liking in the files:
- *ALGORITHM_PL.ipynb*
- *ALGORITHM_FL.ipynb*
- *ALGORITHM_SL.ipynb*
- *ALGORITHM_PSL.ipynb*
- *ALGORITHM_FSL.ipynb*

