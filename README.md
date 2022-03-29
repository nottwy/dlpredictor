# Manual of dlpredictor

# 1. Installation
The program is written with the python3 grammar so you need to use python3 rather than python2.
## 1.1. required python package

# 2. Format of Input data
Our program needs two input files, a feature file used to store mutations of each sample and a label file used to store the status of being resistant or susceptible. Users should prepare their input files as stated below. 
## 2.1 feature file
One sample per row and one feature per column. Columns are delimited with comma.
      
- column 1 .. n: For each column, there are four status.
    - -1: indicating missing values here.
    - 0:  indicating no mutation found here.
    - 1:  For SNP, 1 indicates there's a mutation.
    - n:  For InDel, n = abs (len(alt) - len(ref)), where len() means the length of bases
                     and abs() means abosolute value.
- row 1 .. n: each row represents one individual.

## 2.2 label file
One sample per row and one label per column. Columns are delimited with comma.

You should ensure that individuals' phenotypes for each drug containing both susceptible and resistant ones. 
Our program will remove drugs whose phenotypes don't contain both susceptible and resistant ones.

- column 1 .. n: each column represents the individuals' phenotypes for this type of drug.
- row 1: header line, drug names.
- row 2 .. n: each row represents one individual.
  - resistant(1)
  - susceptible(0)
  - unknown(-1).
  
# 3. Parameters
Our program supports two ways to specify parameters, reading from a config file or specifying them in command line.
## 3.1 reading from a config file
Use --conf option to specify the name of your config file.

*python dlpredictor.py --conf config_file.cfg*

### The format of config file 

    model=model_name
    fpath=/path/to/feature_file
    lpath=/path/to/label_file
    prefix=prefix_of_output_file
 
## 3.2 specifying parameters in command line
You can run our program with -h option to see the help information and use the options you need. The options about inputs and the model you choose are necessary. Others are optional.

*python dlpredictor.py --model model_name --fpath /path/to/feature_file --lpath /path/to/label_file*

# 4. new model created by inheritance
## 4.1 procedure:
  1. 在dlAdvanced.py中，继承类并覆盖需要修改的函数.  
     ```
     class WDNN_for_yak(WDNN):
        def optimal_threshold(ptrue, ppred):
     ```
  2. 在dlAdvanced.py中，将新增的类添加到支持列表里  
     ```
     my_Model = {  
     "wdnn": WDNN,
     "deepamr": DeepAMR,
     "wdnn_cnngwp": WDNN_cnngwp,
     "WDNN_for_yak": WDNN_for_yak,
     "deepamr_modified": DeepAMR_modified
     }
     ```
  3. 在parameters.cfg处，选择正确的参数  
     `model="WDNN_for_yak`
