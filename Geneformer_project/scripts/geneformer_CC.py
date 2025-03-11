from Geneformer.geneformer import Classifier
import scanpy as sc
import anndata
import datetime
import os
# import scvi
# this is to import the scvi package and test the scvi package
# adata = scvi.data.pbmc_dataset()


current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

# set the output directory and prefix
output_prefix = "cm_classifier_test"
output_dir = f"Geneformer_project/runs/{datestamp}"


# get the data from the scanpy package
adata = sc.read("Datasets/filtered_ms_adata.h5ad")
print (adata)


# make the directory and catch any errors that may occur
try:
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_dir}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")


# set the classifier this is from the examples 
filter_data_dict={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]}

training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay":0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}

cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "disease", "states": "all"},
                filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=None,
                freeze_layers = 2,
                num_crossval_splits = 1,
                forward_batch_size=200,
                nproc=16)





