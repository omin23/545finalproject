from Geneformer.geneformer import Classifier
import scanpy as sc
import datetime
import os

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

# set the output directory and prefix
output_prefix = "cm_classifier_test"
output_dir = f"Geneformer_project/runs/{datestamp}"


# get the data from the scanpy package
adata = sc.read("Datasets/filtered_ms_adata.h5ad")


# make the directory and catch any errors that may occur
try:
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_dir}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")


# set the classifier this is from the examples 
filter_data_dict={"cell_type":["oligodendrocyte A","oligodendrocyte precursor cell","oligodendrocyte C"]}


# set the training arguments
training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay":0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}

# set the classifier
cc = Classifier(classifier="cell",
                cell_state_dict={"state_key": "Sample Characteristic Ontology Term[disease]", "states": "Sample Characteristic Ontology Term[organism status]"},
                filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=None,
                freeze_layers=2,
                num_crossval_splits=1,
                forward_batch_size=200,
                nproc=8)

train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371", "1549", "1515"]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

train_test_id_split_dict = {
    "attr_key": "individual",
    "train": train_ids + eval_ids,
    "test": test_ids
}

# Example input_data_file for 30M model: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
cc.prepare_data(input_data_file="Datasets/filtered_ms_adata.h5ad",
                output_directory=output_dir,
                output_prefix=output_prefix,
                split_id_dict=train_test_id_split_dict)

train_valid_id_split_dict = {"attr_key": "individual",
                            "train": train_ids,
                            "eval": eval_ids}

all_metrics = cc.validate(model_directory="/path/to/Geneformer",
                          prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
                          id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                          output_directory=output_dir,
                          output_prefix=output_prefix,
                          split_id_dict=train_valid_id_split_dict,
                          n_hyperopt_trials=100)  # to optimize hyperparameters, set n_hyperopt_trials=100 (or alternative desired # of trials)


