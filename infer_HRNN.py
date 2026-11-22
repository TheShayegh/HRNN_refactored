'''
This file is for inferring pseudo labels from an HRNN model. 
It takes the HRNN model and config file from an experiments folder. It also takes the input file path and the output file name as arguments.
To run this, we need to run this command:
python infer_HRNN.py PATH_TO_EXP_FOLDER INPUT_FILE_PATH INPUT_EMBEDDING_FILE_PATH OUTPUT_FILE_NAME
It takes 4 arguments: 
1. PATH TO EXP_FOLDER
2. INPUT FILE PATH
3. INPUT FILE EMBEDDINGS PATH
4. OUTPUT_FILE_NAME
'''
from library.utils import get_torch_device, read_entries
from library.HRNN import HRNNtagger
import os
import yaml
import torch
import sys

def main():
    # Read path to experiments folder and get config file from the folder
    experiment_path = sys.argv[1]
    with open(os.path.join(experiment_path, 'HRNN_config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Obtain device and data file path, read entries for data
    device = get_torch_device(config)
   
    data_file_path = sys.argv[2]
    embeddings_file_path = sys.argv[3]
    
    data_entries = read_entries(data_file_path)
    # Extract the relevant embedding file for the input data file
    # filename = os.path.basename(data_file_path)  # Extracts only the filename with extension
    # name = filename.split('/')[-1].split('.')[0]  # Extracts the word between the last / and .
    # embeddings = torch.load(config[name + '_embeddings'], map_location=device)

    # second argument is path to folder containing input files
    # read this from config, it is a list of files and not a folder
    
    embeddings = torch.load(embeddings_file_path, map_location=device)

    data = list(zip(embeddings, data_entries))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    
    model_path = os.path.join(experiment_path,'best_model.pt')
    hrnn_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
   
    loss, output_entries = hrnn_model.predict(
        data,
        [entry.get_words() for entry in data_entries],
        device=device,
    )
    
    # Read output file name
    output_file_name = sys.argv[4]

    # There will be an output folder in the same experiments folder that will contain the output file.
    output_folder = os.path.join(experiment_path,'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder,output_file_name)
    # Write output to the file in txt format
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i in range(len(output_entries)):
            f.write(str(output_entries[i])+'\n')

    
if __name__ == "__main__":
	main()
