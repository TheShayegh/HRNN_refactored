from library.utils import read_entries,create_experiment_csv
from library.validate_util import eval

import sys

def main():
    predictions = sys.argv[1]
    gold = sys.argv[2]

    pred_entries = read_entries(predictions)
    gold_entries = read_entries(gold)
 
    # These are to be printed or stored somewhere??
    f1, acc = eval(gold_entries, pred_entries)
    print("F1 Score: ", f1, "AND", "Accuracy =", acc)



    # with open(os.path.join(experiment_path, 'config.yml')) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # device = get_torch_device(config)
   
    # test_entries = read_entries(config['test_data'])
    # test_embeddings = torch.load(config['test_embeddings'], map_location=device)
    # test_data = list(zip(test_embeddings, test_entries))

   
    # test_file = create_experiment_csv(experiment_path,config,"test_results.csv",["Model","Loss","F1","Accuracy"])

    # validate(
    #     hrnn_model,
    #     test_data,
    #     test_entries,
    #     'trained',
    #     test_file,
    #     device=device
    # )

    
if __name__ == "__main__":
	main()
