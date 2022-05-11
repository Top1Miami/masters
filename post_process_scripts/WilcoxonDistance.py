import pandas as pd

for i in range(1, 7):
    df = pd.read_csv('../results/' + str(i) + '/comparison_pear/pmelif_plots_op/pmelif_dataset.csv')

# save_path = '../results/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
