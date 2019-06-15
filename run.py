from functions import *

# load dataset
data = pd.read_csv('./datasets/Titanic Database.csv')
# set dataset, target column, type of visualization: '2d', ''Pseudo-3D', '3D'
visualize_data(data, 'Survived', type='2D')
