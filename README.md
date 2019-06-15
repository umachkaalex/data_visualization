This script allows to plot all paired (2d plots) and tripled (3d plots) combinations of columns from any dataset without manual defining names of columns: https://youtu.be/kMKZ3AxObfk

For example for Iris dataset with 6 columns (features) we can plot 6 pairs (2d graphs):
1. sepal length/sepal width
2. sepal length/petal length
3. sepal length/petal width
4. sepal width/petal length
5. sepal width/petal width
6. petal length/petal width

...and 3 triples of columns (3d graphs):
1. sepal length/sepal width/petal length
2. sepal length/petal length/petal width
3. sepal width/petal length/petal width

It has to be only one target column with up to 20 classes (or more, but you will need to enlarge amount of colors in 'functions.py' file by yourself). This column can be or string or float/interger. The feautures (columns) have to be only floats/integers.

To run script it is needed the following libraries. Install in the same order (if you cannot install traits, download it from the corresponding folder):
- traits 4.6.0
- VTK 8.1.2
- mayavi
- matplotlib
- pandas
- PyQT5
