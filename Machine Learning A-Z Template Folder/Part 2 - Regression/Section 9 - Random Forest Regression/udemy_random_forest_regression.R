# Random Forest Regression
# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)


# Fitting Regression Model to the dataset
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
# keep in mind x is a dataframe and y is a vector
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)


# Predicting a new result 
y_pred = predict(regressor, data.frame(Level = 6.5))




# Visualizing the Random Forestresults (for higher resolution and smoother curve )
library(ggplot2)
# for better resolution, changed 0.1 to 0.01
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest)') +
  xlab('Enter lable for x here') +
  ylab('Enter lable for y here')
