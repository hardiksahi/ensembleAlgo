# ensembleAlgo
This is the code for random forests algorithm. It uses decision stumps as weak learner. 
It has a configurable number of weak learners.
It is not as fast as industrial level random forest implementation because bootstrapping training set takes a lot of time and can be done much more efficiently and faster using parallelization instead of sequential evaluation.
