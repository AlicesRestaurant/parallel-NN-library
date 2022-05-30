# Parallel Neural Network Library

Parallel Neural Network Library provides the user with tools needed for parallel training,
evaluation and of the model.

The library features:
1. ```Model``` class representing any popular kind of 
MLP (Multilayer Perceptron) with method to adjust the architecture of the model to your
needs.
2. Own matrix class `MatrixD` with parallel matrix multiplication.
3. `Trainer` class for training your models.
4. Different ways of parallelization:
   1. MPI
   2. MatrixD matrix-matrix multiplication

## Prerequisites
You need the following tools preinstalled to use the library:
- Eigen3
- MPI
- Boost library (Boost::mpi and Boost::serialization should be present)
- 

## Compilation (Linux)
Run the following in the project's root directory
```bash
$ mkdir build
$ cmake -B build -S . -DUSE_EIGEN=OFF
$ cmake --build build
```

To compile with own implementation of matrix, set `-DUSE_EIGEN=ON` when generating
build files.

## Usage

### Creating Model
```c++
int numberOfIputNodes = 1;
std::shared_ptr lossPtr = std::make_shared<MSELossFunction>();
Model model{1, lossPtr};
```

### Setting Layers
Use `addLayer()` template method to set layers of the model:
```c++
model.addLayer<FCLayer>(10, 1);
model.addLayer<SigmoidActivationLayer>(10);
model.addLayer<FCLayer>(10, 1);
model.addLayer<SigmoidActivationLayer>(10);
model.addLayer<FCLayer>(1, 10);
```

### Find training and test datasets
Store the training datasets either in this library own `MatrixD` or
in `Eigen::MatrixXd`. Note that all function and methods of the library assume that each column correspond the example,
while each row corresponds to the feature.
#### MatrixD
```c++
MatrixD trainX{{1, 2, 3, 4}};
MatrixD trainY{{-1, -4, -9, -16}};

MatrixD trainX{{2.5, 3.5}};
MatrixD trainY{{6.25, 12.25}};
```

#### Eigen::MatrixXd
```c++
Eigen::MatrixXd trainX{{1, 2, 3, 4}};
Eigen::MatrixXd trainY{{-1, -4, -9, -16}};

Eigen::MatrixXd trainX{{2.5, 3.5}};
Eigen::MatrixXd trainY{{6.25, 12.25}};
```


### Training the model
After creating model and setting layers, your can train it using `train()` method:
```c++
model.trainBatch(trainX, trainY, alpha);
```

To see other way of training the model, see `Trainer` class that encapsulates the training 
process.

### DistributedTrainer class
Alternative way to train the model is with `DistributedTrainer` class ecapsulating MPI:
```c++
mpi::environment env;
mpi::communicator comm;

MatrixD trainX = ...;
MatrixD trainY = ...;

size_t numInputsNeurons = trainX.rows();
size_t numOutputNeurons = trainY.rows();
Model model{numInputsNeurons, std::make_shared<SoftMaxLossFunction>()};
model.addLayer<FCLayer>(numOutputNeurons, numInputsNeurons, -1.0/255, 1.0/255);

size_t batchSize = 30;
size_t numProcessors = comm.size();

DistributedTrainer distTrainer{std::make_shared<Model>(model), batchSize, alpha, 
                               std::make_shared<mpi::communicator>(comm), numProcessors};

distTrainer.trainDataset(trainImages, trainLabels, numIters);
```

### Evaluating the model
Use `forwardPass()` method to calculate output of the model:d
```c++
MatrixD testX = ...;
auto predicted = model.forwardPass(testX);
std::cout << "Predictions: " << predicted << std::endl;
```
