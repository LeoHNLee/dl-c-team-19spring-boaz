Efficient BackProp
==================

Summary
-------

-	order
	1.	intro
	2.	7 tricks
	3.	2 training methods
	4.	further more
-	other references
	-	[gradient, jacobian, hessian, laplacian](https://darkpgmr.tistory.com/132)

Abstract
--------

-	***연구자들의 back-prop convergence learning에 대한 분석***
-	***tricks*** to avoid undesirable behaviors of back-prop
-	classic : 2nd order optimization, not work in large NN
-	***few methods*** : no limit

1 Itroduction
-------------

-	back-prop : NN learning algorithm
	-	cuz : simple concept, efficient computation, often works
-	arbitrary choice : \# of nodes, layers, learning rates, train/test set

2 Learning and Generalization
-----------------------------

-	![fig01](fig01.JPG)

3 Standard Backpropagation
--------------------------

-	![equation01](equation01.JPG)
-	![equation02](equation02.JPG)
-	![equation03](equation03.JPG)

4 A Few Practical Tricks
------------------------

-	back-prop disadvantage : slow
-	why? cost surface is non-quadratic, non-convex, n-D => local min / flat region
-	no formula to guarantee bellow
	1.	NN converge to good sol
	2.	convergence is swift
	3.	convergence occurs at all

### 4-1 Stochastic VS Batch learning

-	advantage of SGD
	1.	fast
	2.	better sol
	3.	can track changes
-	advantage of batch
	1.	condition of convergence are well understood
	2.	many acc tech (eg. conjugate gradient)
	3.	theoretical analysis of the weight dynamics and convergence rates are simpler

### 4-2 Shuffling the Examples

-	choose examples w/ max info content
	1.	shuffle the training set => training examples never belong to same class
	2.	input examples often create large error than small

### 4-3 Normalizing the Inputs

-	transforming the inputs
	1.	mean of each input variable in train-set => 0
	2.	Scaling => same covar
	3.	uncorrelated
-	![fig03](fig03.JPG)

### 4-4 The Sigmoid

-	nonlinear activation function give NN nonlinear capability
-	sigmoid : standard logistic function, $`f(x) = \frac{1}{1+e^{-x}}`$
	-	monotone increasing
	-	asymptote finite value
	-	var(output) close to 1
-	symmetric sigmoids : eg) $`f(x)=\tanh(x)`$
	-	faster than standard logistic

![compare standard logistic and symmetric sigmoid](fig04.JPG)

-	advanced
	-	stimes, computationally expensive => ratio of polynomials
	-	stimes, to add linear term is helpful : $`f(x)=\tanh(x)+ax`$
	-	constants : **when transformed inputs**
	-	(now) ReLU and further more.
-	problems
	-	***error surface can be very flat near the origin***
		-	to add linear term stimes helpful
	-	***inputs should be normaalized to produce outputs that are on average close to zero***

### 4-5 Choosing Target Values

-	binary target problem
	1.	instablility : $`W\rightarrow \infty`$, where $`\partial S \rightarrow 0`$ (S=sigmoid)
	2.	if input falls near a decision boundary, output should be two values
-	set target to be within range of sigmoid rather than asymptotic value

### 4-6 Initializing the weights

-	warning
	-	if all W are large, sigmoid results small gradient that making learning slow
	-	if very small, also gradient will be very small
-	normalizing to sd 1, output of each nodes, before activation function

### 4.7 Choosing Learning rates

-	give each weight its own learning rate
-	lr should be proportional to the sqrt of the # of inputs to the unit
-	weights in lower layers should typically be larger than in the higher layers

#### adaptive learning rate

### 4.8 Radical Basis Functions vs Sigmoid Units

-	radical foo : 무리함수

-	Convergence of Gradient Descent
	-------------------------------

### 5.1 A Little Theory

### 5.2 Examples

### 5.3 Input Transformations and Error Surface Transformations Revisited

1.	Classical second order optimization methods ----------------------------------------------

### 6.1 Newton Algorithm

### 6.2 Conjugate Gradient

### 6.3 Quasi-Newton (BFGS)

### 6.4 Gauss-Newton and Levenberg Marquardt

1.	Tricks to compute the Hessian information in multilayer networks -------------------------------------------------------------------

### 7.1 Finite Difference

### 7.2 Square Jacobian approximation for the Gauss-Newton and Levenberg-Marquardt algorithms

### 7.3 Backpropagating second derivatives

### 7.4 Backpropagating the diagonal Hessian in neural nets

### 7.5 computing the product of the Hessian and a vector

1.	Analysis of the Hessian in multi-layer networks -----------------------------------------------
2.	Applying Second Order Methods to Multilayer Networks ----------------------------------------------------

### 9.1 A stochastic diagonal Levenberg Marquardt method

### 9.2 Computing the principal Eigenvalue/vector of the Hessian

1.	Discussion and conclusion -------------------------
2.	tricks

	1.	[shuffle the examples](#4-2-shuffling-the-examples)
	2.	[normalizing](#4-3-normalizing-the-inputs)
		-	centering : the input variables by subtracting the mean
		-	$`\sigma = 1`$ : normalize the input variable to a standard deviation of 1
		-	decorrelate : (if possible)
	3.	[sigmoid](#4-4-the-sigmoid) : w/ NN
		-	![pick a network with the sigmoid function](fig04.JPG)
	4.	$`-1\le target\le 1`$[link](#4-5-choosing-target-values) : set target values within sigmoid range
	5.	[init W to rv](#4-6-initializing-the-weights)

3.	training method

	1.	SGD w/ careful tuning | stochastic diagonal Levenberg Marquardt method : if classification & huge data (what's mean redundant?)
	2.	conjugate gradient : if small data | regression

4.	bad : classic 2nd order method

Reference
---------
