Training Method
===============

-	5장부터 9장까지는 Training method에 대한 설명
-	여러 방법에 대한 짧은 소개가 대부분(자세한 설명x)
-	3가지를 주로 설명할 것
-	수식은 거의 다 다른 논문에서 인용

<img src = https://i.imgur.com/JRW6g8d.png width = 400>

```
-5. Gradient descent 설명
-6. Second-order method 설명
-7. 멀티레이어에서 쓰이는 헤시안 행렬 설명
-8. 다중레이어 헤시안으로 분석
-9. 다중레이어 신경망에 Second-order method 적용해보기
```

Index
-----

-	SGD (5-2,5-3)
-	Gauss-Newton (6-4)
-	Levenberg Marquardt (7-2, 9-1)

5 Convergence of Gradient Descent
---------------------------------

### 5-1 A Little Theory

-	1차원 gradient descent
-	최적의 learning rate 찾기
-	learing rate 범위를 어떻게 봐야 하는가
-	최적의 learning rate보다 작게하면 오래걸리고 크게 하면

<img src=https://i.imgur.com/6mukIph.png width="200">

<img src = https://i.imgur.com/sgPXutW.png width = 500>

-	이후에 나오는 수식들은 최적화 가중치를 찾는 과정들
-	처음에는 1차원 수식 설명
-	1차원에서 optimal learning rate 찾기
-	2차원에서 가중치 찾기
-	2차원에서 optimal learning rate 찾기
-	multiple dimensions에서 optimal learning rate 찾기
-	error function으로 LMS 사용

-	오차 제곱합이 최소화 되도록 모델 파라미터 p를 정하는 방법을 최소자승법(least square method)

#### 1차원일 때 loss function

<img src = https://i.imgur.com/JRW6g8d.png width = 400>

#### 최적의 Weight

<img src = https://i.imgur.com/Qfre6T5.png width = 400>

### 5-2 Examples

#### Linear Network

<img src=https://i.imgur.com/FW4af54.png width = 400>

추가
----

### SGD (Stochastic Gradient Descent)

-	확률 경사하강법
-	batch mode 대신 SGD 사용
-	(구체적인 설명이 없어서 아쉬움)

<img src=https://i.imgur.com/jkAwfFC.png width = 500>

### Multilayer Network

<img src = https://i.imgur.com/SzvaSyz.png width = 300>

### 5-3 Input Transformations and Error Surface Transformations Revisited

1.	입력 변수에서 평균 빼기
2.	입력 변수의 분산 정규화
3.	입력 변수의 상관관계 확인
4.	각각의 가중치에 learning rate를 따로 사용해라

참고
----

### Frist-order optimization

-	Gradient descent 알고리즘의 변형 알고리즘들
-	Momentum, NAG, AdaGrad, AdaDelta, RMSProp, Adam 등

### Second order optimization

-	Gradient descent 알고리즘의 변형 알고리즘들
-	비싼 작업이라 잘 사용 안함
-	왜 비쌀까? Hessian matrix 라는 2차 편미분 행렬을 계산한 후 역행렬을 구하기 때문

#### first와 second를 나누는 기준은 미분을 한 번 했냐 두 번했냐의 차이

-	2차 편미분이 포함된 식은 second
-	1차 편마분만 포함된 식은 first

---

6 Classical second order optimization methods
---------------------------------------------

-	second-order optimization methods가 왜 imparctical 한지 증명하지는 않을것

### 6-1 Newton Algorithm

-	뉴턴 알고리즘은 21번에서 23번을 빼준것

-	error function이 2차원일때 one step으로 수렴한다

#### 참고 사이트

-	[뉴턴법/뉴턴-랩슨법의 이해와 활용(Newton's method)](https://darkpgmr.tistory.com/58)

뉴턴 알고리즘과 gradient descent 비교<img src = https://i.imgur.com/TXmLscq.png width = 500>

### 6-2 Conjugate Gradient

-	켤레기울기법 또는 공역기울기법
-	수학에서 대칭인 양의 준정부호행렬(陽-準定符號行列, 영어: positive-semidefinite matrix)을 갖는 선형계의 해를 구하는 수치 알고리즘
-	헤시안 행렬을 사용하지않음 (왜?)
-	O(N) method 사용
-	bactch mode 사용

<img src = https://i.imgur.com/FWCFVjT.png width = 400>

first gradient descent에서 직각을 찾음 최소화하는 gradient descent direction을 찾음 벡터에서 다시 직각이 되는 gradient를 찾음

<img src = https://i.imgur.com/Btrof8P.png width = 500>

<img src = https://i.imgur.com/vXr3MMy.png width = 400>

<img src = https://i.imgur.com/Tsa365s.png width = 400>

### 6-3 Quasi-Newton (BFGS)

-	헤시안 역행렬의 추정을 이용해 계산하는 방법
-	오직 batch learning만 사용 (mini-batch 사용안함)
-	O(N^2) 알고리즘 사용
	-	M = positive definite Matrix
	-	M = I

<img src =https://i.imgur.com/eiMpFDS.png width = 200>

### 6-4 Gauss-Newton and Levenberg Marquardt

-	Levenberg–Marquardt 방법은 가우스-뉴턴법(Gauss–Newton method)과 Gradient descent 방법이 결합된 형태로 볼 수 있다.
-	Levenberg–Marquardt를 사용하거나 이해하기 위해서는 위 두가지 방법을 알아야한다,

-	Gauss-Newton and Levenberg Marquardt는 Jacobian(자코비언) 근사를 사용한다.

-	또한 주로 batch learning을 사용

-	O(N^2) 알고리즘 사용

-	MSE를 loss function으로 사용

<img src = https://i.imgur.com/9XugirZ.png width = 400>

#### 참고 사이트

-	[Levenberg-Marquardt 방법](http://blog.naver.com/PostView.nhn?blogId=tlaja&logNo=220735045887&categoryNo=42&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView)
-	[자코비안(Jacobian)이란 무엇인가](http://t-robotics.blogspot.com/2013/12/jacobian.html#.XJtLB5gzY2w)

### 7 Tricks to compute the Hessian information in multilayer networks

-	멀리레이터 신경망에서 계산하는 트릭

#### 7-1 Finite Difference

-	헤시안 정의<img src = https://i.imgur.com/dyhwlXP.png width = 300>

#### 7-2 Square Jacobian approximation for the Gauss-Newton and Levenberg-Marquardt algorithms

가우스 뉴턴과 리번버그 말콸디에프 알고리즘

loss function

<img src = https://i.imgur.com/mAwuyQQ.png width = 400>

Gradient

<img src = https://i.imgur.com/fhx9qgR.png width = 400>

<img src = https://i.imgur.com/BKoX4Yx.png width = 400>

#### 7-3 Backpropagating second derivatives

-	forward to compute this matrix

<img src = https://i.imgur.com/d1zERhq.png width = 400>

#### 7-4 Backpropagating the diagonal Hessian in neural nets

-	가우스-뉴턴 근사를 이용해 대각 헤시안 행렬을 구해보자

<img src = https://i.imgur.com/J2wQArC.png width = 400>

#### 7-5 Computing the product of the Hessian and a vector

-	헤시안 행렬 계산 방법

<img src = https://i.imgur.com/vKH1BOX.png width = 400>

### 8 Analysis of the Hessian in multi-layer networks

-	다층 신경망에서 헤시안행렬 분석

트레인 과정에서 large eigenvalues에서 발생하는 문제들에 대해 논증

데이터 : OCR

<img src = https://i.imgur.com/UlWWy62.png width = 500>

눈에 잘 안들어온다

보기 쉽게 log를 취해줌<img src = https://i.imgur.com/mb7s30p.png width = 500>

### 9 Applying Second Order Methods to Multilayer Networks

-	다층 신경망에 second-order method 적용해보기

<img src = https://i.imgur.com/6zPgTGL.png width = 400>

-	Full batch말고 mini-batch 쓰자

#### 9-1 A stochastic diagonal Levenberg Marquardf method

#### 9-2 Computing the principal Eigenvalue/vector of the Hessian

증명
