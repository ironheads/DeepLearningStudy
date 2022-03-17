# DeepLearning

## BatchNorm梯度计算

$L$是损失函数，

$y=\gamma \hat{x} + \beta$
$$
\frac{\partial L}{\partial \gamma} =\sum_{i}{\frac{\partial L}{\partial y_{i}}\frac{\partial y_i}{\partial \gamma}} =  \sum_{i}{\frac{\partial L}{\partial y_{i}}\hat{x}_{i}}\\
\frac{\partial L}{\partial \beta} =\sum_{i}{\frac{\partial L}{\partial y_{i}}\frac{\partial y_i}{\partial \beta}}= \sum_{i}{\frac{\partial L}{\partial y_{i}}}
$$

## Dropout梯度计算

$$
x=(x_1,x_2,x_3,\cdots,x_n) \\

\hat{x}=dropout(x) \\

\frac{\part L}{\part x}=\frac{\part L}{\part \hat{x}}\frac{\part \hat{x}}{\part x} \\

\frac{\part \hat{x}}{\part x} = \mathbf{M} \\

\mathbf{M}_j= \left\{
\begin{aligned}
0 & , & r_j <p \\
\frac{1}{1-p} & , & r_j \geq p 
\end{aligned}
\right. \\

\frac{\part L}{\part x}=\frac{\part L}{\part \hat{x}}\mathbf{M}
$$

##
