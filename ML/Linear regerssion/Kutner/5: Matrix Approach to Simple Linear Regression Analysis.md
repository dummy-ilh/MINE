Chapter 5: Matrix Approach to Simple Linear Regression Analysis
Overview
Matrix algebra is widely used for mathematical and statistical analysis. The matrix approach is practically a necessity in multiple regression analysis, since it permits extensive systems of equations and large arrays of data to be denoted compactly and operated upon efficiently.
Why Learn Matrix Methods for Simple Linear Regression?
Although matrix algebra is not really required for simple linear regression, the application of matrix methods to this case will provide a useful transition to multiple regression, which will be taken up in Parts II and III.
Chapter Structure:

Brief introduction to matrix algebra (comprehensive treatments available in specialized texts like Reference 5.1)
Application of matrix methods to the simple linear regression model discussed in previous chapters


Note for Experienced Readers: Those familiar with matrix algebra may wish to scan the introductory parts and focus upon the later parts dealing with the use of matrix methods in regression analysis.


5.1 Matrices
Definition of Matrix
A matrix is a rectangular array of elements arranged in rows and columns.
Example of a Matrix:
Column 1Column 2Row 116,00023Row 233,00047Row 321,00035
Interpretation: The elements of this particular matrix are numbers representing income (column 1) and age (column 2) of three persons. The elements are arranged by row (person) and column (characteristic of person).
Key Points:

The element in the first row and first column (16,000) represents the income of the first person
The element in the first row and second column (23) represents the age of the first person
The dimension of the matrix is 3×23 \times 2
3×2 (i.e., 3 rows by 2 columns)



Dimension Notation
If we wanted to present income and age for 1,000 persons in a matrix with the same format, we would require a 1,000×21,000 \times 2
1,000×2 matrix.

Convention: When giving the dimension of a matrix, we always specify the number of rows first and then the number of columns.

Examples of Other Matrices
[10510][47121631598]\begin{bmatrix} 1 & 0 \\ 5 & 10 \end{bmatrix} \quad\quad \begin{bmatrix} 4 & 7 & 12 & 16 \\ 3 & 15 & 9 & 8 \end{bmatrix}[15​010​][43​715​129​168​]
These two matrices have dimensions of 2×22 \times 2
2×2 and 2×42 \times 4
2×4, respectively.


Element Notation
We may use symbols to identify the elements of a matrix. For instance:
$$\begin{matrix}
& j=1 & j=2 & j=3 \
i=1 & a_{11} & a_{12} & a_{13} \
i=2 & a_{21} & a_{22} & a_{23}
\end{matrix}$$
Key Convention: The first subscript identifies the row number and the second the column number.
General notation aija_{ij}
aij​ = element in the ii
ith row and the jj
jth column.

In our above example, i=1,2i = 1, 2
i=1,2 and j=1,2,3j = 1, 2, 3
j=1,2,3.


Matrix Symbols
A matrix may be denoted by a symbol such as A, X, or Z. The symbol is in boldface to identify that it refers to a matrix.
Example: For the above matrix:
A=[a11a12a13a21a22a23]\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}A=[a11​a21​​a12​a22​​a13​a23​​]
Reference to the matrix A then implies reference to the 2×32 \times 3
2×3 array just given.


Compact Notation
Another notation for the matrix A just given is:
A=[aij]i=1,2;j=1,2,3\mathbf{A} = [a_{ij}] \quad\quad i = 1, 2; j = 1, 2, 3A=[aij​]i=1,2;j=1,2,3
This notation avoids the need for writing out all elements of the matrix by stating only the general element. It can only be used when the elements of a matrix are symbols.

General Matrix Representation
To summarize, a matrix with rr
r rows and cc
c columns will be represented either in full:

$$\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1j} & \cdots & a_{1c} \
a_{21} & a_{22} & \cdots & a_{2j} & \cdots & a_{2c} \
\vdots & \vdots &  & \vdots &  & \vdots \
a_{i1} & a_{i2} & \cdots & a_{ij} & \cdots & a_{ic} \
\vdots & \vdots &  & \vdots &  & \vdots \
a_{r1} & a_{r2} & \cdots & a_{rj} & \cdots & a_{rc}
\end{bmatrix}$$
(5.1)
or in abbreviated form:
A=[aij]i=1,…,r;j=1,…,c\mathbf{A} = [a_{ij}] \quad\quad i = 1, \ldots, r; j = 1, \ldots, cA=[aij​]i=1,…,r;j=1,…,c
or simply by a boldface symbol, such as A.

Important Comment on Matrix Definition
**Comment 1**: Do not think of a matrix as a number. It is a set of elements arranged in an array. Only when the matrix has dimension 1×11 \times 1
1×1 is there a single number in a matrix, in which case one *can* think of it interchangeably as either a matrix or a number.


What is NOT a Matrix
The following is not a matrix:
$$\begin{bmatrix}
& 14 \
& 8 \
10 & 15 \
9 & 16
\end{bmatrix}$$
since the numbers are not arranged in columns and rows. ■

5.2 Square Matrix
A matrix is said to be square if the number of rows equals the number of columns.
Two Examples:
[4739][a11a12a13a21a22a23a31a32a33]\begin{bmatrix} 4 & 7 \\ 3 & 9 \end{bmatrix} \quad\quad \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}[43​79​]​a11​a21​a31​​a12​a22​a32​​a13​a23​a33​​​
Both are square matrices (dimensions 2×22 \times 2
2×2 and 3×33 \times 3
3×3, respectively).


5.3 Vector
Column Vector
A matrix containing only one column is called a column vector or simply a vector.
Two Examples:
A=[4710]C=[c1c2c3c4c5]\mathbf{A} = \begin{bmatrix} 4 \\ 7 \\ 10 \end{bmatrix} \quad\quad \mathbf{C} = \begin{bmatrix} c_1 \\ c_2 \\ c_3 \\ c_4 \\ c_5 \end{bmatrix}A=​4710​​C=​c1​c2​c3​c4​c5​​​
The vector A is a 3×13 \times 1
3×1 matrix, and the vector
C is a 5×15 \times 1
5×1 matrix.


Row Vector
A matrix containing only one row is called a row vector.
Two Examples:
B′=[152550]F′=[f1f2]\mathbf{B'} = [15 \quad 25 \quad 50] \quad\quad \mathbf{F'} = [f_1 \quad f_2]B′=[152550]F′=[f1​f2​]
Important Convention: We use the prime symbol for row vectors for reasons to be seen shortly.
Key Point: The row vector B′\mathbf{B'}
B′ is a 1×31 \times 3
1×3 matrix and the row vector F′\mathbf{F'}
F′ is a 1×21 \times 2
1×2 matrix.

A single subscript suffices to identify the elements of a vector.

5.4 Transpose
The transpose of a matrix A is another matrix, denoted by A′\mathbf{A'}
A′, that is obtained by interchanging corresponding columns and rows of the matrix
A.
Example:
If:

A3×2=[2571034]\mathbf{A}_{3 \times 2} = \begin{bmatrix} 2 & 5 \\ 7 & 10 \\ 3 & 4 \end{bmatrix}A3×2​=​273​5104​​
then the transpose A′\mathbf{A'}
A′ is:

A′2×3=[2735104]\mathbf{A'}_{2 \times 3} = \begin{bmatrix} 2 & 7 & 3 \\ 5 & 10 & 4 \end{bmatrix}A′2×3​=[25​710​34​]
Key Observation:

The first column of A is the first row of A′\mathbf{A'}
A′
The second column of A is the second row of A′\mathbf{A'}
A′
Correspondingly, the first row of A has become the first column of A′\mathbf{A'}
A′

Dimension Change: Note that the dimension of A, indicated under the symbol A, becomes reversed for the dimension of A′\mathbf{A'}
A′.


Another Example
As another example, consider:
C3×1=[4710]C′1×3=[4710]\mathbf{C}_{3 \times 1} = \begin{bmatrix} 4 \\ 7 \\ 10 \end{bmatrix} \quad\quad \mathbf{C'}_{1 \times 3} = [4 \quad 7 \quad 10]C3×1​=​4710​​C′1×3​=[4710]
Thus, the transpose of a column vector is a row vector, and vice versa. This is the reason why we used the symbol B′\mathbf{B'}
B′ earlier to identify a row vector, since it may be thought of as the transpose of a column vector
B.

General Transpose Formula
In general, we have:
$$\mathbf{A}{r \times c} = \begin{bmatrix}
a{11} & \cdots & a_{1c} \
\vdots &  & \vdots \
a_{r1} & \cdots & a_{rc}
\end{bmatrix} = [a_{ij}] \quad\quad i = 1, \ldots, r; j = 1, \ldots, c$$
(5.2)
$$\mathbf{A'}{c \times r} = \begin{bmatrix}
a{11} & \cdots & a_{r1} \
\vdots &  & \vdots \
a_{1c} & \cdots & a_{rc}
\end{bmatrix} = [a_{ji}] \quad\quad j = 1, \ldots, c; i = 1, \ldots, r$$
(5.3)
Thus, the element in the ii
ith row and jj
jth column in
A is found in the jj
jth row and ii
ith column in A′\mathbf{A'}
A′.


5.5 Equality of Matrices
Two matrices A and B are said to be equal if they have the same dimension and if all corresponding elements are equal. Conversely, if two matrices are equal, their corresponding elements are equal.
Example:
If:

A3×1=[a1a2a3]B3×1=[473]\mathbf{A}_{3 \times 1} = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix} \quad\quad \mathbf{B}_{3 \times 1} = \begin{bmatrix} 4 \\ 7 \\ 3 \end{bmatrix}A3×1​=​a1​a2​a3​​​B3×1​=​473​​
then A=B\mathbf{A} = \mathbf{B}
A=B implies:

a1=4a2=7a3=3a_1 = 4 \quad\quad a_2 = 7 \quad\quad a_3 = 3a1​=4a2​=7a3​=3

Another Example
Similarly, if:
A3×2=[a11a12a21a22a31a32]B3×2=[172145139]\mathbf{A}_{3 \times 2} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix} \quad\quad \mathbf{B}_{3 \times 2} = \begin{bmatrix} 17 & 2 \\ 14 & 5 \\ 13 & 9 \end{bmatrix}A3×2​=​a11​a21​a31​​a12​a22​a32​​​B3×2​=​171413​259​​
then A=B\mathbf{A} = \mathbf{B}
A=B implies:

a11=17a12=2a_{11} = 17 \quad a_{12} = 2a11​=17a12​=2
a21=14a22=5a_{21} = 14 \quad a_{22} = 5a21​=14a22​=5
a31=13a32=9a_{31} = 13 \quad a_{32} = 9a31​=13a32​=9

5.6 Regression Examples
Observations Vector Y
In regression analysis, one basic matrix is the vector Y, consisting of the nn
n observations on the response variable:

\mathbf{Y}_{n \times 1} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix}$
(5.4)
Note that the transpose Y′\mathbf{Y'}
Y′ is the row vector:

\mathbf{Y'}_{1 \times n} = [Y_1 \quad Y_2 \quad \cdots \quad Y_n]$
(5.5)

Design Matrix X
Another basic matrix in regression analysis is the X matrix, which is defined as follows for simple linear regression analysis:
$$\mathbf{X}_{n \times 2} = \begin{bmatrix}
1 & X_1 \
1 & X_2 \
\vdots & \vdots \
1 & X_n
\end{bmatrix}$$
(5.6)
The matrix X consists of a column of 1s and a column containing the nn
n observations on the predictor variable XX
X.

Note: The transpose of X is:
\mathbf{X'}_{2 \times n} = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ X_1 & X_2 & \cdots & X_n \end{bmatrix}$
(5.7)
The X matrix is often referred to as the design matrix.

5.7 Matrix Addition and Subtraction
Adding or subtracting two matrices requires that they have the same dimension. The sum, or difference, of two matrices is another matrix whose elements each consist of the sum, or difference, of the corresponding elements of the two matrices.
Suppose:
A3×2=[142536]B3×2=[122334]\mathbf{A}_{3 \times 2} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix} \quad\quad \mathbf{B}_{3 \times 2} = \begin{bmatrix} 1 & 2 \\ 2 & 3 \\ 3 & 4 \end{bmatrix}A3×2​=​123​456​​B3×2​=​123​234​​
then:
A+B3×2=[1+14+22+25+33+36+4]=[2648610]\mathbf{A} + \mathbf{B}_{3 \times 2} = \begin{bmatrix} 1+1 & 4+2 \\ 2+2 & 5+3 \\ 3+3 & 6+4 \end{bmatrix} = \begin{bmatrix} 2 & 6 \\ 4 & 8 \\ 6 & 10 \end{bmatrix}A+B3×2​=​1+12+23+3​4+25+36+4​​=​246​6810​​

Subtraction Example
Similarly:
A−B3×2=[1−14−22−25−33−36−4]=[020202]\mathbf{A} - \mathbf{B}_{3 \times 2} = \begin{bmatrix} 1-1 & 4-2 \\ 2-2 & 5-3 \\ 3-3 & 6-4 \end{bmatrix} = \begin{bmatrix} 0 & 2 \\ 0 & 2 \\ 0 & 2 \end{bmatrix}A−B3×2​=​1−12−23−3​4−25−36−4​​=​000​222​​

General Formula
In general, if:
Ar×c=[aij]Br×c=[bij]i=1,…,r;j=1,…,c\mathbf{A}_{r \times c} = [a_{ij}] \quad\quad \mathbf{B}_{r \times c} = [b_{ij}] \quad\quad i = 1, \ldots, r; j = 1, \ldots, cAr×c​=[aij​]Br×c​=[bij​]i=1,…,r;j=1,…,c
then:
\mathbf{A} + \mathbf{B}_{r \times c} = [a_{ij} + b_{ij}] \quad\text{and}\quad \mathbf{A} - \mathbf{B}_{r \times c} = [a_{ij} - b_{ij}]$
**(5.8)**

Formula (5.8) generalizes in an obvious way to addition and subtraction of more than two matrices. Note also that A+B=B+A\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}
A+B=B+A, as in ordinary algebra.


Regression Example
The regression model:
Yi=E{Yi}+εii=1,…,nY_i = E\{Y_i\} + \varepsilon_i \quad\quad i = 1, \ldots, nYi​=E{Yi​}+εi​i=1,…,n
can be written compactly in matrix notation. First, let us define the vector of the mean responses:
\mathbf{E\{Y\}}_{n \times 1} = \begin{bmatrix} E\{Y_1\} \\ E\{Y_2\} \\ \vdots \\ E\{Y_n\} \end{bmatrix}$
(5.9)
and the vector of the error terms:
\boldsymbol{\varepsilon}_{n \times 1} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}$
(5.10)
Recalling the definition of the observations vector Y in (5.4), we can write the regression model as follows:
Yn×1⏟n×1=E{Y}n×1⏟n×1+εn×1⏟n×1\underbrace{\mathbf{Y}_{n \times 1}}_{n \times 1} = \underbrace{\mathbf{E\{Y\}}_{n \times 1}}_{n \times 1} + \underbrace{\boldsymbol{\varepsilon}_{n \times 1}}_{n \times 1}n×1Yn×1​​​=n×1E{Y}n×1​​​+n×1εn×1​​​
because:
[Y1Y2⋮Yn]=[E{Y1}E{Y2}⋮E{Yn}]+[ε1ε2⋮εn]=[E{Y1}+ε1E{Y2}+ε2⋮E{Yn}+εn]\begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} = \begin{bmatrix} E\{Y_1\} \\ E\{Y_2\} \\ \vdots \\ E\{Y_n\} \end{bmatrix} + \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix} = \begin{bmatrix} E\{Y_1\} + \varepsilon_1 \\ E\{Y_2\} + \varepsilon_2 \\ \vdots \\ E\{Y_n\} + \varepsilon_n \end{bmatrix}​Y1​Y2​⋮Yn​​​=​E{Y1​}E{Y2​}⋮E{Yn​}​​+​ε1​ε2​⋮εn​​​=​E{Y1​}+ε1​E{Y2​}+ε2​⋮E{Yn​}+εn​​​
Thus, the observations vector Y equals the sum of two vectors, a vector containing the expected values and another containing the error terms.

5.8 Matrix Multiplication
Multiplication of a Matrix by a Scalar
A scalar is an ordinary number or a symbol representing a number. In multiplication of a matrix by a scalar, every element of the matrix is multiplied by the scalar.
For example, suppose the matrix A is given by:
A=[2793]\mathbf{A} = \begin{bmatrix} 2 & 7 \\ 9 & 3 \end{bmatrix}A=[29​73​]
Then 4A, where 4 is the scalar, equals:
4A=4[2793]=[8283612]4\mathbf{A} = 4\begin{bmatrix} 2 & 7 \\ 9 & 3 \end{bmatrix} = \begin{bmatrix} 8 & 28 \\ 36 & 12 \end{bmatrix}4A=4[29​73​]=[836​2812​]

Another Scalar Multiplication Example
Similarly, kAk\mathbf{A}
kA equals:

kA=k[2793]=[2k7k9k3k]k\mathbf{A} = k\begin{bmatrix} 2 & 7 \\ 9 & 3 \end{bmatrix} = \begin{bmatrix} 2k & 7k \\ 9k & 3k \end{bmatrix}kA=k[29​73​]=[2k9k​7k3k​]
where kk
k denotes a scalar.


Factoring Out Common Scalar
If every element of a matrix has a common factor, this factor can be taken outside the matrix and treated as a scalar. For example:
[9271518]=3[3956]\begin{bmatrix} 9 & 27 \\ 15 & 18 \end{bmatrix} = 3\begin{bmatrix} 3 & 9 \\ 5 & 6 \end{bmatrix}[915​2718​]=3[35​96​]
Similarly:
[52kk38kk]=1k[52k38]\begin{bmatrix} 5 & 2 \\ k & k \\ 3 & 8 \\ k & k \end{bmatrix} = \frac{1}{k}\begin{bmatrix} 5 & 2 \\ k & 3 & 8 \end{bmatrix}​5k3k​2k8k​​=k1​[5k​23​8​]

General Formula
In general, if A=[aij]\mathbf{A} = [a_{ij}]
A=[aij​] and kk
k is a scalar, we have:

k\mathbf{A} = \mathbf{A}k = [ka_{ij}]$
(5.11)

5.9 Multiplication of a Matrix by a Matrix
Multiplication of a matrix by a matrix may appear somewhat complicated at first, but a little practice will make it a routine operation.
Consider the two matrices:
A2×2=[2541]B2×2=[4658]\mathbf{A}_{2 \times 2} = \begin{bmatrix} 2 & 5 \\ 4 & 1 \end{bmatrix} \quad\quad \mathbf{B}_{2 \times 2} = \begin{bmatrix} 4 & 6 \\ 5 & 8 \end{bmatrix}A2×2​=[24​51​]B2×2​=[45​68​]
The product AB will be a 2×22 \times 2
2×2 matrix whose elements are obtained by finding the
cross products of rows of A with columns of B and summing the cross products.

Finding Element (1,1) of AB
For instance, to find the element in the first row and the first column of the product AB, we work with the first row of A and the first column of B, as follows:
A:
Row 1: [2  5]
B:
Col. 1: [4, 5]ᵀ
AB:
Row 1, Col. 1: [33]
We take the cross products and sum:

2(4)+5(5)=332(4) + 5(5) = 332(4)+5(5)=33
The number 33 is the element in the first row and first column of the matrix AB.

Finding Element (1,2) of AB
To find the element in the first row and second column of AB, we work with the first row of A and the second column of B:
A:
Row 1: [2  5]
B:
Col. 1, Col. 2: [4, 6; 5, 8]
AB:
Row 1: [33  52]
The sum of the cross products is:

2(6)+5(8)=522(6) + 5(8) = 522(6)+5(8)=52

Completing the Product
Continuing this process, we find the product AB to be:
AB2×2=[2541][4658]=[33522132]\mathbf{AB}_{2 \times 2} = \begin{bmatrix} 2 & 5 \\ 4 & 1 \end{bmatrix}\begin{bmatrix} 4 & 6 \\ 5 & 8 \end{bmatrix} = \begin{bmatrix} 33 & 52 \\ 21 & 32 \end{bmatrix}AB2×2​=[24​51​][45​68​]=[3321​5232​]

Another Example
Let us consider another example:
A2×3=[134058]B3×1=[352]\mathbf{A}_{2 \times 3} = \begin{bmatrix} 1 & 3 & 4 \\ 0 & 5 & 8 \end{bmatrix} \quad\quad \mathbf{B}_{3 \times 1} = \begin{bmatrix} 3 \\ 5 \\ 2 \end{bmatrix}A2×3​=[10​35​48​]B3×1​=​352​​
AB2×1=[134058][352]=[2641]\mathbf{AB}_{2 \times 1} = \begin{bmatrix} 1 & 3 & 4 \\ 0 & 5 & 8 \end{bmatrix}\begin{bmatrix} 3 \\ 5 \\ 2 \end{bmatrix} = \begin{bmatrix} 26 \\ 41 \end{bmatrix}AB2×1​=[10​35​48​]​352​​=[2641​]

Important Note on Multiplication Order
When obtaining the product AB, we say that A is postmultiplied by B or B is premultiplied by A.
Critical Rule: The reason for this precise terminology is that multiplication rules for ordinary algebra do not apply to matrix algebra. In ordinary algebra, xy=yxxy = yx
xy=yx. In matrix algebra, AB≠BA\mathbf{AB} \neq \mathbf{BA}
AB=BA usually.

In fact, even though the product AB may be defined, the product BA may not be defined at all.

When is Matrix Multiplication Defined?
In general, the product AB is defined only when the number of columns in A equals the number of rows in B so that there will be corresponding terms in the cross products. Thus, in our previous two examples, we had:
Example 1:

A: 2×2, B: 2×2 → AB: 2×2 ✓
Number of columns in A (2) = Number of rows in B (2)

Example 2:

A: 2×3, B: 3×1 → AB: 2×1 ✓
Number of columns in A (3) = Number of rows in B (3)


Dimension of Product
Key Rule: Note that the dimension of the product AB is given by the number of rows in A and the number of columns in B. Note also that in the second case the product BA would not be defined since the number of columns in B is not equal to the number of rows in A.
Unequal Case:

B: 3×1, A: 2×3 → BA: undefined ✗
Number of columns in B (1) ≠ Number of rows in A (2)


General Matrix Multiplication Formula
Here is another example of matrix multiplication:
AB=[a11a12a13a21a22a23][b11b12b21b22b31b32]\mathbf{AB} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}\begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \\ b_{31} & b_{32} \end{bmatrix}AB=[a11​a21​​a12​a22​​a13​a23​​]​b11​b21​b31​​b12​b22​b32​​​
=[a11b11+a12b21+a13b31a11b12+a12b22+a13b32a21b11+a22b21+a23b31a21b12+a22b22+a23b32]= \begin{bmatrix} a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} & a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32} \\ a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31} & a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32} \end{bmatrix}=[a11​b11​+a12​b21​+a13​b31​a21​b11​+a22​b21​+a23​b31​​a11​b12​+a12​b22​+a13​b32​a21​b12​+a22​b22​+a23​b32​​]
In general, if A has dimension r×cr \times c
r×c and
B has dimension c×sc \times s
c×s, the product
AB is a matrix of dimension r×sr \times s
r×s whose element in the ii
ith row and jj
jth column is:

∑k=1caikbkj\sum_{k=1}^{c} a_{ik}b_{kj}k=1∑c​aik​bkj​
so that:
\mathbf{AB}_{r \times s} = \left[\sum_{k=1}^{c} a_{ik}b_{kj}\right] \quad\quad i = 1, \ldots, r; j = 1, \ldots, s$
**(5.12)**

Thus, in the foregoing example, the element in the first row and second column of the product AB is:
∑k=13a1kbk2=a11b12+a12b22+a13b32\sum_{k=1}^{3} a_{1k}b_{k2} = a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32}k=1∑3​a1k​bk2​=a11​b12​+a12​b22​+a13​b32​
as indeed we found by taking the cross products of the elements in the first row of A and second column of B and summing.

Additional Examples
Example 1:
[4258][a1a2]=[4a1+2a25a1+8a2]\begin{bmatrix} 4 & 2 \\ 5 & 8 \end{bmatrix}\begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} 4a_1 + 2a_2 \\ 5a_1 + 8a_2 \end{bmatrix}[45​28​][a1​a2​​]=[4a1​+2a2​5a1​+8a2​​]
Example 2:
[235][235]=[22+32+52]=[38][2 \quad 3 \quad 5]\begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = [2^2 + 3^2 + 5^2] = [38][235]​235​​=[22+32+52]=[38]
Here, the product is a 1×11 \times 1
1×1 matrix, which is equivalent to a scalar. Thus, the matrix product here equals the number 38.

Example 3:
[1X11X21X3][β0β1]=[β0+β1X1β0+β1X2β0+β1X3]\begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ 1 & X_3 \end{bmatrix}\begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} = \begin{bmatrix} \beta_0 + \beta_1 X_1 \\ \beta_0 + \beta_1 X_2 \\ \beta_0 + \beta_1 X_3 \end{bmatrix}​111​X1​X2​X3​​​[β0​β1​​]=​β0​+β1​X1​β0​+β1​X2​β0​+β1​X3​​​

Regression Example: Y′Y\mathbf{Y'Y}
Y′Y
A product frequently needed is Y′Y\mathbf{Y'Y}
Y′Y, where
Y is the vector of observations on the response variable as defined in (5.4):
\mathbf{Y'Y}_{1 \times 1} = [Y_1 \quad Y_2 \quad \cdots \quad Y_n]\begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} = [Y_1^2 + Y_2^2 + \cdots + Y_n^2] = \left[\sum Y_i^2\right]$
(5.13)
Note that Y′Y\mathbf{Y'Y}
Y′Y is a 1×11 \times 1
1×1 matrix, or a scalar. We thus have a compact way of writing a sum of squared terms: Y′Y=∑Yi2\mathbf{Y'Y} = \sum Y_i^2
Y′Y=∑Yi2​.


Regression Example: X′X\mathbf{X'X}
X′X
We also will need X′X\mathbf{X'X}
X′X, which is a 2×22 \times 2
2×2 matrix, where
X is defined in (5.6):
\mathbf{X'X}_{2 \times 2} = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ X_1 & X_2 & \cdots & X_n \end{bmatrix}\begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix} = \begin{bmatrix} n & \sum X_i \\ \sum X_i & \sum X_i^2 \end{bmatrix}$
(5.14)

Regression Example: X′Y\mathbf{X'Y}
X′Y
and X′Y\mathbf{X'Y}
X′Y, which is a 2×12 \times 1
2×1 matrix:

\mathbf{X'Y}_{2 \times 1} = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ X_1 & X_2 & \cdots & X_n \end{bmatrix}\begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} = \begin{bmatrix} \sum Y_i \\ \sum X_i Y_i \end{bmatrix}$
(5.15)

5.10 Special Types of Matrices
Certain special types of matrices arise regularly in regression analysis. We consider the most important of these.
Symmetric Matrix
If A=A′\mathbf{A} = \mathbf{A'}
A=A′,
A is said to be symmetric. Thus, A below is symmetric:
A3×3=[146425653]A′3×3=[146425653]\mathbf{A}_{3 \times 3} = \begin{bmatrix} 1 & 4 & 6 \\ 4 & 2 & 5 \\ 6 & 5 & 3 \end{bmatrix} \quad\quad \mathbf{A'}_{3 \times 3} = \begin{bmatrix} 1 & 4 & 6 \\ 4 & 2 & 5 \\ 6 & 5 & 3 \end{bmatrix}A3×3​=​146​425​653​​A′3×3​=​146​425​653​​
Key Property: A symmetric matrix necessarily is square. Symmetric matrices arise typically in regression analysis when we premultiply a matrix, say, X, by its transpose, X′\mathbf{X'}
X′. The resulting matrix, X′X\mathbf{X'X}
X′X, is symmetric, as can readily be seen from (5.14).


Diagonal Matrix
A diagonal matrix is a square matrix whose off-diagonal elements are all zeros, such as:
A3×3=[a1000a2000a3]B4×4=[40000100001000005]\mathbf{A}_{3 \times 3} = \begin{bmatrix} a_1 & 0 & 0 \\ 0 & a_2 & 0 \\ 0 & 0 & a_3 \end{bmatrix} \quad\quad \mathbf{B}_{4 \times 4} = \begin{bmatrix} 4 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 10 & 0 \\ 0 & 0 & 0 & 5 \end{bmatrix}A3×3​=​a1​00​0a2​0​00a3​​​B4×4​=​4000​0100​00100​0005​​
We will often not show all zeros for a diagonal matrix, presenting it in the form:
A3×3=[a1a2a3]B4×4=[41105]\mathbf{A}_{3 \times 3} = \begin{bmatrix} a_1 & & \\ & a_2 & \\ & & a_3 \end{bmatrix} \quad\quad \mathbf{B}_{4 \times 4} = \begin{bmatrix} 4 & & & \\ & 1 & & \\ & & 10 & \\ & & & 5 \end{bmatrix}A3×3​=​a1​​a2​​a3​​​B4×4​=​4​1​10​5​​

Identity Matrix
Two important types of diagonal matrices are the identity matrix and the scalar matrix.
Identity Matrix: The identity matrix or unit matrix is denoted by I. It is a diagonal matrix whose elements on the main diagonal are all 1s. Premultiplying or postmultiplying any r×rr \times r
r×r matrix
A by the r×rr \times r
r×r identity matrix
I leaves A unchanged. For example:
IA=[100010001][a11a12a13a21a22a23a31a32a33]=[a11a12a13a21a22a23a31a32a33]\mathbf{IA} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}IA=​100​010​001​​​a11​a21​a31​​a12​a22​a32​​a13​a23​a33​​​=​a11​a21​a31​​a12​a22​a32​​a13​a23​a33​​​
Similarly, we have:
AI=[a11a12a13a21a22a23a31a32a33][100010001]=[a11a12a13a21a22a23a31a32a33]\mathbf{AI} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}AI=​a11​a21​a31​​a12​a22​a32​​a13​a23​a33​​​​100​010​001​​=​a11​a21​a31​​a12​a22​a32​​a13​a23​a33​​​
Key Property: Note that the identity matrix I therefore corresponds to the number 1 in ordinary algebra, since we have there that 1⋅x=x⋅1=x1 \cdot x = x \cdot 1 = x
1⋅x=x⋅1=x.

In general, we have for any r×rr \times r
r×r matrix
A:
\mathbf{AI} = \mathbf{IA} = \mathbf{A}$
(5.16)
Thus, the identity matrix can be inserted or dropped from a matrix expression whenever it is convenient to do so.

Scalar Matrix
Scalar Matrix: A scalar matrix is a diagonal matrix whose main-diagonal elements are the same. Two examples of scalar matrices are:
[2002][k000k000k]\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \quad\quad \begin{bmatrix} k & 0 & 0 \\ 0 & k & 0 \\ 0 & 0 & k \end{bmatrix}[20​02​]​k00​0k0​00k​​
A scalar matrix can be expressed as kIk\mathbf{I}
kI, where kk
k is the scalar. For instance:

[2002]=2[1001]=2I\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} = 2\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 2\mathbf{I}[20​02​]=2[10​01​]=2I
[k000k000k]=k[100010001]=kI\begin{bmatrix} k & 0 & 0 \\ 0 & k & 0 \\ 0 & 0 & k \end{bmatrix} = k\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = k\mathbf{I}​k00​0k0​00k​​=k​100​010​001​​=kI
Key Property: Multiplying an r×rr \times r
r×r matrix
A by the r×rr \times r
r×r scalar matrix kIk\mathbf{I}
kI is equivalent to multiplying
A by the scalar kk
k.


Vector and Matrix with All Elements Unity
A column vector with all elements 1 will be denoted by 1:
\mathbf{1}_{r \times 1} = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}$
(5.17)
and a square matrix with all elements 1 will be denoted by J:
\mathbf{J}_{r \times r} = \begin{bmatrix} 1 & \cdots & 1 \\ \vdots &  & \vdots \\ 1 & \cdots & 1 \end{bmatrix}$
(5.18)

Examples
For instance, we have:
13×1=[111]J3×3=[111111111]\mathbf{1}_{3 \times 1} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \quad\quad \mathbf{J}_{3 \times 3} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}13×1​=​111​​J3×3​=​111​111​111​​
Useful Properties:
Note that for an n×1n \times 1
n×1 vector
1 we obtain:
1′11×1=[1⋯1][1⋮1]=[n]=n\mathbf{1'1}_{1 \times 1} = [1 \quad \cdots \quad 1]\begin{bmatrix} 1 \\ \vdots \\ 1 \end{bmatrix} = [n] = n1′11×1​=[1⋯1]​1⋮1​​=[n]=n
and:
11′n×n=[1⋮1][1⋯1]=[1⋯1⋮⋮1⋯1]=Jn×n\mathbf{11'}_{n \times n} = \begin{bmatrix} 1 \\ \vdots \\ 1 \end{bmatrix}[1 \quad \cdots \quad 1] = \begin{bmatrix} 1 & \cdots & 1 \\ \vdots &  & \vdots \\ 1 & \cdots & 1 \end{bmatrix} = \mathbf{J}_{n \times n}11′n×n​=​1⋮1​​[1⋯1]=​1⋮1​⋯⋯​1⋮1​​=Jn×n​

Zero Vector
A zero vector is a vector containing only zeros. The zero column vector will be denoted by 0:
\mathbf{0}_{r \times 1} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$
(5.19)
For example, we have:
03×1=[000]\mathbf{0}_{3 \times 1} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}03×1​=​000​​

5.11 Linear Dependence and Rank of Matrix
Linear Dependence
Consider the following matrix:
A=[12512210634151]\mathbf{A} = \begin{bmatrix} 1 & 2 & 5 & 1 \\ 2 & 2 & 10 & 6 \\ 3 & 4 & 15 & 1 \end{bmatrix}A=​123​224​51015​161​​
Let us think now of the columns of this matrix as vectors. Thus, we view A as being made up of four column vectors. It happens here that the columns are interrelated in a special manner. Note that the third column vector is a multiple of the first column vector:
[51015]=5[123]\begin{bmatrix} 5 \\ 10 \\ 15 \end{bmatrix} = 5\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}​51015​​=5​123​​
We say that the columns of A are linearly dependent. They contain redundant information, so to speak, since one column can be obtained as a linear combination of the others.

Definition of Linear Dependence
We define the set of cc
c column vectors C1,…,Cc\mathbf{C}_1, \ldots, \mathbf{C}_c
C1​,…,Cc​ in an r×cr \times c
r×c matrix to be
linearly dependent if one vector can be expressed as a linear combination of the others. If no vector in the set can be so expressed, we define the set of cc
c vectors to be
linearly independent. A more general, though equivalent, definition is:
When cc
c scalars k1,…,kck_1, \ldots, k_c
k1​,…,kc​, not all zero, can be found such that:

k1C1+k2C2+⋯+kcCc=0k_1\mathbf{C}_1 + k_2\mathbf{C}_2 + \cdots + k_c\mathbf{C}_c = \mathbf{0}k1​C1​+k2​C2​+⋯+kc​Cc​=0
where 0 denotes the zero column vector, the cc
c column vectors are
linearly dependent (5.20). If the only set of scalars for which the equality holds is k1=0,…,kc=0k_1 = 0, \ldots, k_c = 0
k1​=0,…,kc​=0, the set of cc
c column vectors is
linearly independent.
To illustrate for our example, k1=5,k2=0,k3=−1,k4=0k_1 = 5, k_2 = 0, k_3 = -1, k_4 = 0
k1​=5,k2​=0,k3​=−1,k4​=0 leads to:

5[123]+0[224]−1[51015]+0[161]=[000]5\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + 0\begin{bmatrix} 2 \\ 2 \\ 4 \end{bmatrix} - 1\begin{bmatrix} 5 \\ 10 \\ 15 \end{bmatrix} + 0\begin{bmatrix} 1 \\ 6 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}5​123​​+0​224​​−1​51015​​+0​161​​=​000​​
Hence, the column vectors are linearly dependent. Note that some of the kjk_j
kj​ equal zero here. For linear dependence, it is only required that not all kjk_j
kj​ be zero.


Rank of Matrix
The rank of a matrix is defined to be the maximum number of linearly independent columns in the matrix. We know that the rank of A in our earlier example cannot be 4, since the four columns are linearly dependent. We can, however, find three columns (1, 2, and 4) which are linearly independent. There are no scalars k1,k2,k4k_1, k_2, k_4
k1​,k2​,k4​ such that k1C1+k2C2+k4C4=0k_1\mathbf{C}_1 + k_2\mathbf{C}_2 + k_4\mathbf{C}_4 = \mathbf{0}
k1​C1​+k2​C2​+k4​C4​=0 other than k1=k2=k4=0k_1 = k_2 = k_4 = 0
k1​=k2​=k4​=0. Thus, the rank of
A in our example is 3.
Important Property: The rank of a matrix is unique and can equivalently be defined as the maximum number of linearly independent rows. It follows that the rank of an r×cr \times c
r×c matrix cannot exceed min⁡(r,c)\min(r, c)
min(r,c), the minimum of the two values rr
r and cc
c.


Rank of Product Matrices
When a matrix is the product of two matrices, its rank cannot exceed the smaller of the two ranks for the matrices being multiplied. Thus, if C=AB\mathbf{C} = \mathbf{AB}
C=AB, the rank of
C cannot exceed min⁡(rank A,rank B)\min(\text{rank } \mathbf{A}, \text{rank } \mathbf{B})
min(rank A,rank B).


5.12 Inverse of a Matrix
In ordinary algebra, the inverse of a number is its reciprocal. Thus, the inverse of 6 is 16\frac{1}{6}
61​. A number multiplied by its inverse always equals 1:

6⋅16=16⋅6=16 \cdot \frac{1}{6} = \frac{1}{6} \cdot 6 = 16⋅61​=61​⋅6=1
x⋅1x=x⋅x−1=x−1⋅x=1x \cdot \frac{1}{x} = x \cdot x^{-1} = x^{-1} \cdot x = 1x⋅x1​=x⋅x−1=x−1⋅x=1
In matrix algebra, the inverse of a matrix A is another matrix, denoted by A−1\mathbf{A}^{-1}
A−1, such that:

\mathbf{A}^{-1}\mathbf{A} = \mathbf{AA}^{-1} = \mathbf{I}$
(5.21)
where I is the identity matrix. Thus, again, the identity matrix I plays the same role as the number 1 in ordinary algebra.
Important Limitation: An inverse of a matrix is defined only for square matrices. Even so, many square matrices do not have inverses. If a square matrix does have an inverse, the inverse is unique.

Examples of Matrix Inverses
Example 1: The inverse of the matrix:
A2×2=[2431]\mathbf{A}_{2 \times 2} = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}A2×2​=[23​41​]
is:
A2×2−1=[−.1.4.3−.2]\mathbf{A}^{-1}_{2 \times 2} = \begin{bmatrix} -.1 & .4 \\ .3 & -.2 \end{bmatrix}A2×2−1​=[−.1.3​.4−.2​]
since:
A−1A=[−.1.4.3−.2][2431]=[1001]=I\mathbf{A}^{-1}\mathbf{A} = \begin{bmatrix} -.1 & .4 \\ .3 & -.2 \end{bmatrix}\begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \mathbf{I}A−1A=[−.1.3​.4−.2​][23​41​]=[10​01​]=I
or:
AA−1=[2431][−.1.4.3−.2]=[1001]=I\mathbf{AA}^{-1} = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}\begin{bmatrix} -.1 & .4 \\ .3 & -.2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \mathbf{I}AA−1=[23​41​][−.1.3​.4−.2​]=[10​01​]=I
Example 2: The inverse of the matrix:
A3×3=[300040002]\mathbf{A}_{3 \times 3} = \begin{bmatrix} 3 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 2 \end{bmatrix}A3×3​=​300​040​002​​
is:
A3×3−1=[130001400012]\mathbf{A}^{-1}_{3 \times 3} = \begin{bmatrix} \frac{1}{3} & 0 & 0 \\ 0 & \frac{1}{4} & 0 \\ 0 & 0 & \frac{1}{2} \end{bmatrix}A3×3−1​=​31​00​041​0​0021​​​
since:
A−1A=[130001400012][300040002]=[100010001]=I\mathbf{A}^{-1}\mathbf{A} = \begin{bmatrix} \frac{1}{3} & 0 & 0 \\ 0 & \frac{1}{4} & 0 \\ 0 & 0 & \frac{1}{2} \end{bmatrix}\begin{bmatrix} 3 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \mathbf{I}A−1A=​31​00​041​0​0021​​​​300​040​002​​=​100​010​001​​=I
Key Observation: Note that the inverse of a diagonal matrix is a diagonal matrix consisting simply of the reciprocals of the elements on the diagonal.

Finding the Inverse
Up to this point, the inverse of a matrix A has been given, and we have only checked to make sure it is the inverse by seeing whether or not A−1A=I\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}
A−1A=I. But how does one find the inverse, and when does it exist?

Existence of Inverse: An inverse of a square r×rr \times r
r×r matrix exists if the rank of the matrix is rr
r. Such a matrix is said to be
nonsingular or of full rank. An r×rr \times r
r×r matrix with rank less than rr
r is said to be
singular or not of full rank, and does not have an inverse. The inverse of an r×rr \times r
r×r matrix of full rank also has rank rr
r.

Computing Inverses: Finding the inverse of a matrix can often require a large amount of computing. We shall take the approach in this book that the inverse of a 2×22 \times 2
2×2 matrix and a 3×33 \times 3
3×3 matrix can be calculated by hand. For any larger matrix, one ordinarily uses a computer to find the inverse, unless the matrix is of a special form such as a diagonal matrix. It can be shown that the inverses for 2×22 \times 2
2×2 and 3×33 \times 3
3×3 matrices are as follows:


Formula for 2×22 \times 2
2×2 Inverse

1. If:
A2×2=[abcd]\mathbf{A}_{2 \times 2} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}A2×2​=[ac​bd​]
then:
\mathbf{A}^{-1}_{2 \times 2} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \begin{bmatrix} \frac{d}{D} & \frac{-b}{D} \\ \frac{-c}{D} & \frac{a}{D} \end{bmatrix}$
(5.22)
where:
D = ad - bc$
(5.22a)
DD
D is called the
determinant of the matrix A. If A were singular, its determinant would equal zero and no inverse of A would exist.

Formula for 3×33 \times 3
3×3 Inverse

2. If:
B3×3=[abcdefghk]\mathbf{B}_{3 \times 3} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & k \end{bmatrix}B3×3​=​adg​beh​cfk​​
then:
\mathbf{B}^{-1}_{3 \times 3} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & k \end{bmatrix}^{-1} = \begin{bmatrix} A & B & C \\ D & E & F \\ G & H & K \end{bmatrix}$
(5.23)
where:
A=(ek−fh)/ZB=−(bk−ch)/ZC=(bf−ce)/ZA = (ek - fh)/Z \quad\quad B = -(bk - ch)/Z \quad\quad C = (bf - ce)/ZA=(ek−fh)/ZB=−(bk−ch)/ZC=(bf−ce)/Z
D = -(dk - fg)/Z \quad\quad E = (ak - cg)/Z \quad\quad F = -(af - cd)/Z$
(5.23a)
G=(dh−eg)/ZH=−(ah−bg)/ZK=(ae−bd)/ZG = (dh - eg)/Z \quad\quad H = -(ah - bg)/Z \quad\quad K = (ae - bd)/ZG=(dh−eg)/ZH=−(ah−bg)/ZK=(ae−bd)/Z
and:
Z = a(ek - fh) - b(dk - fg) + c(dh - eg)$
(5.23b)
ZZ
Z is called the determinant of the matrix
B.

Example: Finding Inverse of 2×22 \times 2
2×2 Matrix

Let us use (5.22) to find the inverse of:
A=[2431]\mathbf{A} = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}A=[23​41​]
We have:
a=2b=4a = 2 \quad b = 4a=2b=4
c=3d=1c = 3 \quad d = 1c=3d=1
D=ad−bc=2(1)−4(3)=−10D = ad - bc = 2(1) - 4(3) = -10D=ad−bc=2(1)−4(3)=−10
Hence:
A−1=[1−10−4−10−3−102−10]=[−.1.4.3−.2]\mathbf{A}^{-1} = \begin{bmatrix} \frac{1}{-10} & \frac{-4}{-10} \\ \frac{-3}{-10} & \frac{2}{-10} \end{bmatrix} = \begin{bmatrix} -.1 & .4 \\ .3 & -.2 \end{bmatrix}A−1=[−101​−10−3​​−10−4​−102​​]=[−.1.3​.4−.2​]
as was given in an earlier example.
Important Check: When an inverse A−1\mathbf{A}^{-1}
A−1 has been obtained by hand calculations or from a computer program for which the accuracy of inverting a matrix is not known, it may be wise to compute A−1A\mathbf{A}^{-1}\mathbf{A}
A−1A to check whether the product equals the identity matrix, allowing for minor rounding departures from 0 and 1.


Regression Example: Inverse of X′X\mathbf{X'X}
X′X
The principal inverse matrix encountered in regression analysis is the inverse of the matrix X′X\mathbf{X'X}
X′X in (5.14):

X′X2×2=[n∑Xi∑Xi∑Xi2]\mathbf{X'X}_{2 \times 2} = \begin{bmatrix} n & \sum X_i \\ \sum X_i & \sum X_i^2 \end{bmatrix}X′X2×2​=[n∑Xi​​∑Xi​∑Xi2​​]
Using rule (5.22), we have:
a=nb=∑Xia = n \quad\quad b = \sum X_ia=nb=∑Xi​
c=∑Xid=∑Xi2c = \sum X_i \quad\quad d = \sum X_i^2c=∑Xi​d=∑Xi2​
so that:
D=n∑Xi2−(∑Xi)(∑Xi)=n[∑Xi2−(∑Xi)2n]=n∑(Xi−Xˉ)2D = n\sum X_i^2 - \left(\sum X_i\right)\left(\sum X_i\right) = n\left[\sum X_i^2 - \frac{(\sum X_i)^2}{n}\right] = n\sum(X_i - \bar{X})^2D=n∑Xi2​−(∑Xi​)(∑Xi​)=n[∑Xi2​−n(∑Xi​)2​]=n∑(Xi​−Xˉ)2
Hence:
(X′X)2×2−1=[∑Xi2n∑(Xi−Xˉ)2−∑Xin∑(Xi−Xˉ)2−∑Xin∑(Xi−Xˉ)2nn∑(Xi−Xˉ)2](\mathbf{X'X})^{-1}_{2 \times 2} = \begin{bmatrix} \frac{\sum X_i^2}{n\sum(X_i - \bar{X})^2} & \frac{-\sum X_i}{n\sum(X_i - \bar{X})^2} \\ \frac{-\sum X_i}{n\sum(X_i - \bar{X})^2} & \frac{n}{n\sum(X_i - \bar{X})^2} \end{bmatrix}(X′X)2×2−1​=[n∑(Xi​−Xˉ)2∑Xi2​​n∑(Xi​−Xˉ)2−∑Xi​​​n∑(Xi​−Xˉ)2−∑Xi​​n∑(Xi​−Xˉ)2n​​]
Since ∑Xi=nXˉ\sum X_i = n\bar{X}
∑Xi​=nXˉ and ∑(Xi−Xˉ)2=∑Xi2−nXˉ2\sum(X_i - \bar{X})^2 = \sum X_i^2 - n\bar{X}^2
∑(Xi​−Xˉ)2=∑Xi2​−nXˉ2, we can simplify (5.24):

(X′X)2×2−1=[1n+Xˉ2∑(Xi−Xˉ)2−Xˉ∑(Xi−Xˉ)2−Xˉ∑(Xi−Xˉ)21∑(Xi−Xˉ)2](\mathbf{X'X})^{-1}_{2 \times 2} = \begin{bmatrix} \frac{1}{n} + \frac{\bar{X}^2}{\sum(X_i - \bar{X})^2} & \frac{-\bar{X}}{\sum(X_i - \bar{X})^2} \\ \frac{-\bar{X}}{\sum(X_i - \bar{X})^2} & \frac{1}{\sum(X_i - \bar{X})^2} \end{bmatrix}(X′X)2×2−1​=[n1​+∑(Xi​−Xˉ)2Xˉ2​∑(Xi​−Xˉ)2−Xˉ​​∑(Xi​−Xˉ)2−Xˉ​∑(Xi​−Xˉ)21​​]

Uses of Inverse Matrix
In ordinary algebra, we solve an equation of the type:
5y=205y = 205y=20
by multiplying both sides of the equation by the inverse of 5, namely:
15(5y)=15(20)\frac{1}{5}(5y) = \frac{1}{5}(20)51​(5y)=51​(20)
and we obtain the solution:
y=15(20)=4y = \frac{1}{5}(20) = 4y=51​(20)=4
In matrix algebra, if we have an equation:
AY=C\mathbf{AY} = \mathbf{C}AY=C
we correspondingly premultiply both sides by A−1\mathbf{A}^{-1}
A−1, assuming
A has an inverse:
A−1AY=A−1C\mathbf{A}^{-1}\mathbf{AY} = \mathbf{A}^{-1}\mathbf{C}A−1AY=A−1C
Since A−1AY=IY=Y\mathbf{A}^{-1}\mathbf{AY} = \mathbf{IY} = \mathbf{Y}
A−1AY=IY=Y, we obtain the solution:

Y=A−1C\mathbf{Y} = \mathbf{A}^{-1}\mathbf{C}Y=A−1C
To illustrate this use, suppose we have two simultaneous equations:
2y1+4y2=202y_1 + 4y_2 = 202y1​+4y2​=20
3y1+y2=103y_1 + y_2 = 103y1​+y2​=10
which can be written as follows in matrix notation:
[2431][y1y2]=[2010]\begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 20 \\ 10 \end{bmatrix}[23​41​][y1​y2​​]=[2010​]
The solution of these equations then is:
[y1y2]=[2431]−1[2010]\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 3 & 1 \end{bmatrix}^{-1}\begin{bmatrix} 20 \\ 10 \end{bmatrix}[y1​y2​​]=[23​41​]−1[2010​]
Earlier we found the required inverse, so we obtain:
[y1y2]=[−.1.4.3−.2][2010]=[24]\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} -.1 & .4 \\ .3 & -.2 \end{bmatrix}\begin{bmatrix} 20 \\ 10 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}[y1​y2​​]=[−.1.3​.4−.2​][2010​]=[24​]
Hence, y1=2y_1 = 2
y1​=2 and y2=4y_2 = 4
y2​=4 satisfy these two equations.


5.13 Some Basic Results for Matrices
We list here, without proof, some basic results for matrices which we will utilize in later work.
\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$
(5.25)
(\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})$
(5.26)
(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$
(5.27)
\mathbf{C}(\mathbf{A} + \mathbf{B}) = \mathbf{CA} + \mathbf{CB}$
(5.28)
k(\mathbf{A} + \mathbf{B}) = k\mathbf{A} + k\mathbf{B}$
(5.29)
(\mathbf{A})' = \mathbf{A}$
(5.30)
(\mathbf{A} + \mathbf{B})' = \mathbf{A}' + \mathbf{B}'$
(5.31)
(\mathbf{AB})' = \mathbf{B}'\mathbf{A}'$
(5.32)
(\mathbf{ABC})' = \mathbf{C}'\mathbf{B}'\mathbf{A}'$
(5.33)
(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
(5.34)
(\mathbf{ABC})^{-1} = \mathbf{C}^{-1}\mathbf{B}^{-1}\mathbf{A}^{-1}$
(5.35)
(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
(5.36)
(\mathbf{A}')^{-1} = (\mathbf{A}^{-1})'$
(5.37)

5.14 Random Vectors and Matrices
A random vector or a random matrix contains elements that are random variables. Thus, the observations vector Y in (5.4) is a random vector since the YiY_i
Yi​ elements are random variables.

Expectation of Random Vector or Matrix
Suppose we have n=3n = 3
n=3 observations in the observations vector
Y:
Y3×1=[Y1Y2Y3]\mathbf{Y}_{3 \times 1} = \begin{bmatrix} Y_1 \\ Y_2 \\ Y_3 \end{bmatrix}Y3×1​=​Y1​Y2​Y3​​​
The expected value of Y is a vector, denoted by E{Y}\mathbf{E\{Y\}}
E{Y}, that is defined as follows:

E{Y}3×1=[E{Y1}E{Y2}E{Y3}]\mathbf{E\{Y\}}_{3 \times 1} = \begin{bmatrix} E\{Y_1\} \\ E\{Y_2\} \\ E\{Y_3\} \end{bmatrix}E{Y}3×1​=​E{Y1​}E{Y2​}E{Y3​}​​
Thus, the expected value of a random vector is a vector whose elements are the expected values of the random variables that are the elements of the random vector. Similarly, the expectation of a random matrix is a matrix whose elements are the expected values of the corresponding random variables in the original matrix. We encountered a vector of expected values earlier in (5.9).

General Formulas
In general, for a random vector Y the expectation is:
\mathbf{E\{Y\}}_{n \times 1} = [E\{Y_i\}] \quad\quad i = 1, \ldots, n$
(5.38)
and for a random matrix Y with dimension n×pn \times p
n×p, the expectation is:

\mathbf{E\{Y\}}_{n \times p} = [E\{Y_{ij}\}] \quad\quad i = 1, \ldots, n; j = 1, \ldots, p$
**(5.39)**


Regression Example
Suppose the number of cases in a regression application is n=3n = 3
n=3. The three error terms ε1,ε2,ε3\varepsilon_1, \varepsilon_2, \varepsilon_3
ε1​,ε2​,ε3​ each have expectation zero. For the error terms vector:

ε3×1=[ε1ε2ε3]\boldsymbol{\varepsilon}_{3 \times 1} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \varepsilon_3 \end{bmatrix}ε3×1​=​ε1​ε2​ε3​​​
we have:
E{ε}3×1=03×1\mathbf{E\{\boldsymbol{\varepsilon}\}}_{3 \times 1} = \mathbf{0}_{3 \times 1}E{ε}3×1​=03×1​
since:
[E{ε1}E{ε2}E{ε3}]=[000]\begin{bmatrix} E\{\varepsilon_1\} \\ E\{\varepsilon_2\} \\ E\{\varepsilon_3\} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}​E{ε1​}E{ε2​}E{ε3​}​​=​000​​

Variance-Covariance Matrix of Random Vector
Consider again the random vector Y consisting of three observations Y1,Y2,Y3Y_1, Y_2, Y_3
Y1​,Y2​,Y3​. The variances of the three random variables, σ2{Yi}\sigma^2\{Y_i\}
σ2{Yi​}, and the covariances between any two of the random variables, σ{Yi,Yj}\sigma\{Y_i, Y_j\}
σ{Yi​,Yj​}, are assembled in the
variance-covariance matrix of Y, denoted by σ2{Y}\sigma^2\{\mathbf{Y}\}
σ2{Y}, in the following form:

\sigma^2\{\mathbf{Y}\} = \begin{bmatrix} \sigma^2\{Y_1\} & \sigma\{Y_1, Y_2\} & \sigma\{Y_1, Y_3\} \\ \sigma\{Y_2, Y_1\} & \sigma^2\{Y_2\} & \sigma\{Y_2, Y_3\} \\ \sigma\{Y_3, Y_1\} & \sigma\{Y_3, Y_2\} & \sigma^2\{Y_3\} \end{bmatrix}$
(5.40)
Key Points:

The variances are on the main diagonal
The covariance σ{Yi,Yj}\sigma\{Y_i, Y_j\}
σ{Yi​,Yj​} is found in the ii
ith row and jj
jth column of the matrix

Thus, σ{Y2,Y1}\sigma\{Y_2, Y_1\}
σ{Y2​,Y1​} is found in the second row, first column

σ{Y1,Y2}\sigma\{Y_1, Y_2\}
σ{Y1​,Y2​} is found in the first row, second column


Remember that σ{Y2,Y1}=σ{Y1,Y2}\sigma\{Y_2, Y_1\} = \sigma\{Y_1, Y_2\}
σ{Y2​,Y1​}=σ{Y1​,Y2​}. Since σ{Yi,Yj}=σ{Yj,Yi}\sigma\{Y_i, Y_j\} = \sigma\{Y_j, Y_i\}
σ{Yi​,Yj​}=σ{Yj​,Yi​} for all i≠ji \neq j
i=j, σ2{Y}\sigma^2\{\mathbf{Y}\}
σ2{Y} is a symmetric matrix.


Formula for Variance-Covariance Matrix
It follows readily that:
\sigma^2\{\mathbf{Y}\} = \mathbf{E}\{[\mathbf{Y} - \mathbf{E\{Y\}}][\mathbf{Y} - \mathbf{E\{Y\}}]'\}$
(5.41)

Illustration of Formula (5.41)
For our illustration, we have:
σ2{Y}=E{[Y1−E{Y1}Y2−E{Y2}Y3−E{Y3}][Y1−E{Y1}Y2−E{Y2}Y3−E{Y3}]}\sigma^2\{\mathbf{Y}\} = \mathbf{E}\left\{\begin{bmatrix} Y_1 - E\{Y_1\} \\ Y_2 - E\{Y_2\} \\ Y_3 - E\{Y_3\} \end{bmatrix}[Y_1 - E\{Y_1\} \quad Y_2 - E\{Y_2\} \quad Y_3 - E\{Y_3\}]\right\}σ2{Y}=E⎩⎨⎧​​Y1​−E{Y1​}Y2​−E{Y2​}Y3​−E{Y3​}​​[Y1​−E{Y1​}Y2​−E{Y2​}Y3​−E{Y3​}]⎭⎬⎫​
Multiplying the two matrices and then taking expectations, we obtain:
Location in ProductTermExpected ValueRow 1, column 1(Y1−E{Y1})2(Y_1 - E\{Y_1\})^2
(Y1​−E{Y1​})2σ2{Y1}\sigma^2\{Y_1\}
σ2{Y1​}Row 1, column 2(Y1−E{Y1})(Y2−E{Y2})(Y_1 - E\{Y_1\})(Y_2 - E\{Y_2\})
(Y1​−E{Y1​})(Y2​−E{Y2​})σ{Y1,Y2}\sigma\{Y_1, Y_2\}
σ{Y1​,Y2​}Row 1, column 3(Y1−E{Y1})(Y3−E{Y3})(Y_1 - E\{Y_1\})(Y_3 - E\{Y_3\})
(Y1​−E{Y1​})(Y3​−E{Y3​})σ{Y1,Y3}\sigma\{Y_1, Y_3\}
σ{Y1​,Y3​}Row 2, column 1(Y2−E{Y2})(Y1−E{Y1})(Y_2 - E\{Y_2\})(Y_1 - E\{Y_1\})
(Y2​−E{Y2​})(Y1​−E{Y1​})σ{Y2,Y1}\sigma\{Y_2, Y_1\}
σ{Y2​,Y1​}etc.etc.etc.
This, of course, leads to the variance-covariance matrix in (5.40). Remember the definitions of variance and covariance in (A.15) and (A.21), respectively, when taking expectations.

General Variance-Covariance Matrix
To generalize, the variance-covariance matrix for an n×1n \times 1
n×1 random vector
Y is:
\sigma^2\{\mathbf{Y}\}_{n \times n} = \begin{bmatrix} \sigma^2\{Y_1\} & \sigma\{Y_1, Y_2\} & \cdots & \sigma\{Y_1, Y_n\} \\ \sigma\{Y_2, Y_1\} & \sigma^2\{Y_2\} & \cdots & \sigma\{Y_2, Y_n\} \\ \vdots & \vdots &  & \vdots \\ \sigma\{Y_n, Y_1\} & \sigma\{Y_n, Y_2\} & \cdots & \sigma^2\{Y_n\} \end{bmatrix}$
(5.42)
Note again that σ2{Y}\sigma^2\{\mathbf{Y}\}
σ2{Y} is a symmetric matrix.


Regression Example: Variance-Covariance Matrix of Errors
Let us return to the example based on n=3n = 3
n=3 cases. Suppose that the three error terms have constant variance, σ2{εi}=σ2\sigma^2\{\varepsilon_i\} = \sigma^2
σ2{εi​}=σ2, and are uncorrelated so that σ{εi,εj}=0\sigma\{\varepsilon_i, \varepsilon_j\} = 0
σ{εi​,εj​}=0 for i≠ji \neq j
i=j. The variance-covariance matrix for the random vector
ε of the previous example is therefore as follows:
σ2{ε}3×3=[σ2000σ2000σ2]\sigma^2\{\boldsymbol{\varepsilon}\}_{3 \times 3} = \begin{bmatrix} \sigma^2 & 0 & 0 \\ 0 & \sigma^2 & 0 \\ 0 & 0 & \sigma^2 \end{bmatrix}σ2{ε}3×3​=​σ200​0σ20​00σ2​​
Note that all variances are σ2\sigma^2
σ2 and all covariances are zero. Note also that this variance-covariance matrix is a scalar matrix, with the common variance σ2\sigma^2
σ2 the scalar. Hence, we can express the variance-covariance matrix in the following simple fashion:

σ2{ε}3×3=σ2I3×3\sigma^2\{\boldsymbol{\varepsilon}\}_{3 \times 3} = \sigma^2\mathbf{I}_{3 \times 3}σ2{ε}3×3​=σ2I3×3​
since:
σ2I=σ2[100010001]=[σ2000σ2000σ2]\sigma^2\mathbf{I} = \sigma^2\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} \sigma^2 & 0 & 0 \\ 0 & \sigma^2 & 0 \\ 0 & 0 & \sigma^2 \end{bmatrix}σ2I=σ2​100​010​001​​=​σ200​0σ20​00σ2​​

Some Basic Results
Frequently, we shall encounter a random vector W that is obtained by premultiplying the random vector Y by a constant matrix A (a matrix whose elements are fixed):
\mathbf{W} = \mathbf{AY}$
(5.43)
Some basic results for this case are:
\mathbf{E\{A\}} = \mathbf{A}$
(5.44)
\mathbf{E\{W\}} = \mathbf{E\{AY\}} = \mathbf{AE\{Y\}}$
(5.45)
\sigma^2\{\mathbf{W}\} = \sigma^2\{\mathbf{AY}\} = \mathbf{A}\sigma^2\{\mathbf{Y}\}\mathbf{A}'$
(5.46)
where σ2{Y}\sigma^2\{\mathbf{Y}\}
σ2{Y} is the variance-covariance matrix of
Y.

Example of Basic Results
As a simple illustration of the use of these results, consider:
[W1W2]=[1−111][Y1Y2]=[Y1−Y2Y1+Y2]\begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} Y_1 \\ Y_2 \end{bmatrix} = \begin{bmatrix} Y_1 - Y_2 \\ Y_1 + Y_2 \end{bmatrix}[W1​W2​​]=[11​−11​][Y1​Y2​​]=[Y1​−Y2​Y1​+Y2​​]
W2×1⏟WA2×2⏟AY2×1⏟Y\underbrace{\mathbf{W}_{2 \times 1}}_{\mathbf{W}} \quad\quad \underbrace{\mathbf{A}_{2 \times 2}}_{\mathbf{A}} \quad\quad \underbrace{\mathbf{Y}_{2 \times 1}}_{\mathbf{Y}}WW2×1​​​AA2×2​​​YY2×1​​​
We then have by (5.45):
E{W}2×1=[1−111][E{Y1}E{Y2}]=[E{Y1}−E{Y2}E{Y1}+E{Y2}]\mathbf{E\{W\}}_{2 \times 1} = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} E\{Y_1\} \\ E\{Y_2\} \end{bmatrix} = \begin{bmatrix} E\{Y_1\} - E\{Y_2\} \\ E\{Y_1\} + E\{Y_2\} \end{bmatrix}E{W}2×1​=[11​−11​][E{Y1​}E{Y2​}​]=[E{Y1​}−E{Y2​}E{Y1​}+E{Y2​}​]
and by (5.46):
σ2{W}2×2=[1−111][σ2{Y1}σ{Y1,Y2}σ{Y2,Y1}σ2{Y2}][11−11]\sigma^2\{\mathbf{W}\}_{2 \times 2} = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} \sigma^2\{Y_1\} & \sigma\{Y_1, Y_2\} \\ \sigma\{Y_2, Y_1\} & \sigma^2\{Y_2\} \end{bmatrix}\begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}σ2{W}2×2​=[11​−11​][σ2{Y1​}σ{Y2​,Y1​}​σ{Y1​,Y2​}σ2{Y2​}​][1−1​11​]
=[σ2{Y1}+σ2{Y2}−2σ{Y1,Y2}σ2{Y1}−σ2{Y2}σ2{Y1}−σ2{Y2}σ2{Y1}+σ2{Y2}+2σ{Y1,Y2}]= \begin{bmatrix} \sigma^2\{Y_1\} + \sigma^2\{Y_2\} - 2\sigma\{Y_1, Y_2\} & \sigma^2\{Y_1\} - \sigma^2\{Y_2\} \\ \sigma^2\{Y_1\} - \sigma^2\{Y_2\} & \sigma^2\{Y_1\} + \sigma^2\{Y_2\} + 2\sigma\{Y_1, Y_2\} \end{bmatrix}=[σ2{Y1​}+σ2{Y2​}−2σ{Y1​,Y2​}σ2{Y1​}−σ2{Y2​}​σ2{Y1​}−σ2{Y2​}σ2{Y1​}+σ2{Y2​}+2σ{Y1​,Y2​}​]
Thus:
σ2{W1}=σ2(Y1−Y2)=σ2{Y1}+σ2{Y2}−2σ{Y1,Y2}\sigma^2\{W_1\} = \sigma^2(Y_1 - Y_2) = \sigma^2\{Y_1\} + \sigma^2\{Y_2\} - 2\sigma\{Y_1, Y_2\}σ2{W1​}=σ2(Y1​−Y2​)=σ2{Y1​}+σ2{Y2​}−2σ{Y1​,Y2​}
σ2{W2}=σ2(Y1+Y2)=σ2{Y1}+σ2{Y2}+2σ{Y1,Y2}\sigma^2\{W_2\} = \sigma^2(Y_1 + Y_2) = \sigma^2\{Y_1\} + \sigma^2\{Y_2\} + 2\sigma\{Y_1, Y_2\}σ2{W2​}=σ2(Y1​+Y2​)=σ2{Y1​}+σ2{Y2​}+2σ{Y1​,Y2​}
σ{W1,W2}=σ{Y1−Y2,Y1+Y2}=σ2{Y1}−σ2{Y2}\sigma\{W_1, W_2\} = \sigma\{Y_1 - Y_2, Y_1 + Y_2\} = \sigma^2\{Y_1\} - \sigma^2\{Y_2\}σ{W1​,W2​}=σ{Y1​−Y2​,Y1​+Y2​}=σ2{Y1​}−σ2{Y2​}

Multivariate Normal Distribution
Density Function: The density function for the multivariate normal distribution is best given in matrix form. We first need to define some vectors and matrices. The observations vector Y containing an observation on each of the pp
p YY
Y variables is defined as usual:

\mathbf{Y}_{p \times 1} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_p \end{bmatrix}$
(5.47)
The mean vector E{Y}\mathbf{E\{Y\}}
E{Y}, denoted by μ\boldsymbol{\mu}
μ, contains the expected values for each of the pp
p YY
Y variables:

\boldsymbol{\mu}_{p \times 1} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix}$
(5.48)
Finally, the variance-covariance matrix σ2{Y}\sigma^2\{\mathbf{Y}\}
σ2{Y} is denoted by Σ\boldsymbol{\Sigma}
Σ and contains as always the variances and covariances of the pp
p YY
Y variables:

\boldsymbol{\Sigma}_{p \times p} = \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1p} \\ \sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2p} \\ \vdots & \vdots &  & \vdots \\ \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_p^2 \end{bmatrix}$
**(5.49)**

Here, σ12\sigma_1^2
σ12​ denotes the variance of Y1Y_1
Y1​, σ12\sigma_{12}
σ12​ denotes the covariance of Y1Y_1
Y1​ and Y2Y_2
Y2​, and the like.

The density function of the multivariate normal distribution can now be stated as follows:
f(\mathbf{Y}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left[-\frac{1}{2}(\mathbf{Y} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{Y} - \boldsymbol{\mu})\right]$
(5.50)
Here, ∣Σ∣|\boldsymbol{\Sigma}|
∣Σ∣ is the determinant of the variance-covariance matrix Σ\boldsymbol{\Sigma}
Σ. When there are p=2p = 2
p=2 variables, the multivariate normal density function (5.50) simplifies to the bivariate normal density function (2.74).

Key Properties: The multivariate normal density function has properties that correspond to the ones described for the bivariate normal distribution. For instance, if Y1,…,YpY_1, \ldots, Y_p
Y1​,…,Yp​ are jointly normally distributed (i.e., they follow the multivariate normal distribution), the marginal probability distribution of each variable YkY_k
Yk​ is normal, with mean μk\mu_k
μk​ and standard deviation σk\sigma_k
σk​.


5.15 Simple Linear Regression Model in Matrix Terms
We are now ready to develop simple linear regression in matrix terms. Remember again that we will not present any new results, but shall only state in matrix terms the results obtained earlier. We begin with the normal error regression model (2.1):
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i \quad\quad i = 1, \ldots, n$
(5.51)
This implies:
Y1=β0+β1X1+ε1Y_1 = \beta_0 + \beta_1 X_1 + \varepsilon_1Y1​=β0​+β1​X1​+ε1​
Y2=β0+β1X2+ε2Y_2 = \beta_0 + \beta_1 X_2 + \varepsilon_2Y2​=β0​+β1​X2​+ε2​
\vdots$
(5.51a)
Yn=β0+β1Xn+εnY_n = \beta_0 + \beta_1 X_n + \varepsilon_nYn​=β0​+β1​Xn​+εn​
We defined earlier the observations vector Y in (5.4), the X matrix in (5.6), and the ε vector in (5.10). Let us repeat these definitions and also define the β vector of the regression coefficients:
\mathbf{Y}_{n \times 1} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} \quad\quad \mathbf{X}_{n \times 2} = \begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix} \quad\quad \boldsymbol{\beta}_{2 \times 1} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} \quad\quad \boldsymbol{\varepsilon}_{n \times 1} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}$
**(5.52)**

Now we can write (5.51a) in matrix terms compactly as follows:
\mathbf{Y}_{n \times 1} = \mathbf{X}_{n \times 2}\boldsymbol{\beta}_{2 \times 1} + \boldsymbol{\varepsilon}_{n \times 1}$
**(5.53)**

since:
[Y1Y2⋮Yn]=[1X11X2⋮⋮1Xn][β0β1]+[ε1ε2⋮εn]\begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} = \begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix}\begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} + \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}​Y1​Y2​⋮Yn​​​=​11⋮1​X1​X2​⋮Xn​​​[β0​β1​​]+​ε1​ε2​⋮εn​​​
=[β0+β1X1β0+β1X2⋮β0+β1Xn]+[ε1ε2⋮εn]=[β0+β1X1+ε1β0+β1X2+ε2⋮β0+β1Xn+εn]= \begin{bmatrix} \beta_0 + \beta_1 X_1 \\ \beta_0 + \beta_1 X_2 \\ \vdots \\ \beta_0 + \beta_1 X_n \end{bmatrix} + \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix} = \begin{bmatrix} \beta_0 + \beta_1 X_1 + \varepsilon_1 \\ \beta_0 + \beta_1 X_2 + \varepsilon_2 \\ \vdots \\ \beta_0 + \beta_1 X_n + \varepsilon_n \end{bmatrix}=​β0​+β1​X1​β0​+β1​X2​⋮β0​+β1​Xn​​​+​ε1​ε2​⋮εn​​​=​β0​+β1​X1​+ε1​β0​+β1​X2​+ε2​⋮β0​+β1​Xn​+εn​​​
Note that Xβ\mathbf{X}\boldsymbol{\beta}
Xβ is the vector of the expected values of the YiY_i
Yi​ observations since E{Yi}=β0+β1XiE\{Y_i\} = \beta_0 + \beta_1 X_i
E{Yi​}=β0​+β1​Xi​; hence:

\mathbf{E\{Y\}}_{n \times 1} = \mathbf{X}\boldsymbol{\beta}_{n \times 1}$
**(5.54)**

where E{Y}\mathbf{E\{Y\}}
E{Y} is defined in (5.9).

The column of 1s in the X matrix may be viewed as consisting of the constant X0≡1X_0 \equiv 1
X0​≡1 in the alternative regression model (1.5):

Yi=β0X0+β1Xi+εiwhere X0≡1Y_i = \beta_0 X_0 + \beta_1 X_i + \varepsilon_i \quad\text{where } X_0 \equiv 1Yi​=β0​X0​+β1​Xi​+εi​where X0​≡1
Thus, the X matrix may be considered to contain a column vector consisting of 1s and another column vector consisting of the predictor variable observations XiX_i
Xi​.

With respect to the error terms, regression model (2.1) assumes that E{εi}=0,σ2{εi}=σ2E\{\varepsilon_i\} = 0, \sigma^2\{\varepsilon_i\} = \sigma^2
E{εi​}=0,σ2{εi​}=σ2, and that the εi\varepsilon_i
εi​ are independent normal random variables. The condition E{εi}=0E\{\varepsilon_i\} = 0
E{εi​}=0 in matrix terms is:

\mathbf{E\{\boldsymbol{\varepsilon}\}}_{n \times 1} = \mathbf{0}_{n \times 1}$
**(5.55)**

since (5.55) states:
[E{ε1}E{ε2}⋮E{εn}]=[00⋮0]\begin{bmatrix} E\{\varepsilon_1\} \\ E\{\varepsilon_2\} \\ \vdots \\ E\{\varepsilon_n\} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}​E{ε1​}E{ε2​}⋮E{εn​}​​=​00⋮0​​
The condition that the error terms have constant variance σ2\sigma^2
σ2 and that all covariances σ{εi,εj}\sigma\{\varepsilon_i, \varepsilon_j\}
σ{εi​,εj​} for i≠ji \neq j
i=j are zero (since the εi\varepsilon_i
εi​ are independent) is expressed in matrix terms through the variance-covariance matrix of the error terms:

\sigma^2\{\boldsymbol{\varepsilon}\}_{n \times n} = \begin{bmatrix} \sigma^2 & 0 & 0 & \cdots & 0 \\ 0 & \sigma^2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots &  & \vdots \\ 0 & 0 & 0 & \cdots & \sigma^2 \end{bmatrix}$
(5.56)
Since this is a scalar matrix, we know from the earlier example that it can be expressed in the following simple fashion:
\sigma^2\{\boldsymbol{\varepsilon}\}_{n \times n} = \sigma^2\mathbf{I}_{n \times n}$
**(5.56a)**

Thus, the normal error regression model (2.1) in matrix terms is:
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$
(5.57)
where:
ε is a vector of independent normal random variables with E{ε}=0\mathbf{E\{\boldsymbol{\varepsilon}\}} = \mathbf{0}
E{ε}=0 and σ2{ε}=σ2I\sigma^2\{\boldsymbol{\varepsilon}\} = \sigma^2\mathbf{I}
σ2{ε}=σ2I

5.16 Least Squares Estimation of Regression Parameters
Normal Equations
The normal equations (1.9):
nb0+b1∑Xi=∑Yinb_0 + b_1 \sum X_i = \sum Y_inb0​+b1​∑Xi​=∑Yi​
b_0 \sum X_i + b_1 \sum X_i^2 = \sum X_i Y_i$
(5.58)
in matrix terms are:
\mathbf{X'X}_{2 \times 2}\mathbf{b}_{2 \times 1} = \mathbf{X'Y}_{2 \times 1}$
**(5.59)**

where b is the vector of the least squares regression coefficients:
\mathbf{b}_{2 \times 1} = \begin{bmatrix} b_0 \\ b_1 \end{bmatrix}$
(5.59a)
To see this, recall that we obtained X′X\mathbf{X'X}
X′X in (5.14) and X′Y\mathbf{X'Y}
X′Y in (5.15). Equation (5.59) thus states:

[n∑Xi∑Xi∑Xi2][b0b1]=[∑Yi∑XiYi]\begin{bmatrix} n & \sum X_i \\ \sum X_i & \sum X_i^2 \end{bmatrix}\begin{bmatrix} b_0 \\ b_1 \end{bmatrix} = \begin{bmatrix} \sum Y_i \\ \sum X_i Y_i \end{bmatrix}[n∑Xi​​∑Xi​∑Xi2​​][b0​b1​​]=[∑Yi​∑Xi​Yi​​]
or:
[nb0+b1∑Xib0∑Xi+b1∑Xi2]=[∑Yi∑XiYi]\begin{bmatrix} nb_0 + b_1 \sum X_i \\ b_0 \sum X_i + b_1 \sum X_i^2 \end{bmatrix} = \begin{bmatrix} \sum Y_i \\ \sum X_i Y_i \end{bmatrix}[nb0​+b1​∑Xi​b0​∑Xi​+b1​∑Xi2​​]=[∑Yi​∑Xi​Yi​​]
These are precisely the normal equations in (5.58).

Estimated Regression Coefficients
To obtain the estimated regression coefficients from the normal equations (5.59) by matrix methods, we premultiply both sides by the inverse of X′X\mathbf{X'X}
X′X (we assume this exists):

(X′X)−1X′Xb=(X′X)−1X′Y(\mathbf{X'X})^{-1}\mathbf{X'Xb} = (\mathbf{X'X})^{-1}\mathbf{X'Y}(X′X)−1X′Xb=(X′X)−1X′Y
We then find, since (X′X)−1X′X=I(\mathbf{X'X})^{-1}\mathbf{X'X} = \mathbf{I}
(X′X)−1X′X=I and Ib=b\mathbf{Ib} = \mathbf{b}
Ib=b:

\mathbf{b}_{2 \times 1} = (\mathbf{X'X})^{-1}_{2 \times 2}\mathbf{X'Y}_{2 \times 1}$
**(5.60)**

The estimators b0b_0
b0​ and b1b_1
b1​ in
b are the same as those given earlier in (1.10a) and (1.10b). We shall demonstrate this by an example.

📊 Example 5.1: Toluca Company (Matrix Methods)
We shall use matrix methods to obtain the estimated regression coefficients for the Toluca Company example. The data on the YY
Y and XX
X variables were given in Table 1.1. Using these data, we define the
Y observations vector and the X matrix as follows:
(5.61a)
Y=[399121⋮323]\mathbf{Y} = \begin{bmatrix} 399 \\ 121 \\ \vdots \\ 323 \end{bmatrix}Y=​399121⋮323​​
(5.61b)
\mathbf{X} = \begin{bmatrix} 1 & 80 \\ 1 & 30 \\ \vdots & \vdots \\ 1 & 70 \end{bmatrix}$
(5.61)
We now require the following matrix products:
\mathbf{X'X} = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 80 & 30 & \cdots & 70 \end{bmatrix}\begin{bmatrix} 1 & 80 \\ 1 & 30 \\ \vdots & \vdots \\ 1 & 70 \end{bmatrix} = \begin{bmatrix} 25 & 1,750 \\ 1,750 & 142,300 \end{bmatrix}$
(5.62)
\mathbf{X'Y} = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 80 & 30 & \cdots & 70 \end{bmatrix}\begin{bmatrix} 399 \\ 121 \\ \vdots \\ 323 \end{bmatrix} = \begin{bmatrix} 7,807 \\ 617,180 \end{bmatrix}$
(5.63)
Using (5.22), we find the inverse of X′X\mathbf{X'X}
X′X:

(\mathbf{X'X})^{-1} = \begin{bmatrix} .287475 & -.003535 \\ -.003535 & .00005051 \end{bmatrix}$
(5.64)
In subsequent matrix calculations utilizing this inverse matrix and other matrix results, we shall actually utilize more digits for the matrix elements than are shown.
Finally, we employ (5.60) to obtain:
b=[b0b1]=(X′X)−1X′Y=[.287475−.003535−.003535.00005051][7,807617,180]\mathbf{b} = \begin{bmatrix} b_0 \\ b_1 \end{bmatrix} = (\mathbf{X'X})^{-1}\mathbf{X'Y} = \begin{bmatrix} .287475 & -.003535 \\ -.003535 & .00005051 \end{bmatrix}\begin{bmatrix} 7,807 \\ 617,180 \end{bmatrix}b=[b0​b1​​]=(X′X)−1X′Y=[.287475−.003535​−.003535.00005051​][7,807617,180​]
= \begin{bmatrix} 62.37 \\ 3.5702 \end{bmatrix}$
(5.65)
or b0=62.37b_0 = 62.37
b0​=62.37 and b1=3.5702b_1 = 3.5702
b1​=3.5702. These results agree with the ones in Chapter 1. Any differences would have been due to rounding effects.


Comments
1. Deriving Normal Equations by Matrix Methods
To derive the normal equations by the method of least squares, we minimize the quantity:
Q=∑[Yi−(β0+β1Xi)]2Q = \sum[Y_i - (\beta_0 + \beta_1 X_i)]^2Q=∑[Yi​−(β0​+β1​Xi​)]2
In matrix notation:
Q = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$
(5.66)
Expanding, we obtain:
Q=Y′Y−β′X′Y−Y′Xβ+β′X′XβQ = \mathbf{Y'Y} - \boldsymbol{\beta}'\mathbf{X'Y} - \mathbf{Y'X}\boldsymbol{\beta} + \boldsymbol{\beta}'\mathbf{X'X}\boldsymbol{\beta}Q=Y′Y−β′X′Y−Y′Xβ+β′X′Xβ
since (Xβ)′=β′X′(\mathbf{X}\boldsymbol{\beta})' = \boldsymbol{\beta}'\mathbf{X'}
(Xβ)′=β′X′ by (5.32). Note now that Y′Xβ\mathbf{Y'X}\boldsymbol{\beta}
Y′Xβ is 1×11 \times 1
1×1, hence is equal to its transpose, which according to (5.33) is β′X′Y\boldsymbol{\beta}'\mathbf{X'Y}
β′X′Y. Thus, we find:

Q = \mathbf{Y'Y} - 2\boldsymbol{\beta}'\mathbf{X'Y} + \boldsymbol{\beta}'\mathbf{X'X}\boldsymbol{\beta}$
(5.67)
To find the value of β that minimizes QQ
Q, we differentiate with respect to β0\beta_0
β0​ and β1\beta_1
β1​. Let:

\frac{\partial}{\partial\boldsymbol{\beta}}(Q) = \begin{bmatrix} \frac{\partial Q}{\partial\beta_0} \\ \frac{\partial Q}{\partial\beta_1} \end{bmatrix}$
(5.68)
Then it follows that:
\frac{\partial}{\partial\boldsymbol{\beta}}(Q) = -2\mathbf{X'Y} + 2\mathbf{X'X}\boldsymbol{\beta}$
(5.69)
Equating to the zero vector, dividing by 2, and substituting b for β gives the matrix form of the least squares normal equations in (5.59).
2. Uniqueness of Solutions
A comparison of the normal equations and X′X\mathbf{X'X}
X′X shows that whenever the columns of X′X\mathbf{X'X}
X′X are linearly dependent, the normal equations will be linearly dependent also. No unique solutions can then be obtained for b0b_0
b0​ and b1b_1
b1​. Fortunately, in most regression applications, the columns of X′X\mathbf{X'X}
X′X are linearly independent, leading to unique solutions for b0b_0
b0​ and b1b_1
b1​. ■


5.17 Fitted Values and Residuals
Fitted Values
Let the vector of the fitted values Y^i\hat{Y}_i
Y^i​ be denoted by Y^\hat{\mathbf{Y}}
Y^:

\hat{\mathbf{Y}}_{n \times 1} = \begin{bmatrix} \hat{Y}_1 \\ \hat{Y}_2 \\ \vdots \\ \hat{Y}_n \end{bmatrix}$
(5.70)
In matrix notation, we then have:
\hat{\mathbf{Y}}_{n \times 1} = \mathbf{X}_{n \times 2}\mathbf{b}_{2 \times 1}$
**(5.71)**

because:
[Y^1Y^2⋮Y^n]=[1X11X2⋮⋮1Xn][b0b1]=[b0+b1X1b0+b1X2⋮b0+b1Xn]\begin{bmatrix} \hat{Y}_1 \\ \hat{Y}_2 \\ \vdots \\ \hat{Y}_n \end{bmatrix} = \begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix}\begin{bmatrix} b_0 \\ b_1 \end{bmatrix} = \begin{bmatrix} b_0 + b_1 X_1 \\ b_0 + b_1 X_2 \\ \vdots \\ b_0 + b_1 X_n \end{bmatrix}​Y^1​Y^2​⋮Y^n​​​=​11⋮1​X1​X2​⋮Xn​​​[b0​b1​​]=​b0​+b1​X1​b0​+b1​X2​⋮b0​+b1​Xn​​​

Example: Fitted Values for Toluca Company
For the Toluca Company example, we obtain the vector of fitted values using the matrices in (5.61b) and (5.65):
\hat{\mathbf{Y}} = \mathbf{Xb} = \begin{bmatrix} 1 & 80 \\ 1 & 30 \\ \vdots & \vdots \\ 1 & 70 \end{bmatrix}\begin{bmatrix} 62.37 \\ 3.5702 \end{bmatrix} = \begin{bmatrix} 347.98 \\ 169.47 \\ \vdots \\ 312.28 \end{bmatrix}$
(5.72)
The fitted values are the same, of course, as in Table 1.2.

Hat Matrix
We can express the matrix result for Y^\hat{\mathbf{Y}}
Y^ in (5.71) as follows by using the expression for
b in (5.60):
Y^=X(X′X)−1X′Y\hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X'X})^{-1}\mathbf{X'Y}Y^=X(X′X)−1X′Y
or, equivalently:
\hat{\mathbf{Y}}_{n \times 1} = \mathbf{H}_{n \times n}\mathbf{Y}_{n \times 1}$
**(5.73)**

where:
\mathbf{H}_{n \times n} = \mathbf{X}(\mathbf{X'X})^{-1}\mathbf{X'}$
(5.73a)
We see from (5.73) that the fitted values Y^i\hat{Y}_i
Y^i​ can be expressed as linear combinations of the response variable observations YiY_i
Yi​, with the coefficients being elements of the matrix
H. The H matrix involves only the observations on the predictor variable XX
X, as is evident from (5.73a).

Key Property: The square n×nn \times n
n×n matrix
H is called the hat matrix. It plays an important role in diagnostics for regression analysis, as we shall see in Chapter 10 when we consider whether regression results are unduly influenced by one or a few observations. The matrix H is symmetric and has the special property (called idempotency):
\mathbf{HH} = \mathbf{H}$
(5.74)
In general, a matrix M is said to be idempotent if MM=M\mathbf{MM} = \mathbf{M}
MM=M.


Residuals
Let the vector of the residuals ei=Yi−Y^ie_i = Y_i - \hat{Y}_i
ei​=Yi​−Y^i​ be denoted by
e:
\mathbf{e}_{n \times 1} = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_n \end{bmatrix}$
(5.75)
In matrix notation, we then have:
\mathbf{e}_{n \times 1} = \mathbf{Y}_{n \times 1} - \hat{\mathbf{Y}}_{n \times 1} = \mathbf{Y}_{n \times 1} - \mathbf{Xb}_{n \times 1}$
**(5.76)**


Example: Residuals for Toluca Company
For the Toluca Company example, we obtain the vector of the residuals by using the results in (5.61a) and (5.72):
\mathbf{e} = \begin{bmatrix} 399 \\ 121 \\ \vdots \\ 323 \end{bmatrix} - \begin{bmatrix} 347.98 \\ 169.47 \\ \vdots \\ 312.28 \end{bmatrix} = \begin{bmatrix} 51.02 \\ -48.47 \\ \vdots \\ 10.72 \end{bmatrix}$
(5.77)
The residuals are the same as in Table 1.2.

Variance-Covariance Matrix of Residuals
The residuals eie_i
ei​, like the fitted values Y^i\hat{Y}_i
Y^i​, can be expressed as linear combinations of the response variable observations YiY_i
Yi​, using the result in (5.73) for Y^\hat{\mathbf{Y}}
Y^:

e=Y−Y^=Y−HY=(I−H)Y\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}} = \mathbf{Y} - \mathbf{HY} = (\mathbf{I} - \mathbf{H})\mathbf{Y}e=Y−Y^=Y−HY=(I−H)Y
We thus have the important result:
\mathbf{e}_{n \times 1} = (\mathbf{I} - \mathbf{H})_{n \times n}\mathbf{Y}_{n \times 1}$
**(5.78)**

where H is the hat matrix defined in (5.53a). The matrix I−H\mathbf{I} - \mathbf{H}
I−H, like the matrix
H, is symmetric and idempotent.
The variance-covariance matrix of the vector of residuals e involves the matrix I−H\mathbf{I} - \mathbf{H}
I−H:

\sigma^2\{\mathbf{e}\}_{n \times n} = \sigma^2(\mathbf{I} - \mathbf{H})$
(5.79)
and is estimated by:
s^2\{\mathbf{e}\}_{n \times n} = MSE(\mathbf{I} - \mathbf{H})$
(5.80)

Comment
The variance-covariance matrix of e in (5.79) can be derived by means of (5.46). Since e=(I−H)Y\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{Y}
e=(I−H)Y, we obtain:

σ2{e}=(I−H)σ2{Y}(I−H)′\sigma^2\{\mathbf{e}\} = (\mathbf{I} - \mathbf{H})\sigma^2\{\mathbf{Y}\}(\mathbf{I} - \mathbf{H})'σ2{e}=(I−H)σ2{Y}(I−H)′
Now σ2{Y}=σ2{ε}=σ2I\sigma^2\{\mathbf{Y}\} = \sigma^2\{\boldsymbol{\varepsilon}\} = \sigma^2\mathbf{I}
σ2{Y}=σ2{ε}=σ2I for the normal error model according to (5.56a). Also, (I−H)′=I−H(\mathbf{I} - \mathbf{H})' = \mathbf{I} - \mathbf{H}
(I−H)′=I−H because of the symmetry of the matrix. Hence:

σ2{e}=σ2(I−H)I(I−H)\sigma^2\{\mathbf{e}\} = \sigma^2(\mathbf{I} - \mathbf{H})\mathbf{I}(\mathbf{I} - \mathbf{H})σ2{e}=σ2(I−H)I(I−H)
=σ2(I−H)(I−H)= \sigma^2(\mathbf{I} - \mathbf{H})(\mathbf{I} - \mathbf{H})=σ2(I−H)(I−H)
In view of the fact that the matrix I−H\mathbf{I} - \mathbf{H}
I−H is idempotent, we know that (I−H)(I−H)=I−H(\mathbf{I} - \mathbf{H})(\mathbf{I} - \mathbf{H}) = \mathbf{I} - \mathbf{H}
(I−H)(I−H)=I−H and we obtain formula (5.79). ■


5.18 Analysis of Variance Results
Sums of Squares
To see how the sums of squares are expressed in matrix notation, we begin with the total sum of squares SSTOSSTO
SSTO, defined in (2.43). It will be convenient to use an algebraically equivalent expression:

SSTO = \sum(Y_i - \bar{Y})^2 = \sum Y_i^2 - \frac{(\sum Y_i)^2}{n}$
(5.81)
We know from (5.13) that:
Y′Y=∑Yi2\mathbf{Y'Y} = \sum Y_i^2Y′Y=∑Yi2​
The subtraction term (∑Yi)2/n(\sum Y_i)^2/n
(∑Yi​)2/n in matrix form uses
J, the matrix of 1s defined in (5.18), as follows:
\frac{(\sum Y_i)^2}{n} = \left(\frac{1}{n}\right)\mathbf{Y'JY}$
(5.82)
For instance, if n=2n = 2
n=2, we have:

(12)[Y1Y2][1111][Y1Y2]=(Y1+Y2)(Y1+Y2)2\left(\frac{1}{2}\right)[Y_1 \quad Y_2]\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} Y_1 \\ Y_2 \end{bmatrix} = \frac{(Y_1 + Y_2)(Y_1 + Y_2)}{2}(21​)[Y1​Y2​][11​11​][Y1​Y2​​]=2(Y1​+Y2​)(Y1​+Y2​)​
Hence, it follows that:
SSTO = \mathbf{Y'Y} - \left(\frac{1}{n}\right)\mathbf{Y'JY}$
(5.83)
Just as ∑Yi2\sum Y_i^2
∑Yi2​ is represented by Y′Y\mathbf{Y'Y}
Y′Y in matrix terms, so SSE=∑ei2=∑(Yi−Y^i)2SSE = \sum e_i^2 = \sum(Y_i - \hat{Y}_i)^2
SSE=∑ei2​=∑(Yi​−Y^i​)2 can be represented as follows:

SSE = \mathbf{e'e} = (\mathbf{Y} - \mathbf{Xb})'(\mathbf{Y} - \mathbf{Xb})$
(5.84)
which can be shown to equal:
SSE = \mathbf{Y'Y} - \mathbf{b'X'Y}$
(5.84a)
Finally, it can be shown that:
SSR = \mathbf{b'X'Y} - \left(\frac{1}{n}\right)\mathbf{Y'JY}$
(5.85)

Example: SSE for Toluca Company
Let us find SSESSE
SSE for the Toluca Company example by matrix methods, using (5.84a). Using (5.61a), we obtain:

Y′Y=[399121⋯323][399121⋮323]=2,745,173\mathbf{Y'Y} = [399 \quad 121 \quad \cdots \quad 323]\begin{bmatrix} 399 \\ 121 \\ \vdots \\ 323 \end{bmatrix} = 2,745,173Y′Y=[399121⋯323]​399121⋮323​​=2,745,173
and using (5.65) and (5.63), we find:
b′X′Y=[62.373.5702][7,807617,180]=2,690,348\mathbf{b'X'Y} = [62.37 \quad 3.5702]\begin{bmatrix} 7,807 \\ 617,180 \end{bmatrix} = 2,690,348b′X′Y=[62.373.5702][7,807617,180​]=2,690,348
Hence:
SSE=Y′Y−b′X′Y=2,745,173−2,690,348=54,825SSE = \mathbf{Y'Y} - \mathbf{b'X'Y} = 2,745,173 - 2,690,348 = 54,825SSE=Y′Y−b′X′Y=2,745,173−2,690,348=54,825
which is the same result as that obtained in Chapter 1. Any difference would have been due to rounding effects.

Comment
To illustrate the derivation of the sums of squares expressions in matrix notation, consider SSESSE
SSE:

SSE=e′e=(Y−Xb)′(Y−Xb)=Y′Y−2b′X′Y+b′X′XbSSE = \mathbf{e'e} = (\mathbf{Y} - \mathbf{Xb})'(\mathbf{Y} - \mathbf{Xb}) = \mathbf{Y'Y} - 2\mathbf{b'X'Y} + \mathbf{b'X'Xb}SSE=e′e=(Y−Xb)′(Y−Xb)=Y′Y−2b′X′Y+b′X′Xb
In substituting for the rightmost b we obtain by (5.60):
SSE=Y′Y−2b′X′Y+b′X′X(X′X)−1X′YSSE = \mathbf{Y'Y} - 2\mathbf{b'X'Y} + \mathbf{b'X'X}(\mathbf{X'X})^{-1}\mathbf{X'Y}SSE=Y′Y−2b′X′Y+b′X′X(X′X)−1X′Y
=Y′Y−2b′X′Y+b′IX′Y= \mathbf{Y'Y} - 2\mathbf{b'X'Y} + \mathbf{b'IX'Y}=Y′Y−2b′X′Y+b′IX′Y
In dropping I and subtracting, we obtain the result in (5.84a). ■

Sums of Squares as Quadratic Forms
The ANOVA sums of squares can be shown to be quadratic forms. An example of a quadratic form of the observations YiY_i
Yi​ when n=2n = 2
n=2 is:

5Y_1^2 + 6Y_1Y_2 + 4Y_2^2$
(5.86)
Note that this expression is a second-degree polynomial containing terms involving the squares of the observations and the cross product. We can express (5.86) in matrix terms as follows:
[Y_1 \quad Y_2]\begin{bmatrix} 5 & 3 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} Y_1 \\ Y_2 \end{bmatrix} = \mathbf{Y'AY}$
(5.86a)
where A is a symmetric matrix.

General Quadratic Form
In general, a quadratic form is defined as:
\mathbf{Y'AY} = \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij}Y_iY_j \quad\text{where } a_{ij} = a_{ji}$
(5.87)
A is a symmetric n×nn \times n
n×n matrix and is called the
matrix of the quadratic form.
Key Result: The ANOVA sums of squares SSTOSSTO
SSTO, SSESSE
SSE, and SSRSSR
SSR are all quadratic forms, as can be seen by reexpressing b′X′\mathbf{b'X'}
b′X′. From (5.71), we know, using (5.32), that:

b′X′=(Xb)′=Y^′\mathbf{b'X'} = (\mathbf{Xb})' = \hat{\mathbf{Y}}'b′X′=(Xb)′=Y^′
We now use the result in (5.73) to obtain:
b′X′=(HY)′\mathbf{b'X'} = (\mathbf{HY})'b′X′=(HY)′
Since H is a symmetric matrix so that H′=H\mathbf{H}' = \mathbf{H}
H′=H, we finally obtain, using (5.32):

\mathbf{b'X'} = \mathbf{Y'H}$
(5.88)
This result enables us to express the ANOVA sums of squares as follows:
SSTO = \mathbf{Y}'\left[\mathbf{I} - \left(\frac{1}{n}\right)\mathbf{J}\right]\mathbf{Y}$
(5.89a)
SSE = \mathbf{Y'}(\mathbf{I} - \mathbf{H})\mathbf{Y}$
(5.89b)
SSR = \mathbf{Y}'\left[\mathbf{H} - \left(\frac{1}{n}\right)\mathbf{J}\right]\mathbf{Y}$
(5.89c)
Each of these sums of squares can now be seen to be of the form Y′AY\mathbf{Y'AY}
Y′AY, where the three
A matrices are:
\mathbf{I} - \left(\frac{1}{n}\right)\mathbf{J}$
(5.90a)
\mathbf{I} - \mathbf{H}$
(5.90b)
\mathbf{H} - \left(\frac{1}{n}\right)\mathbf{J}$
(5.90c)
Since each of these A matrices is symmetric, SSTOSSTO
SSTO, SSESSE
SSE, and SSRSSR
SSR are quadratic forms, with the matrices of the quadratic forms given in (5.90). Quadratic forms play an important role in statistics because all sums of squares in the analysis of variance for linear statistical models can be expressed as quadratic forms.


5.19 Inferences in Regression Analysis
As we saw in earlier chapters, all interval estimates are of the following form: point estimator plus and minus a certain number of estimated standard deviations of the point estimator. Similarly, all tests require the point estimator and the estimated standard deviation of the point estimator or, in the case of analysis of variance tests, various sums of squares.
Matrix algebra is of principal help in inference making when obtaining the estimated standard deviations and sums of squares. We have already given the matrix equivalents of the sums of squares for the analysis of variance. We focus here chiefly on the matrix expressions for the estimated variances of point estimators of interest.

Regression Coefficients
The variance-covariance matrix of b:
\sigma^2\{\mathbf{b}\}_{2 \times 2} = \begin{bmatrix} \sigma^2\{b_0\} & \sigma\{b_0, b_1\} \\ \sigma\{b_1, b_0\} & \sigma^2\{b_1\} \end{bmatrix}$
(5.91)
is:
\sigma^2\{\mathbf{b}\}_{2 \times 2} = \sigma^2(\mathbf{X'X})^{-1}$
(5.92)
or, from (5.24a):
\sigma^2\{\mathbf{b}\}_{2 \times 2} = \begin{bmatrix} \frac{\sigma^2}{n} + \frac{\sigma^2\bar{X}^2}{\sum(X_i - \bar{X})^2} & \frac{-\bar{X}\sigma^2}{\sum(X_i - \bar{X})^2} \\ \frac{-\bar{X}\sigma^2}{\sum(X_i - \bar{X})^2} & \frac{\sigma^2}{\sum(X_i - \bar{X})^2} \end{bmatrix}$
(5.92a)
When MSEMSE
MSE is substituted for σ2\sigma^2
σ2 in (5.92a), we obtain the estimated variance-covariance matrix of
b, denoted by s2{b}s^2\{\mathbf{b}\}
s2{b}:

s^2\{\mathbf{b}\}_{2 \times 2} = MSE(\mathbf{X'X})^{-1} = \begin{bmatrix} \frac{MSE}{n} + \frac{\bar{X}^2MSE}{\sum(X_i - \bar{X})^2} & \frac{-\bar{X}MSE}{\sum(X_i - \bar{X})^2} \\ \frac{-\bar{X}MSE}{\sum(X_i - \bar{X})^2} & \frac{MSE}{\sum(X_i - \bar{X})^2} \end{bmatrix}$
(5.93)
In (5.92a), you will recognize the variances of b0b_0
b0​ in (2.22b) and of b1b_1
b1​ in (2.3b) and the covariance of b0b_0
b0​ and b1b_1
b1​ in (4.5). Likewise, the estimated variances in (5.93) are familiar from earlier chapters.


Example: Variance-Covariance Matrix for Toluca Company
We wish to find s2{b0}s^2\{b_0\}
s2{b0​} and s2{b1}s^2\{b_1\}
s2{b1​} for the Toluca Company example by matrix methods. Using the results in Figure 2.2 and in (5.64), we obtain:

s2{b}=MSE(X′X)−1=2,384[.287475−.003535−.003535.00005051]s^2\{\mathbf{b}\} = MSE(\mathbf{X'X})^{-1} = 2,384\begin{bmatrix} .287475 & -.003535 \\ -.003535 & .00005051 \end{bmatrix}s2{b}=MSE(X′X)−1=2,384[.287475−.003535​−.003535.00005051​]
= \begin{bmatrix} 685.34 & -8.428 \\ -8.428 & .12040 \end{bmatrix}$
(5.94)
Thus, s2{b0}=685.34s^2\{b_0\} = 685.34
s2{b0​}=685.34 and s2{b1}=.12040s^2\{b_1\} = .12040
s2{b1​}=.12040. These are the same as the results obtained in Chapter 2.


Comment
To derive the variance-covariance matrix of b, recall that:
b=(X′X)−1X′Y=AY\mathbf{b} = (\mathbf{X'X})^{-1}\mathbf{X'Y} = \mathbf{AY}b=(X′X)−1X′Y=AY
where A is a constant matrix:
A=(X′X)−1X′\mathbf{A} = (\mathbf{X'X})^{-1}\mathbf{X'}A=(X′X)−1X′
Hence, by (5.46) we have:
σ2{b}=Aσ2{Y}A′\sigma^2\{\mathbf{b}\} = \mathbf{A}\sigma^2\{\mathbf{Y}\}\mathbf{A}'σ2{b}=Aσ2{Y}A′
Now σ2{Y}=σ2I\sigma^2\{\mathbf{Y}\} = \sigma^2\mathbf{I}
σ2{Y}=σ2I. Further, it follows from (5.32) and the fact that (X′X)−1(\mathbf{X'X})^{-1}
(X′X)−1 is symmetric that:

A′=X(X′X)−1\mathbf{A}' = \mathbf{X}(\mathbf{X'X})^{-1}A′=X(X′X)−1
We find therefore:
σ2{b}=(X′X)−1X′σ2IX(X′X)−1\sigma^2\{\mathbf{b}\} = (\mathbf{X'X})^{-1}\mathbf{X}'\sigma^2\mathbf{IX}(\mathbf{X'X})^{-1}σ2{b}=(X′X)−1X′σ2IX(X′X)−1
=σ2(X′X)−1X′X(X′X)−1= \sigma^2(\mathbf{X'X})^{-1}\mathbf{X'X}(\mathbf{X'X})^{-1}=σ2(X′X)−1X′X(X′X)−1
=σ2(X′X)−1I= \sigma^2(\mathbf{X'X})^{-1}\mathbf{I}=σ2(X′X)−1I

Mean Response
To estimate the mean response at XhX_h
Xh​, let us define the vector:

\mathbf{X}_h = \begin{bmatrix} 1 \\ X_h \end{bmatrix}_{2 \times 1} \quad\text{or}\quad \mathbf{X}'_h = [1 \quad X_h]_{1 \times 2}$
**(5.95)**

The fitted value in matrix notation then is:
\hat{Y}_h = \mathbf{X}'_h\mathbf{b}$
(5.96)
since:
Xh′b=[1Xh][b0b1]=[b0+b1Xh]=[Y^h]=Y^h\mathbf{X}'_h\mathbf{b} = [1 \quad X_h]\begin{bmatrix} b_0 \\ b_1 \end{bmatrix} = [b_0 + b_1 X_h] = [\hat{Y}_h] = \hat{Y}_hXh′​b=[1Xh​][b0​b1​​]=[b0​+b1​Xh​]=[Y^h​]=Y^h​
Note that Xh′b\mathbf{X}'_h\mathbf{b}
Xh′​b is a 1×11 \times 1
1×1 matrix; hence, we can write the final result as a scalar.

The variance of Y^h\hat{Y}_h
Y^h​, given earlier in (2.29b), in matrix notation is:

\sigma^2\{\hat{Y}_h\} = \sigma^2\mathbf{X}'_h(\mathbf{X'X})^{-1}\mathbf{X}_h$
(5.97)
The variance of Y^h\hat{Y}_h
Y^h​ in (5.93) can be expressed as a function of σ2{b}\sigma^2\{\mathbf{b}\}
σ2{b}, the variance-covariance matrix of the estimated regression coefficients, by making use of the result in (5.92):

\sigma^2\{\hat{Y}_h\} = \mathbf{X}'_h\sigma^2\{\mathbf{b}\}\mathbf{X}_h$
(5.97a)
The estimated variance of Y^h\hat{Y}_h
Y^h​, given earlier in (2.30), in matrix notation is:

s^2\{\hat{Y}_h\} = MSE(\mathbf{X}'_h(\mathbf{X'X})^{-1}\mathbf{X}_h)$
(5.98)

Example: Variance of Mean Response for Toluca Company
We wish to find s2{Y^h}s^2\{\hat{Y}_h\}
s2{Y^h​} for the Toluca Company example when Xh=65X_h = 65
Xh​=65. We define:

Xh′=[165]\mathbf{X}'_h = [1 \quad 65]Xh′​=[165]
and use the result in (5.94) to obtain:
s2{Y^h}=Xh′s2{b}Xhs^2\{\hat{Y}_h\} = \mathbf{X}'_hs^2\{\mathbf{b}\}\mathbf{X}_hs2{Y^h​}=Xh′​s2{b}Xh​
=[165][685.34−8.428−8.428.12040][165]=98.37= [1 \quad 65]\begin{bmatrix} 685.34 & -8.428 \\ -8.428 & .12040 \end{bmatrix}\begin{bmatrix} 1 \\ 65 \end{bmatrix} = 98.37=[165][685.34−8.428​−8.428.12040​][165​]=98.37
This is the same result as that obtained in Chapter 2.

Comment
The result in (5.97a) can be derived directly by using (5.46), since Y^h=Xh′b\hat{Y}_h = \mathbf{X}'_h\mathbf{b}
Y^h​=Xh′​b:

σ2{Y^h}=Xh′σ2{b}Xh\sigma^2\{\hat{Y}_h\} = \mathbf{X}'_h\sigma^2\{\mathbf{b}\}\mathbf{X}_hσ2{Y^h​}=Xh′​σ2{b}Xh​
Hence:
σ2{Y^h}=[1Xh][σ2{b0}σ{b0,b1}σ{b1,b0}σ2{b1}][1Xh]\sigma^2\{\hat{Y}_h\} = [1 \quad X_h]\begin{bmatrix} \sigma^2\{b_0\} & \sigma\{b_0, b_1\} \\ \sigma\{b_1, b_0\} & \sigma^2\{b_1\} \end{bmatrix}\begin{bmatrix} 1 \\ X_h \end{bmatrix}σ2{Y^h​}=[1Xh​][σ2{b0​}σ{b1​,b0​}​σ{b0​,b1​}σ2{b1​}​][1Xh​​]
or:
\sigma^2\{\hat{Y}_h\} = \sigma^2\{b_0\} + 2X_h\sigma\{b_0, b_1\} + X_h^2\sigma^2\{b_1\}$
(5.99)
Using the results from (5.92a), we obtain:
σ2{Y^h}=σ2n+σ2Xˉ2∑(Xi−Xˉ)2+2Xh(−Xˉ)σ2∑(Xi−Xˉ)2+Xh2σ2∑(Xi−Xˉ)2\sigma^2\{\hat{Y}_h\} = \frac{\sigma^2}{n} + \frac{\sigma^2\bar{X}^2}{\sum(X_i - \bar{X})^2} + \frac{2X_h(-\bar{X})\sigma^2}{\sum(X_i - \bar{X})^2} + \frac{X_h^2\sigma^2}{\sum(X_i - \bar{X})^2}σ2{Y^h​}=nσ2​+∑(Xi​−Xˉ)2σ2Xˉ2​+∑(Xi​−Xˉ)22Xh​(−Xˉ)σ2​+∑(Xi​−Xˉ)2Xh2​σ2​
which reduces to the familiar expression:
\sigma^2\{\hat{Y}_h\} = \sigma^2\left[\frac{1}{n} + \frac{(X_h - \bar{X})^2}{\sum(X_i - \bar{X})^2}\right]$
**(5.99a
chapter 61 Mar2 / 2)**
Thus, we see explicitly that the variance expression in (5.99a) contains contributions from σ2{b0}\sigma^2\{b_0\}
σ2{b0​}, σ2{b1}\sigma^2\{b_1\}
σ2{b1​}, and σ{b0,b1}\sigma\{b_0, b_1\}
σ{b0​,b1​}, which it must according to (A.30b) since Y^h=b0+b1Xh\hat{Y}_h = b_0 + b_1 X_h
Y^h​=b0​+b1​Xh​ is a linear combination of b0b_0
b0​ and b1b_1
b1​. ■


Prediction of New Observation
The estimated variance s2[pred]s^2[\text{pred}]
s2[pred], given earlier in (2.38), in matrix notation is:

s^2[\text{pred}] = MSE(1 + \mathbf{X}'_h(\mathbf{X'X})^{-1}\mathbf{X}_h)$
(5.100)

Cited Reference
5.1: Graybill, F. A. Matrices with Applications in Statistics. 2nd ed. Belmont, Calif.: Wadsworth, 2002.

Problems
5.1: For the matrices below, obtain (1) A + B, (2) A − B, (3) AC, (4) AB', (5) B'A.
A=[142638]B=[131425]C=[381540]\mathbf{A} = \begin{bmatrix} 1 & 4 \\ 2 & 6 \\ 3 & 8 \end{bmatrix} \quad\quad \mathbf{B} = \begin{bmatrix} 1 & 3 \\ 1 & 4 \\ 2 & 5 \end{bmatrix} \quad\quad \mathbf{C} = \begin{bmatrix} 3 & 8 & 1 \\ 5 & 4 & 0 \end{bmatrix}A=​123​468​​B=​112​345​​C=[35​84​10​]
State the dimension of each resulting matrix.
5.2: For the matrices below, obtain (1) A + C, (2) A − C, (3) B'A, (4) AC, (5) C'A.
A=[21355748]B=[6941]C=[38865124]\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 3 & 5 \\ 5 & 7 \\ 4 & 8 \end{bmatrix} \quad\quad \mathbf{B} = \begin{bmatrix} 6 \\ 9 \\ 4 \\ 1 \end{bmatrix} \quad\quad \mathbf{C} = \begin{bmatrix} 3 & 8 \\ 8 & 6 \\ 5 & 1 \\ 2 & 4 \end{bmatrix}A=​2354​1578​​B=​6941​​C=​3852​8614​​
State the dimension of each resulting matrix.
5.3: Show how the following expressions are written in terms of matrices: (1) Yi−Y^i=eiY_i - \hat{Y}_i = e_i
Yi​−Y^i​=ei​, (2) ∑Xiei=0\sum X_i e_i = 0
∑Xi​ei​=0. Assume i=1,…,4i = 1, \ldots, 4
i=1,…,4.


End of Chapter 5 Notes
