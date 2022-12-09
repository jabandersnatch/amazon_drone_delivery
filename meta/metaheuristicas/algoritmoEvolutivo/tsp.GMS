*************************************************************************
***      Travelling Salesman Problem                                  ***
***                                                                   ***
***      Author: Germán Montoya                                       ***
*************************************************************************

$Set NODES 6

$Set SOURCE 1

Sets
  i   network nodes / n1*n%NODES% /
  e   secuencia de cada enlace seleccionado / e1*e%NODES% /

alias(j,i);
alias(d,e);

$ontext
Table c(i,j) link cost
                 n1       n2      n3      n4      n5     n6      n7
n1               999       1       1       1       1      1       1
n2                 1     999       1       1       2      1       1
n3                 1       1     999       1       2      1       1
n4                 1       1       1     999       2      1       1
n5                 1       2       2       2     999      2       1
n6                 1       1       1       1       2    999       1
n7                 1       1       1       1       1      1     999

;
$offtext

*$ontext
Table c(i,j) link cost
                 n1       n2      n3      n4      n5     n6
n1               999       4       1       4       5      6
n2                 2     999       4       2       6      1
n3                 3       5     999       2       2      3
n4                 4       2       2     999       3      2
n5                 2       2       1       3     999      4
n6                 6       2       3       6       4    999
;
*$offtext

*Dynamic set for reducing the compilation time.
*$ontext
Set links(i,j);

Loop( (i,j),
      if( c(i,j)<999,
        links[i,j]=Yes;
      )
);
*$offtext


Variables
  x(i,j,e)      Indicates if the link i-j is selected or not.
  y(i,e)        Indicates if a node was selected for the link e.
  L(e)        Indicates the link e was selected.
  z           Objective function  ;

Binary Variable x;
Binary Variable y;
Binary Variable L;

Equations
objectiveFunction        objective function
restr1(i,j,e)
restr2(e)
restr3(i,j)
restr4(i)
restr5(j)
restr6(i,j)
restr7(i,e)
restr8(e,d)
restr9(e)
*restr10(e)
restr11(i)
restr12(e)



;

objectiveFunction       ..  z =e= sum((i,j,e), c(i,j) * x(i,j,e));

restr1(i,j,e)                 ..  x(i,j,e) + x(j,i,e) =l= 1;

restr2(e)   ..   sum((i,j), x(i,j,e))=e=1;

restr3(i,j)   ..   sum(e, x(i,j,e))=l=1;

restr4(i)   ..   sum((j,e), x(i,j,e))=e=1;

restr5(j)   ..   sum((i,e), x(i,j,e))=e=1;

restr6(i,j)   ..   sum(e, x(i,j,e) + x(j,i,e))=l=1;

restr7(i,e)$(ord(i)=%SOURCE% and ord(e)=1)   ..   L(e)*sum(j, x(i,j,e)*y(j,e))=e=1;

restr8(e,d)$(ord(d)=ord(e)+1 and ord(e)<%NODES%)
*restr8(e,d)$(ord(e)=1 and ord(d)=ord(e)+1)
                 ..   L(e)*L(d)*sum((i,j)$(links(i,j)), y(i,e)*x(i,j,d)*y(j,d))=e=L(e)*L(d);

restr9(e)   ..    L(e)=e=1;

*restr10(e)$(ord(e)=1)          ..    sum(i, y(i,e))=e=1;

*restr11(e)$(ord(e)=2)          ..    sum(i, y(i,e))=e=1;

restr11(i)          ..    sum(e, y(i,e))=e=1;

restr12(e)          ..    sum(i, y(i,e))=e=1;


Model model1 /all/ ;
option minlp=scip
*option minlp=kestrel
Solve model1 using minlp minimizing z;

Display x.l;
*Display f.l;
Display y.l;
Display L.l;
Display z.l;
