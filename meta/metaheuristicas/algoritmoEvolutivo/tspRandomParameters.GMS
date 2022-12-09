*************************************************************************
***      Travelling Salesman Problem                                  ***
***                                                                   ***
***      Author: Germán Montoya                                       ***
*************************************************************************

$Set NODES 5

$Set SOURCE 1

Sets
  i   network nodes / n1*n%NODES% /
  e   secuencia de cada enlace seleccionado / e1*e%NODES% /

alias(j,i);
alias(d,e);

Scalar dij / 0 /;

Parameter  coorX(i)   Coordenada X
                    / n1 82, n2 91, n3 12, n4 92, n5 63/;

Parameter  coorY(i)   Coordenada Y
                    / n1 66, n2 3, n3 85, n4 94, n5 68/;


Parameter  c(i,j)   costo;

loop( (i,j),

      if( ord(i)=ord(j),

         c(i,j)=999;
      else
         dij = sqrt( sqr(coorX(i) - coorX(j)) +  sqr(coorY(i) - coorY(j)) );

         c(i,j)=dij;

      );

);

Variables
  x(i,j,e)      Indicates if the link i-j is selected or not.
  y(i,e)        Indicates if a node was selected for the link e.
  L(e)        Indicates the link e was selected.
  z           Objective function;

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
                 ..   L(e)*L(d)*sum((i,j), y(i,e)*x(i,j,d)*y(j,d))=e=L(e)*L(d);

restr9(e)   ..    L(e)=e=1;


restr11(i)          ..    sum(e, y(i,e))=e=1;

restr12(e)          ..    sum(i, y(i,e))=e=1;










Model model1 /all/ ;
option minlp=SCIP
Solve model1 using minlp minimizing z;

Display x.l;
Display y.l;
Display L.l;
Display z.l;
