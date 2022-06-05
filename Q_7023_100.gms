$title Stormwater management model: SUBID 7023, 100 yr rain
* Minimax problem: Stormwater management

Sets
   i 'Node' / n1*n9 /
   s(i) 'Supply node'  /n1/
   d(i) 'Demand node'  /n2,n3,n4,n6*n9/
   t(i) 'Transshipment node'  /n5/
;

Alias (i,j);

Set arc(i,j) 'Arcs from node i to node j'
/n1.n2, n1.n3, n2.n4, n2.n5, n3.n4, n2.n6, n3.n6, n6.n7, n6.n8, n6.n9, n4.n9 /;

Display i, s, d, t, arc;

Parameter
   b(s) 'Supply at node s'
        / n1  161530.16
        /;

Display b;

Parameter
   b_up(d) 'Upper limit of supply at node d'
        / n2  0
          n3  -90345.76
          n4  0
          n6  0
          n7  0
          n8  0
          n9  0 /;

*b_up(d)=0;

Parameter
   b_down(d) 'Lower limit of supply at node d'
        / n2  0 
          n3  -90345.76
          n4  0
          n6  0
          n7  -5400
          n8  -36000
          n9  -71184.4 /;
          
*n2 and n6 do not retain any water, i.e. zero
*n3 can retain up to 113569. This is higher than the inflow amount, so the inflow amount is used here instead.
*n9 has no maximum. Hence the lower bound is the total amount from n2.

Parameter
   dd(t)   'Cost of planting a tree at node t'
        / n5 1.736 /;

Parameter
   a(t)   'Absorption rate per tree at node t'
        / n5 0.4 /;


Display b, b_up, b_down, dd, a;

Parameter
   c(i,j) 'Cost from sending a unit of water from node i to node j'
            / n1.n2 0, n1.n3 0
              n2.n4 0, n2.n5 0
              n3.n4 0, n2.n6 0
              n3.n6 0, n6.n7 5.5
              n6.n8 5.15, n6.n9 0
              n4.n9 0
               /;

Parameter
   u(i,j) 'Upper bound of capacity on arc from node i to node j'
            / n1.n2 71184.4, n1.n3 90345.76
              n2.n4 71184.4, n2.n5 71184.4
              n3.n4 90345.76,n2.n6 28556.58
              n3.n6 36243.42, n6.n7 5400
              n6.n8 64800, n6.n9 64800
              n4.n9 71184.4 /;

*capacities for n2 and n3 equal the values assigned from node 1, until they reach node 6.
*node 6's capacity here consideres one pumpstation in the area.
*node 4's capacity is represented by the entire precipitation amount.

Parameter
   l(i,j) 'Lower bound of capacity on arc from node i to node j'
            / n1.n2 0, n1.n3 0
              n2.n4 0, n2.n5 0
              n3.n4 0, n2.n6 0
              n3.n6 0, n6.n7 0
              n6.n8 0, n6.n9 0
              n4.n9 0  /;
              
Scalars
    c_t  'Target cost value' / 0.00001 /
    w_t  'Target water amount' / 0.00001 /
    w_c  'Weight of objective function for cost' / 1 /
    w_w  'Weight of objective function for water' / 10 /
    ;

*0.00001 is used to avoid DIV/0 error.

Variable
   X(i,j) 'Water sent from node i to node j'
   W(t)   'Water absorbed by plantation at node t'
   Y(t)   'Number of trees planted at node t'
   V      'Water sent to node d'
   Z      'Total cost from flow'
   Q      'Maximum weighted deviation from target value';

Positive variables
   W;

Integer variables
   Y;

Equation
   WCF       'Weighted cost function'
   WWF       'Weighted water function'
   FC_S(s)   'Flow-conservation constraint for supply node s'
   FC_D_up(d)    'Upper flow-conservation constraint for demand node d'
   FC_D_down(d)  'Lower flow-conservation constraint for demand node d'
   FC_T(t)    'Flow-conservation constraint for transshipment node t'
   W_up(t)       'Upper bound on water absorption for transshipment node t'
   UB(i,j)   'Upper bound for flow from node i to node j'
   LB(i,j)   'Lower bound for flow from node i to node j'
   Y_up(t)      'Upper bound for tree count'
   Unt       'Untreated water at node 9'
   Cost      'Total cost'
   ;

WCF.. Q =G= w_c*((sum(arc(i,j), c(arc)*X(arc))+sum(t, dd(t)*Y(t))-c_t)/c_t);

WwF.. Q =G= w_w*((sum(arc(j,'n9'),X(j,'n9'))-w_t)/w_t);

FC_S(s).. sum(arc(s,j),X(s,j)) - sum(arc(j,s),X(j,s)) =E= b(s);

FC_D_up(d).. sum(arc(d,j),X(d,j)) - sum(arc(j,d),X(j,d)) =L= b_up(d);

FC_D_down(d).. -sum(arc(d,j),X(d,j)) + sum(arc(j,d),X(j,d)) =L= -b_down(d);

W_up(t)..    W(t) =L= a(t)*Y(t);

FC_T(t).. sum(arc(t,j),X(t,j)) - sum(arc(j,t),X(j,t)) =E= -W(t);

UB(i,j)..   X(i,j) =L= u(i,j);

LB(i,j)..   X(i,j) =G= l(i,j);

Y_up(t)..   Y(t) =L= 12942;

Unt..       V =E= sum(arc(j,'n9'),X(j,'n9'));

Cost..      Z =E= sum(arc(i,j), c(arc)*X(arc))+sum(t, dd(t)*Y(t));

Model WMQ / all /;

option optcr=0;
option reslim = 3000000;
Option Iterlim=1000000;

Option MIP = Cplex;

Solve WMQ using mip minimizing Q;

display X.l, W.l, Y.l, V.l, Z.l;