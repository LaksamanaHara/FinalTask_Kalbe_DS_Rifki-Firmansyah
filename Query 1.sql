select distinct "Marital Status" 
from customer c

select "Marital Status" , avg(age)  
from customer c
group by "Marital Status" 