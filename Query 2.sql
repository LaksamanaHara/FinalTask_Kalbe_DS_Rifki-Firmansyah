select distinct gender  
from customer c


select gender, avg(age)  
from customer c
group by gender 