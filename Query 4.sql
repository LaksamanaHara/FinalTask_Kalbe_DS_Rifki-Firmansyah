select p."Product Name", sum(t.totalamount) as total_amount
from product p  
join "Transaction" t 
on p.productid  = t.productid 
group by p."Product Name"  
order by total_amount desc