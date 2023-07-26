select s.storename, sum(t.qty) as total_qty
from store s 
join "Transaction" t 
on s.storeid = t.storeid 
group by s.storename 
order by total_qty desc

