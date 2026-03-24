# SQL Practice Problems — Study Guide
> Based on *SQL Practice Problems* by Sylvia Moestl Vasilik  
> Solutions in **SQLite** dialect by John Weatherwax

---

## Problem 1 — List All Shippers

**Question:**  
Return all columns and rows from the `Shippers` table.

**Answer:**
```sql
SELECT * FROM Shippers;
```

**Sample Output:**

| ShipperID | CompanyName | Phone |
|-----------|-------------|-------|
| 1 | Speedy Express | (503) 555-9831 |
| 2 | United Package | (503) 555-3199 |
| 3 | Federal Shipping | (503) 555-9931 |

---

## Problem 2 — Category Names and Descriptions

**Question:**  
Return only the `CategoryName` and `Description` columns from the `Categories` table.

**Answer:**
```sql
SELECT CategoryName, Description FROM Categories;
```

**Sample Output:**

| CategoryName | Description |
|--------------|-------------|
| Beverages | Soft drinks, coffees, teas, beers, and ales |
| Condiments | Sweet and savory sauces, relishes, spreads, and seasonings |
| Confections | Desserts, candies, and sweet breads |

---

## Problem 3 — Sales Representatives

**Question:**  
Return the `FirstName`, `LastName`, and `HireDate` of all employees whose title is `'Sales Representative'`.

**Answer:**
```sql
SELECT FirstName, LastName, HireDate
FROM Employees
WHERE Title = 'Sales Representative';
```

**Sample Output:**

| FirstName | LastName | HireDate |
|-----------|----------|----------|
| Nancy | Davolio | 1992-05-01 |
| Janet | Leverling | 1992-04-01 |
| Margaret | Peacock | 1993-05-03 |

---

## Problem 4 — US Sales Representatives

**Question:**  
Return `FirstName`, `LastName`, and `HireDate` of employees who are both `'Sales Representative'` **and** based in the USA.

**Answer:**
```sql
SELECT FirstName, LastName, HireDate
FROM Employees
WHERE Title = 'Sales Representative' AND Country = 'USA';
```

**Sample Output:**

| FirstName | LastName | HireDate |
|-----------|----------|----------|
| Nancy | Davolio | 1992-05-01 |
| Janet | Leverling | 1992-04-01 |
| Margaret | Peacock | 1993-05-03 |

---

## Problem 5 — Orders by Employee

**Question:**  
Return `OrderID` and `OrderDate` for all orders placed by `EmployeeID = 5`.

**Answer:**
```sql
SELECT OrderID, OrderDate
FROM Orders
WHERE EmployeeID = 5;
```

**Sample Output:**

| OrderID | OrderDate |
|---------|-----------|
| 10248 | 1996-07-04 |
| 10254 | 1996-07-11 |
| 10269 | 1996-07-31 |

---

## Problem 6 — Non-Manager Suppliers

**Question:**  
Return `SupplierID`, `ContactName`, and `ContactTitle` for all suppliers whose `ContactTitle` is **not** `'Manager'`.

**Answer:**
```sql
SELECT SupplierID, ContactName, ContactTitle
FROM Suppliers
WHERE ContactTitle <> 'Manager';
```

**Sample Output:**

| SupplierID | ContactName | ContactTitle |
|------------|-------------|--------------|
| 1 | Charlotte Cooper | Purchasing Manager |
| 3 | Regina Murphy | Sales Representative |
| 5 | Shelley Burke | Order Administrator |

---

## Problem 7 — Products with "queso"

**Question:**  
Return `ProductID` and `ProductName` for all products whose name contains the word `queso` (case-insensitive).

**Answer:**
```sql
SELECT ProductID, ProductName
FROM Products
WHERE ProductName LIKE '%queso%';
```

**Sample Output:**

| ProductID | ProductName |
|-----------|-------------|
| 11 | Queso Cabrales |
| 12 | Queso Manchego La Pastora |

---

## Problem 8 — Orders from France or Belgium

**Question:**  
Return `OrderID`, `CustomerID`, and `ShipCountry` for orders shipped to France or Belgium.

**Answer:**
```sql
SELECT OrderID, CustomerID, ShipCountry
FROM Orders
WHERE ShipCountry IN ('France', 'Belgium');
```

**Sample Output:**

| OrderID | CustomerID | ShipCountry |
|---------|------------|-------------|
| 10248 | VINET | France |
| 10252 | SUPRD | Belgium |
| 10274 | VINET | France |

---

## Problem 9 — Orders from Latin America

**Question:**  
Return `OrderID`, `CustomerID`, and `ShipCountry` for orders shipped to Brazil, Mexico, Argentina, or Venezuela.

**Answer:**
```sql
SELECT OrderID, CustomerID, ShipCountry
FROM Orders
WHERE ShipCountry IN ('Brazil', 'Mexico', 'Argentina', 'Venezuela');
```

**Sample Output:**

| OrderID | CustomerID | ShipCountry |
|---------|------------|-------------|
| 10250 | HANAR | Brazil |
| 10253 | HANAR | Brazil |
| 10257 | HILAA | Venezuela |

---

## Problem 10 — Employees by Birthdate

**Question:**  
Return `FirstName`, `LastName`, `Title`, and `BirthDate` for all employees, sorted oldest to youngest.

**Answer:**
```sql
SELECT FirstName, LastName, Title, BirthDate
FROM Employees
ORDER BY BirthDate ASC;
```

**Sample Output:**

| FirstName | LastName | Title | BirthDate |
|-----------|----------|-------|-----------|
| Margaret | Peacock | Sales Representative | 1937-09-19 |
| Nancy | Davolio | Sales Representative | 1948-12-08 |
| Andrew | Fuller | Vice President, Sales | 1952-02-19 |

---

## Problem 11 — Birthdate (Date Only)

**Question:**  
Same as Problem 10, but format `BirthDate` to show the date portion only (no time), aliased as `DateOnlyBirthDate`.

**Answer:**
```sql
SELECT FirstName, LastName, Title,
       strftime('%Y-%m-%d', BirthDate) AS DateOnlyBirthDate
FROM Employees
ORDER BY BirthDate ASC;
```

**Sample Output:**

| FirstName | LastName | Title | DateOnlyBirthDate |
|-----------|----------|-------|-------------------|
| Margaret | Peacock | Sales Representative | 1937-09-19 |
| Nancy | Davolio | Sales Representative | 1948-12-08 |
| Andrew | Fuller | Vice President, Sales | 1952-02-19 |

---

## Problem 12 — Full Name Concatenation

**Question:**  
Return `FirstName`, `LastName`, and a concatenated `FullName` (first + space + last) for all employees.

**Answer:**
```sql
SELECT FirstName, LastName,
       FirstName || ' ' || LastName AS FullName
FROM Employees;
```

**Sample Output:**

| FirstName | LastName | FullName |
|-----------|----------|----------|
| Nancy | Davolio | Nancy Davolio |
| Andrew | Fuller | Andrew Fuller |
| Janet | Leverling | Janet Leverling |

---

## Problem 13 — Order Details with Total Price

**Question:**  
Return `OrderID`, `ProductID`, `UnitPrice`, `Quantity`, and a calculated `TotalPrice` (`UnitPrice × Quantity`), ordered by `OrderID` then `ProductID`.

**Answer:**
```sql
SELECT OrderID, ProductID, UnitPrice, Quantity,
       UnitPrice * Quantity AS TotalPrice
FROM [Order Details]
ORDER BY OrderID, ProductID;
```

**Sample Output:**

| OrderID | ProductID | UnitPrice | Quantity | TotalPrice |
|---------|-----------|-----------|----------|------------|
| 10248 | 11 | 14.00 | 12 | 168.00 |
| 10248 | 42 | 9.80 | 10 | 98.00 |
| 10248 | 72 | 34.80 | 5 | 174.00 |

---

## Problem 14 — Count of Customers

**Question:**  
Return the total number of customers in the `Customers` table, aliased as `TotalCustomers`.

**Answer:**
```sql
SELECT COUNT(*) AS TotalCustomers
FROM Customers;
```

**Sample Output:**

| TotalCustomers |
|----------------|
| 91 |

---

## Problem 15 — Earliest Order Date

**Question:**  
Return the earliest (minimum) `OrderDate` from the `Orders` table, aliased as `FirstOrder`.

**Answer:**
```sql
SELECT MIN(OrderDate) AS FirstOrder
FROM Orders;
```

**Sample Output:**

| FirstOrder |
|------------|
| 1996-07-04 |

---

## Problem 16 — Distinct Customer Countries

**Question:**  
Return a distinct list of countries where customers are located.

**Answer:**
```sql
SELECT Country
FROM Customers
GROUP BY Country;
```

**Sample Output:**

| Country |
|---------|
| Argentina |
| Austria |
| Belgium |
| Brazil |
| ... |

---

## Problem 17 — Contact Titles by Count

**Question:**  
Return each `ContactTitle` and the number of customers (`TotalContractTitle`) with that title, sorted by count descending.

**Answer:**
```sql
SELECT ContactTitle, COUNT(*) AS TotalContractTitle
FROM Customers
GROUP BY ContactTitle
ORDER BY TotalContractTitle DESC;
```

**Sample Output:**

| ContactTitle | TotalContractTitle |
|--------------|--------------------|
| Owner | 17 |
| Sales Representative | 17 |
| Marketing Manager | 12 |

---

## Problem 18 — Products with Supplier Name

**Question:**  
Return `ProductID`, `ProductName`, and the supplier's `CompanyName` by joining `Products` with `Suppliers`.

**Answer:**
```sql
SELECT p.ProductID, p.ProductName, s.CompanyName
FROM Products p
LEFT JOIN Suppliers s ON p.SupplierID = s.SupplierID;
```

**Sample Output:**

| ProductID | ProductName | CompanyName |
|-----------|-------------|-------------|
| 1 | Chai | Exotic Liquids |
| 2 | Chang | Exotic Liquids |
| 3 | Aniseed Syrup | Exotic Liquids |

---

## Problem 19 — Orders with Shipper Name (OrderID < 10300)

**Question:**  
Return `OrderID`, formatted `OrderDate`, and the shipper's `CompanyName` for orders with `OrderID < 10300`.

**Answer:**
```sql
SELECT OrderID, date(OrderDate) AS OrderDate, CompanyName
FROM Orders
LEFT JOIN Shippers ON Orders.ShipVia = Shippers.ShipperID
WHERE OrderID < 10300
ORDER BY OrderID;
```

**Sample Output:**

| OrderID | OrderDate | CompanyName |
|---------|-----------|-------------|
| 10248 | 1996-07-04 | Federal Shipping |
| 10249 | 1996-07-05 | Speedy Express |
| 10250 | 1996-07-08 | United Package |

---

## Problem 20 — Products per Category

**Question:**  
Return each `CategoryName` and the count of products in that category, sorted by count descending.

**Answer:**
```sql
SELECT CategoryName, COUNT(*) AS Count
FROM Categories
JOIN Products ON Categories.CategoryID = Products.CategoryID
GROUP BY CategoryName
ORDER BY Count DESC;
```

**Sample Output:**

| CategoryName | Count |
|--------------|-------|
| Confections | 13 |
| Beverages | 12 |
| Condiments | 12 |

---

## Problem 21 — Customers per Country/City

**Question:**  
Return `Country`, `City`, and `TotalCustomers` grouped by both, sorted by count descending.

**Answer:**
```sql
SELECT Country, City, COUNT(*) AS TotalCustomers
FROM Customers
GROUP BY Country, City
ORDER BY TotalCustomers DESC;
```

**Sample Output:**

| Country | City | TotalCustomers |
|---------|------|----------------|
| USA | London | 6 |
| Brazil | Sao Paulo | 4 |
| France | Paris | 3 |

---

## Problem 22 — Products Needing Reorder

**Question:**  
Return products where `UnitsInStock` is at or below the `ReorderLevel`, ordered by `ProductID`.

**Answer:**
```sql
SELECT ProductID, ProductName, UnitsInStock, ReorderLevel
FROM Products
WHERE UnitsInStock <= ReorderLevel
ORDER BY ProductID;
```

**Sample Output:**

| ProductID | ProductName | UnitsInStock | ReorderLevel |
|-----------|-------------|--------------|--------------|
| 2 | Chang | 17 | 25 |
| 3 | Aniseed Syrup | 13 | 25 |
| 11 | Queso Cabrales | 22 | 30 |

---

## Problem 23 — Discontinued Products Needing Reorder

**Question:**  
Return products where combined `UnitsInStock + UnitsOnOrder` is at or below `ReorderLevel` **and** the product is not discontinued.

**Answer:**
```sql
SELECT ProductID, ProductName, UnitsInStock, UnitsOnOrder, ReorderLevel, Discontinued
FROM Products
WHERE (UnitsInStock + UnitsOnOrder) <= ReorderLevel
  AND Discontinued = '0'
ORDER BY ProductID;
```

**Sample Output:**

| ProductID | ProductName | UnitsInStock | UnitsOnOrder | ReorderLevel | Discontinued |
|-----------|-------------|--------------|--------------|--------------|--------------|
| 2 | Chang | 17 | 40 | 25 | 0 |
| 30 | Nord-Ost Matjeshering | 10 | 0 | 15 | 0 |
| 31 | Gorgonzola Telino | 0 | 70 | 20 | 0 |

---

## Problem 24 — Customers with NULL Region Last

**Question:**  
Return `CustomerID`, `CompanyName`, and `Region`, sorted so non-NULL regions appear first (alphabetically), with NULL regions at the end.

**Answer:**
```sql
SELECT CustomerID, CompanyName, Region
FROM Customers
ORDER BY
  (CASE WHEN Region IS NULL THEN 1 ELSE 0 END),
  Region,
  CustomerID;
```

**Sample Output:**

| CustomerID | CompanyName | Region |
|------------|-------------|--------|
| LAMAI | La maison d'Asie | BC |
| MEREP | Mère Paillarde | Québec |
| ANATR | Ana Trujillo Emparedados | (NULL) |

---

## Problem 25 — Top 3 Countries by Average Freight

**Question:**  
Return the top 3 countries with the highest average freight cost.

**Answer:**
```sql
SELECT ShipCountry, AVG(Freight) AS AverageFreight
FROM Orders
GROUP BY ShipCountry
ORDER BY AverageFreight DESC
LIMIT 3;
```

**Sample Output:**

| ShipCountry | AverageFreight |
|-------------|----------------|
| Austria | 184.79 |
| Ireland | 176.27 |
| USA | 138.96 |

---

## Problem 26 — Top 3 Countries by Freight (1997 Only)

**Question:**  
Same as Problem 25, but only for orders placed in 1997.

**Answer:**
```sql
SELECT ShipCountry, AVG(Freight) AS AverageFreight
FROM Orders
WHERE strftime('%Y', OrderDate) = '1997'
GROUP BY ShipCountry
ORDER BY AverageFreight DESC
LIMIT 3;
```

**Sample Output:**

| ShipCountry | AverageFreight |
|-------------|----------------|
| Austria | 186.21 |
| Switzerland | 148.94 |
| USA | 134.76 |

---

## Problem 28 — Top 3 Countries by Freight (Last 365 Days)

**Question:**  
Return the top 3 countries by average freight for orders placed within the last 365 days relative to the most recent order date in the database.

**Answer:**
```sql
SELECT ShipCountry, AVG(Freight) AS AverageFreight
FROM Orders
WHERE OrderDate >= (
  SELECT datetime(julianday(MAX(OrderDate)) - 365) FROM Orders
)
GROUP BY ShipCountry
ORDER BY AverageFreight DESC
LIMIT 3;
```

**Sample Output:**

| ShipCountry | AverageFreight |
|-------------|----------------|
| Austria | 186.21 |
| Ireland | 176.27 |
| USA | 141.06 |

---

## Problem 29 — Employee Order Details (First 20)

**Question:**  
Return `EmployeeID`, `LastName`, `OrderID`, `ProductName`, and `Quantity` by joining Orders, Employees, Order Details, and Products. Show only the first 20 rows ordered by `OrderID` and `ProductID`.

**Answer:**
```sql
SELECT Orders.EmployeeID, Employees.LastName, Orders.OrderID,
       Products.ProductName, [Order Details].Quantity
FROM Orders
LEFT JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
LEFT JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
LEFT JOIN Products ON [Order Details].ProductID = Products.ProductID
ORDER BY Orders.OrderID, [Order Details].ProductID
LIMIT 20;
```

**Sample Output:**

| EmployeeID | LastName | OrderID | ProductName | Quantity |
|------------|----------|---------|-------------|----------|
| 5 | Buchanan | 10248 | Queso Cabrales | 12 |
| 5 | Buchanan | 10248 | Singaporean Hokkien Fried Mee | 10 |
| 5 | Buchanan | 10248 | Mozzarella di Giovanni | 5 |

---

## Problem 30 — Customers with No Orders

**Question:**  
Return customers who have never placed an order. Show both `CustomerID` columns from the join.

**Answer:**
```sql
SELECT Customers.CustomerID AS Customers_CustomerID,
       Orders.CustomerID AS Orders_CustomerID
FROM Customers
LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
WHERE Orders.CustomerID IS NULL
ORDER BY Orders.CustomerID;
```

**Sample Output:**

| Customers_CustomerID | Orders_CustomerID |
|----------------------|-------------------|
| FISSA | (NULL) |
| PARIS | (NULL) |

---

## Problem 31 — Customers with No Orders from Employee 4

**Question:**  
Return customers who have never placed an order handled by `EmployeeID = 4`. Use a conditional JOIN.

**Answer:**
```sql
SELECT Customers.CustomerID AS Customers_CustomerID,
       Orders.CustomerID AS Orders_CustomerID
FROM Customers
LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
  AND Orders.EmployeeID = 4
WHERE Orders.CustomerID IS NULL
ORDER BY Orders.CustomerID;
```

**Sample Output:**

| Customers_CustomerID | Orders_CustomerID |
|----------------------|-------------------|
| ANATR | (NULL) |
| BSBEV | (NULL) |
| CONSH | (NULL) |

---

## Problem 32 — High-Value Orders per Customer (1998)

**Question:**  
Return customers and their individual orders from 1998 where the total order amount (before discount) exceeds $10,000. Show `CustomerID`, `CompanyName`, `OrderID`, and `TotalOrderAmount`.

**Answer:**
```sql
SELECT Customers.CustomerID, Customers.CompanyName, Orders.OrderID,
       SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalOrderAmount
FROM Customers
JOIN Orders ON Customers.CustomerID = Orders.CustomerID
JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
WHERE strftime('%Y', Orders.OrderDate) = '1998'
GROUP BY Customers.CustomerID, Orders.OrderID
HAVING TotalOrderAmount > 10000
ORDER BY TotalOrderAmount DESC;
```

**Sample Output:**

| CustomerID | CompanyName | OrderID | TotalOrderAmount |
|------------|-------------|---------|-----------------|
| QUICK | QUICK-Stop | 11030 | 12615.05 |
| SAVEA | Save-a-lot Markets | 10847 | 11080.00 |
| ERNSH | Ernst Handel | 10979 | 10469.30 |

---

## Problem 33 — Customers with Total Orders > $15,000 (1998)

**Question:**  
Return customers whose total orders in 1998 across all orders exceeded $15,000.

**Answer:**
```sql
SELECT Customers.CustomerID, Customers.CompanyName,
       SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalOrderAmount
FROM Customers
JOIN Orders ON Customers.CustomerID = Orders.CustomerID
JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
WHERE strftime('%Y', Orders.OrderDate) = '1998'
GROUP BY Customers.CustomerID
HAVING TotalOrderAmount > 15000
ORDER BY TotalOrderAmount DESC;
```

**Sample Output:**

| CustomerID | CompanyName | TotalOrderAmount |
|------------|-------------|-----------------|
| QUICK | QUICK-Stop | 61109.90 |
| SAVEA | Save-a-lot Markets | 52245.90 |
| ERNSH | Ernst Handel | 42068.19 |

---

## Problem 34 — Discounted vs. Non-Discounted Totals (1998)

**Question:**  
Return customers with 1998 discounted totals > $15,000, showing both the total without discount and the total with discount applied.

**Answer:**
```sql
SELECT Customers.CustomerID, Customers.CompanyName,
       SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalWithoutDiscount,
       SUM([Order Details].UnitPrice * [Order Details].Quantity * (1 - [Order Details].Discount)) AS TotalWithDiscount
FROM Customers
JOIN Orders ON Customers.CustomerID = Orders.CustomerID
JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
WHERE strftime('%Y', Orders.OrderDate) = '1998'
GROUP BY Customers.CustomerID
HAVING TotalWithDiscount > 15000
ORDER BY TotalWithDiscount DESC;
```

**Sample Output:**

| CustomerID | CompanyName | TotalWithoutDiscount | TotalWithDiscount |
|------------|-------------|----------------------|-------------------|
| QUICK | QUICK-Stop | 61109.90 | 55264.40 |
| SAVEA | Save-a-lot Markets | 52245.90 | 49665.72 |
| ERNSH | Ernst Handel | 42068.19 | 39183.16 |

---

## Problem 35 — Orders Placed on Last Day of Month

**Question:**  
Return `EmployeeID`, `OrderID`, and `OrderDate` for orders where the order was placed on the last day of the month.

**Answer:**
```sql
SELECT Orders.EmployeeID, Orders.OrderID, date(Orders.OrderDate)
FROM Orders
WHERE date(Orders.OrderDate, 'start of month', '+1 month', '-1 day') = date(Orders.OrderDate)
ORDER BY Orders.EmployeeID, Orders.OrderID;
```

**Sample Output:**

| EmployeeID | OrderID | OrderDate |
|------------|---------|-----------|
| 1 | 10461 | 1997-02-28 |
| 1 | 10616 | 1997-07-31 |
| 2 | 10575 | 1997-06-30 |

---

## Problem 36 — Orders with Fewest Line Items

**Question:**  
Return the 10 orders with the fewest distinct line items, showing `OrderID` and `TotalOrderDetails`.

**Answer:**
```sql
SELECT Orders.OrderID, COUNT(Orders.OrderID) AS TotalOrderDetails
FROM Orders
JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
GROUP BY Orders.OrderID
ORDER BY TotalOrderDetails
LIMIT 10;
```

**Sample Output:**

| OrderID | TotalOrderDetails |
|---------|-------------------|
| 10782 | 1 |
| 10807 | 1 |
| 10586 | 1 |

---

## Problem 37 — Random Sample of Orders

**Question:**  
Return 10 random `OrderID`s. Also show how to return a random 2% of all orders.

**Answer:**
```sql
-- Fixed count (10 rows)
SELECT OrderID
FROM Orders
ORDER BY RANDOM()
LIMIT 10;

-- Fixed percentage (2%)
SELECT OrderID
FROM Orders
ORDER BY RANDOM()
LIMIT CAST(0.02 * (SELECT COUNT(*) FROM Orders) AS INTEGER);
```

**Sample Output (varies each run):**

| OrderID |
|---------|
| 10459 |
| 10832 |
| 11001 |

---

## Problem 38 — Line Items with Quantity ≥ 60 in Same Order

**Question:**  
Return orders that have **more than one** line item with a quantity of 60 or more, showing `OrderID`, `Quantity`, and the count.

**Answer:**
```sql
SELECT OrderID, Quantity, COUNT(*) AS Number
FROM [Order Details]
WHERE Quantity >= 60
GROUP BY OrderID, Quantity
HAVING Number > 1;
```

**Sample Output:**

| OrderID | Quantity | Number |
|---------|----------|--------|
| 10263 | 60 | 2 |
| 10658 | 70 | 2 |
| 10990 | 65 | 2 |

---

## Problem 39 — Full Details for Problem 38 Orders (CTE)

**Question:**  
Using a CTE, return all line item details (`OrderID`, `ProductID`, `UnitPrice`, `Quantity`, `Discount`) for the orders identified in Problem 38.

**Answer:**
```sql
WITH PossibleOrderIDs AS (
  SELECT OrderID
  FROM [Order Details]
  WHERE Quantity >= 60
  GROUP BY OrderID, Quantity
  HAVING COUNT(*) > 1
)
SELECT OrderID, ProductID, UnitPrice, Quantity, Discount
FROM [Order Details]
WHERE OrderID IN PossibleOrderIDs
ORDER BY OrderID, Quantity;
```

**Sample Output:**

| OrderID | ProductID | UnitPrice | Quantity | Discount |
|---------|-----------|-----------|----------|----------|
| 10263 | 16 | 13.90 | 60 | 0.25 |
| 10263 | 24 | 3.60 | 28 | 0.00 |
| 10263 | 30 | 20.70 | 60 | 0.25 |

---

## Problem 40 — Fix Duplicate Results with DISTINCT

**Question:**  
The subquery in the provided SQL produces duplicates. Fix it by adding `DISTINCT` to the subquery's `SELECT`.

**Answer:**
```sql
-- Add DISTINCT inside the subquery:
SELECT DISTINCT OrderID
FROM [Order Details]
WHERE ...
```

**Note:** The fix is to add `DISTINCT` in the subquery to eliminate duplicated `OrderID` values before the outer query processes them.

---

## Problem 41 — Late Orders

**Question:**  
Return `OrderID`, `OrderDate`, `RequiredDate`, and `ShippedDate` for orders where `ShippedDate` was on or after `RequiredDate` (i.e., late).

**Answer:**
```sql
SELECT OrderID, OrderDate, RequiredDate, ShippedDate
FROM Orders
WHERE ShippedDate >= RequiredDate
ORDER BY OrderDate ASC;
```

**Sample Output:**

| OrderID | OrderDate | RequiredDate | ShippedDate |
|---------|-----------|--------------|-------------|
| 10264 | 1996-07-24 | 1996-08-21 | 1996-08-23 |
| 10271 | 1996-07-31 | 1996-08-14 | 1996-09-03 |
| 10280 | 1996-08-14 | 1996-09-11 | 1996-09-12 |

---

## Problem 42 — Employees with Most Late Orders

**Question:**  
Return each employee's `EmployeeID`, `LastName`, and total count of late orders (`TotalLateOrders`), sorted by count descending.

**Answer:**
```sql
SELECT Orders.EmployeeID, Employees.LastName, COUNT(*) AS TotalLateOrders
FROM Orders
JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
WHERE ShippedDate >= RequiredDate
GROUP BY Orders.EmployeeID, Employees.LastName
ORDER BY TotalLateOrders DESC;
```

**Sample Output:**

| EmployeeID | LastName | TotalLateOrders |
|------------|----------|-----------------|
| 4 | Peacock | 10 |
| 3 | Leverling | 5 |
| 1 | Davolio | 3 |

---

## Problems 43–47 — Late Order Percentage per Employee

**Question:**  
Return each employee's `EmployeeID`, `LastName`, `TotalOrders`, `TotalLateOrders`, and `PercentLateOrders` (rounded to 2 decimal places), sorted by percentage descending.

**Answer:**
```sql
WITH
TotalNumberOfOrders AS (
  SELECT EmployeeID, COUNT(*) AS TotalOrders
  FROM Orders
  GROUP BY EmployeeID
),
TotalLateOrders AS (
  SELECT EmployeeID, COUNT(*) AS TotalLateOrders
  FROM Orders
  WHERE ShippedDate >= RequiredDate
  GROUP BY EmployeeID
)
SELECT DISTINCT
  Orders.EmployeeID,
  Employees.LastName,
  TotalOrders,
  TotalLateOrders,
  ROUND(CAST(TotalLateOrders AS FLOAT) / TotalOrders, 2) AS PercentLateOrders
FROM Orders
JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
JOIN TotalNumberOfOrders ON Orders.EmployeeID = TotalNumberOfOrders.EmployeeID
LEFT JOIN TotalLateOrders ON Orders.EmployeeID = TotalLateOrders.EmployeeID
ORDER BY PercentLateOrders DESC;
```

**Sample Output:**

| EmployeeID | LastName | TotalOrders | TotalLateOrders | PercentLateOrders |
|------------|----------|-------------|-----------------|-------------------|
| 7 | King | 72 | 10 | 0.14 |
| 4 | Peacock | 156 | 10 | 0.06 |
| 3 | Leverling | 127 | 5 | 0.04 |

---

## Problems 48–49 — Customer Order Size Groups (1998)

**Question:**  
Classify 1998 customers into groups based on their total order amount:
- **low**: $0–$1,000  
- **medium**: $1,001–$5,000  
- **high**: $5,001–$9,999  
- **very high**: $10,000+

Return `CustomerID`, `TotalOrderAmount`, and `CustomerGroup`.

**Answer:**
```sql
WITH CustomerOrderSizes AS (
  SELECT Customers.CustomerID, Customers.CompanyName,
         SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalOrderAmount
  FROM Customers
  JOIN Orders ON Customers.CustomerID = Orders.CustomerID
  JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
  WHERE strftime('%Y', Orders.OrderDate) = '1998'
  GROUP BY Customers.CustomerID
  ORDER BY Customers.CustomerID
)
SELECT CustomerID, TotalOrderAmount,
  CASE
    WHEN TotalOrderAmount > 0    AND TotalOrderAmount <= 1000  THEN 'low'
    WHEN TotalOrderAmount > 1000 AND TotalOrderAmount <= 5000  THEN 'medium'
    WHEN TotalOrderAmount > 5000 AND TotalOrderAmount < 10000  THEN 'high'
    ELSE 'very high'
  END AS CustomerGroup
FROM CustomerOrderSizes
ORDER BY CustomerID;
```

**Sample Output:**

| CustomerID | TotalOrderAmount | CustomerGroup |
|------------|-----------------|---------------|
| ALFKI | 2022.50 | medium |
| ANATR | 514.40 | low |
| ERNSH | 42068.19 | very high |

---

## Problem 50 — Customer Group Summary

**Question:**  
Extend Problems 48–49 to show the count and percentage of customers in each group.

**Answer:**
```sql
WITH CustomerOrderSizes AS (
  SELECT Customers.CustomerID, Customers.CompanyName,
         SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalOrderAmount
  FROM Customers
  JOIN Orders ON Customers.CustomerID = Orders.CustomerID
  JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
  WHERE strftime('%Y', Orders.OrderDate) = '1998'
  GROUP BY Customers.CustomerID
),
CustomerGroups AS (
  SELECT TotalOrderAmount,
    CASE
      WHEN TotalOrderAmount > 0    AND TotalOrderAmount <= 1000  THEN 'low'
      WHEN TotalOrderAmount > 1000 AND TotalOrderAmount <= 5000  THEN 'medium'
      WHEN TotalOrderAmount > 5000 AND TotalOrderAmount < 10000  THEN 'high'
      ELSE 'very high'
    END AS CustomerGroup
  FROM CustomerOrderSizes
)
SELECT CustomerGroup, COUNT(*) AS TotalInGroup,
       ROUND(CAST(COUNT(*) AS DOUBLE) / (SELECT COUNT(*) FROM CustomerGroups), 2) AS PercentageInGroup
FROM CustomerGroups
GROUP BY CustomerGroup
ORDER BY PercentageInGroup DESC;
```

**Sample Output:**

| CustomerGroup | TotalInGroup | PercentageInGroup |
|---------------|--------------|-------------------|
| medium | 37 | 0.45 |
| low | 21 | 0.26 |
| very high | 12 | 0.15 |
| high | 11 | 0.13 |

---

## Problem 51 — Customer Groups via Lookup Table

**Question:**  
Same as Problem 50, but use a `CustomerGroupThresholds` lookup table (with columns `RangeBottom`, `RangeTop`, `CustomerGroupName`) instead of a `CASE` expression.

**Answer:**
```sql
WITH CustomerOrderSizes AS (
  SELECT Customers.CustomerID, Customers.CompanyName,
         SUM([Order Details].UnitPrice * [Order Details].Quantity) AS TotalOrderAmount
  FROM Customers
  JOIN Orders ON Customers.CustomerID = Orders.CustomerID
  JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID
  WHERE strftime('%Y', Orders.OrderDate) = '1998'
  GROUP BY Customers.CustomerID
),
CustomerGroups AS (
  SELECT CustomerGroupName AS CustomerGroup, TotalOrderAmount
  FROM CustomerOrderSizes
  JOIN CustomerGroupThresholds
    ON (RangeBottom < TotalOrderAmount) AND (TotalOrderAmount <= RangeTop)
)
SELECT CustomerGroup, COUNT(*) AS TotalInGroup,
       ROUND(CAST(COUNT(*) AS DOUBLE) / (SELECT COUNT(*) FROM CustomerGroups), 2) AS PercentageInGroup
FROM CustomerGroups
GROUP BY CustomerGroup
ORDER BY PercentageInGroup DESC;
```

**Sample Output:**  
*(Same as Problem 50 — output depends on values in `CustomerGroupThresholds`)*

| CustomerGroup | TotalInGroup | PercentageInGroup |
|---------------|--------------|-------------------|
| medium | 37 | 0.45 |
| low | 21 | 0.26 |
| very high | 12 | 0.15 |
| high | 11 | 0.13 |

---

## Problem 52 — Countries from Suppliers or Customers (UNION)

**Question:**  
Return a combined, sorted list of all distinct countries from both the `Suppliers` and `Customers` tables.

**Answer:**
```sql
SELECT Country FROM Suppliers
UNION
SELECT Country FROM Customers
ORDER BY Country;
```

**Sample Output:**

| Country |
|---------|
| Argentina |
| Australia |
| Austria |
| Belgium |
| Brazil |
| ... |

---

## Problem 53 — Supplier vs. Customer Countries (Full Outer Join Simulation)

**Question:**  
Show all countries from `Suppliers` alongside all countries from `Customers`, using a simulated full outer join (since SQLite lacks `FULL OUTER JOIN`).

**Answer:**
```sql
WITH
SupplierCountries AS (SELECT DISTINCT Country FROM Suppliers),
CustomerCountries AS (SELECT DISTINCT Country FROM Customers)
SELECT SupplierCountries.Country AS SupplierCountry,
       CustomerCountries.Country AS CustomerCountry
FROM SupplierCountries
LEFT JOIN CustomerCountries USING(Country)
UNION
SELECT SupplierCountries.Country AS SupplierCountry,
       CustomerCountries.Country AS CustomerCountry
FROM CustomerCountries
LEFT JOIN SupplierCountries USING(Country)
WHERE NOT ((SupplierCountry IS NULL) AND (CustomerCountry IS NULL));
```

**Sample Output:**

| SupplierCountry | CustomerCountry |
|-----------------|-----------------|
| Australia | Australia |
| Brazil | Brazil |
| Canada | (NULL) |
| (NULL) | Argentina |

---

## Problem 54 — Supplier and Customer Counts by Country

**Question:**  
Return each country with its total number of suppliers and total number of customers. Include countries that appear in only one table (fill the other count with 0).

**Answer:**
```sql
WITH
SupplierCountries AS (
  SELECT Country, COUNT(*) AS TotalSuppliers
  FROM Suppliers WHERE Country IS NOT NULL
  GROUP BY Country
),
CustomerCountries AS (
  SELECT Country, COUNT(*) AS TotalCustomers
  FROM Customers WHERE Country IS NOT NULL
  GROUP BY Country
),
AllCountries AS (
  SELECT Country FROM Suppliers WHERE Country IS NOT NULL
  UNION
  SELECT Country FROM Customers WHERE Country IS NOT NULL
)
SELECT ac.Country,
       IFNULL(sc.TotalSuppliers, 0) AS TotalSuppliers,
       IFNULL(cc.TotalCustomers, 0) AS TotalCustomers
FROM AllCountries ac
LEFT JOIN SupplierCountries sc ON ac.Country = sc.Country
LEFT JOIN CustomerCountries cc ON ac.Country = cc.Country
ORDER BY ac.Country;
```

**Sample Output:**

| Country | TotalSuppliers | TotalCustomers |
|---------|----------------|----------------|
| Argentina | 0 | 3 |
| Australia | 2 | 2 |
| Austria | 0 | 2 |
| Belgium | 1 | 2 |
| Brazil | 1 | 9 |

---

## Problem 55 — First Order Per Country

**Question:**  
Return the first order ever placed to each ship country, including `ShipCountry`, `CustomerID`, `OrderID`, and formatted `OrderDate`.

**Answer:**
```sql
WITH CountriesFirstOrder AS (
  SELECT ShipCountry, MIN(OrderDate) AS FirstOrderDate
  FROM Orders
  GROUP BY ShipCountry
)
SELECT cfo.ShipCountry, CustomerID, OrderID, date(OrderDate) AS OrderDate
FROM CountriesFirstOrder cfo
LEFT JOIN Orders o
  ON (cfo.FirstOrderDate = o.OrderDate) AND (cfo.ShipCountry = o.ShipCountry)
ORDER BY cfo.ShipCountry;
```

**Sample Output:**

| ShipCountry | CustomerID | OrderID | OrderDate |
|-------------|------------|---------|-----------|
| Argentina | OCEAN | 10409 | 1997-01-09 |
| Austria | ERNSH | 10258 | 1996-07-17 |
| Belgium | SUPRD | 10252 | 1996-07-09 |

---

## Problem 56 — Customer Re-orders Within 5 Days

**Question:**  
Return pairs of orders from the same customer where the second order was placed within 5 days of the first, showing both `OrderID`s, both `OrderDate`s, and `DaysBetween`.

**Answer:**
```sql
SELECT
  io.CustomerID,
  io.OrderID AS InitialOrderID,
  date(io.OrderDate) AS InitialOrderDate,
  so.OrderID AS NextOrderID,
  date(so.OrderDate) AS NextOrderDate,
  CAST(julianday(so.OrderDate) - julianday(io.OrderDate) AS INTEGER) AS DaysBetween
FROM Orders io
JOIN Orders so
  ON (io.CustomerID = so.CustomerID) AND (io.OrderID < so.OrderID)
WHERE DaysBetween <= 5;
```

**Sample Output:**

| CustomerID | InitialOrderID | InitialOrderDate | NextOrderID | NextOrderDate | DaysBetween |
|------------|----------------|------------------|-------------|---------------|-------------|
| BOLID | 10326 | 1996-10-10 | 10801 | 1998-01-01 | 3 |
| ERNSH | 10258 | 1996-07-17 | 10263 | 1996-07-23 | 5 |
| QUICK | 10273 | 1996-08-05 | 10285 | 1996-08-26 | 4 |

---

## Problem 57 — Cumulative Sales (Window Function)

**Question:**  
Calculate a running/cumulative total of sales per employee ordered by date. *(Note: SQLite did not support window functions at the time of this solution manual.)*

**Answer:**
```sql
-- SQLite does not currently support window functions.
-- In SQL Server or PostgreSQL, you would use:
SELECT
  EmployeeID,
  OrderDate,
  SUM(Freight) OVER (PARTITION BY EmployeeID ORDER BY OrderDate) AS CumulativeFreight
FROM Orders
ORDER BY EmployeeID, OrderDate;
```

**Note:** This query is written for SQL Server / PostgreSQL syntax. SQLite requires a correlated subquery workaround for cumulative aggregation.

---

*End of Study Guide — 57 Problems*
