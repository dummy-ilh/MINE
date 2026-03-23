# SQL Antipatterns — All Code Examples & Key Notes

---

## Example Database Setup

```sql
CREATE TABLE Accounts (
  account_id SERIAL PRIMARY KEY,
  account_name VARCHAR(20),
  first_name VARCHAR(20),
  last_name VARCHAR(20),
  email VARCHAR(100),
  password_hash CHAR(64),
  portrait_image BLOB,
  hourly_rate NUMERIC(9,2)
);

CREATE TABLE BugStatus (
  status VARCHAR(20) PRIMARY KEY
);

CREATE TABLE Bugs (
  bug_id SERIAL PRIMARY KEY,
  date_reported DATE NOT NULL,
  summary VARCHAR(80),
  description VARCHAR(1000),
  resolution VARCHAR(1000),
  reported_by BIGINT UNSIGNED NOT NULL,
  assigned_to BIGINT UNSIGNED,
  verified_by BIGINT UNSIGNED,
  status VARCHAR(20) NOT NULL DEFAULT 'NEW',
  priority VARCHAR(20),
  hours NUMERIC(9,2),
  FOREIGN KEY (reported_by) REFERENCES Accounts(account_id),
  FOREIGN KEY (assigned_to) REFERENCES Accounts(account_id),
  FOREIGN KEY (verified_by) REFERENCES Accounts(account_id),
  FOREIGN KEY (status) REFERENCES BugStatus(status)
);

CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  bug_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME NOT NULL,
  comment TEXT NOT NULL,
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id),
  FOREIGN KEY (author) REFERENCES Accounts(account_id)
);

CREATE TABLE Screenshots (
  bug_id BIGINT UNSIGNED NOT NULL,
  image_id BIGINT UNSIGNED NOT NULL,
  screenshot_image BLOB,
  caption VARCHAR(100),
  PRIMARY KEY (bug_id, image_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

CREATE TABLE Tags (
  bug_id BIGINT UNSIGNED NOT NULL,
  tag VARCHAR(20) NOT NULL,
  PRIMARY KEY (bug_id, tag),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

CREATE TABLE Products (
  product_id SERIAL PRIMARY KEY,
  product_name VARCHAR(50)
);

CREATE TABLE BugsProducts (
  bug_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (bug_id, product_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id),
  FOREIGN KEY (product_id) REFERENCES Products(product_id)
);
```

---

# PART I — Logical Database Design Antipatterns

---

## Chapter 2 — Jaywalking (Comma-Separated Lists)

### ❌ Antipattern

```sql
-- Storing multiple account IDs as comma-separated string
CREATE TABLE Products (
  product_id SERIAL PRIMARY KEY,
  product_name VARCHAR(1000),
  account_id VARCHAR(100) -- comma-separated list
);

INSERT INTO Products (product_id, product_name, account_id)
VALUES (DEFAULT, 'Visual TurboBuilder', '12,34');

-- Querying is awkward and can't use indexes
SELECT * FROM Products WHERE account_id REGEXP '[[:<:]]12[[:>:]]';

-- Joining is costly
SELECT * FROM Products AS p JOIN Accounts AS a
  ON p.account_id REGEXP '[[:<:]]' || a.account_id || '[[:>:]]'
WHERE p.product_id = 123;

-- Counting is hacky
SELECT product_id,
  LENGTH(account_id) - LENGTH(REPLACE(account_id, ',', '')) + 1
  AS contacts_per_product
FROM Products;

-- Adding a value
UPDATE Products
SET account_id = account_id || ',' || 56
WHERE product_id = 123;
```

### ✅ Solution — Intersection Table

```sql
CREATE TABLE Contacts (
  product_id BIGINT UNSIGNED NOT NULL,
  account_id BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (product_id, account_id),
  FOREIGN KEY (product_id) REFERENCES Products(product_id),
  FOREIGN KEY (account_id) REFERENCES Accounts(account_id)
);

INSERT INTO Contacts (product_id, account_id)
VALUES (123, 12), (123, 34), (345, 23), (567, 12), (567, 34);

-- Query products for an account
SELECT p.*
FROM Products AS p JOIN Contacts AS c ON (p.product_id = c.product_id)
WHERE c.account_id = 34;

-- Aggregate: accounts per product
SELECT product_id, COUNT(*) AS accounts_per_product
FROM Contacts GROUP BY product_id;

-- Add/remove easily
INSERT INTO Contacts (product_id, account_id) VALUES (456, 34);
DELETE FROM Contacts WHERE product_id = 456 AND account_id = 34;
```

> **Note:** Store each value in its own column and row.

---

## Chapter 3 — Naive Trees (Adjacency List)

### ❌ Antipattern — Adjacency List

```sql
CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  parent_id BIGINT UNSIGNED,
  bug_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME NOT NULL,
  comment TEXT NOT NULL,
  FOREIGN KEY (parent_id) REFERENCES Comments(comment_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id),
  FOREIGN KEY (author) REFERENCES Accounts(account_id)
);

-- Only queries 2 levels
SELECT c1.*, c2.*
FROM Comments c1 LEFT OUTER JOIN Comments c2
  ON c2.parent_id = c1.comment_id;

-- Fixed 4-level depth query (awkward)
SELECT c1.*, c2.*, c3.*, c4.*
FROM Comments c1
LEFT OUTER JOIN Comments c2 ON c2.parent_id = c1.comment_id
LEFT OUTER JOIN Comments c3 ON c3.parent_id = c2.comment_id
LEFT OUTER JOIN Comments c4 ON c4.parent_id = c3.comment_id;

-- Delete subtree (multiple steps required)
SELECT comment_id FROM Comments WHERE parent_id = 4; -- returns 5 and 6
SELECT comment_id FROM Comments WHERE parent_id = 5; -- returns none
SELECT comment_id FROM Comments WHERE parent_id = 6; -- returns 7
DELETE FROM Comments WHERE comment_id IN (7);
DELETE FROM Comments WHERE comment_id IN (5, 6);
DELETE FROM Comments WHERE comment_id = 4;
```

### ✅ Legitimate use — Recursive CTE (SQL-99)

```sql
WITH CommentTree (comment_id, bug_id, parent_id, author, comment, depth)
AS (
  SELECT *, 0 AS depth FROM Comments WHERE parent_id IS NULL
  UNION ALL
  SELECT c.*, ct.depth+1 AS depth
  FROM CommentTree ct JOIN Comments c ON (ct.comment_id = c.parent_id)
)
SELECT * FROM CommentTree WHERE bug_id = 1234;
```

### ✅ Solution 1 — Path Enumeration

```sql
CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  path VARCHAR(1000),
  bug_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME NOT NULL,
  comment TEXT NOT NULL
);

-- Query ancestors of comment #7
SELECT * FROM Comments AS c
WHERE '1/4/6/7/' LIKE c.path || '%';

-- Query descendants of comment #4
SELECT * FROM Comments AS c
WHERE c.path LIKE '1/4/' || '%';

-- Insert new node
INSERT INTO Comments (author, comment) VALUES ('Ollie', 'Good job!');
UPDATE Comments
SET path = (SELECT path FROM Comments WHERE comment_id = 7)
  || LAST_INSERT_ID() || '/'
WHERE comment_id = LAST_INSERT_ID();
```

### ✅ Solution 2 — Nested Sets

```sql
CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  nsleft INTEGER NOT NULL,
  nsright INTEGER NOT NULL,
  bug_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME NOT NULL,
  comment TEXT NOT NULL
);

-- Get descendants of comment #4
SELECT c2.*
FROM Comments AS c1
JOIN Comments AS c2 ON c2.nsleft BETWEEN c1.nsleft AND c1.nsright
WHERE c1.comment_id = 4;

-- Get ancestors of comment #6
SELECT c2.*
FROM Comments AS c1
JOIN Comments AS c2 ON c1.nsleft BETWEEN c2.nsleft AND c2.nsright
WHERE c1.comment_id = 6;

-- Get direct parent of comment #6
SELECT parent.*
FROM Comments AS c
JOIN Comments AS parent ON c.nsleft BETWEEN parent.nsleft AND parent.nsright
LEFT OUTER JOIN Comments AS in_between
  ON c.nsleft BETWEEN in_between.nsleft AND in_between.nsright
  AND in_between.nsleft BETWEEN parent.nsleft AND parent.nsright
WHERE c.comment_id = 6
  AND in_between.comment_id IS NULL;

-- Insert new leaf (make space first)
UPDATE Comments
SET nsleft = CASE WHEN nsleft >= 8 THEN nsleft+2 ELSE nsleft END,
    nsright = nsright+2
WHERE nsright >= 7;

INSERT INTO Comments (nsleft, nsright, author, comment)
VALUES (8, 9, 'Fran', 'Me too!');
```

### ✅ Solution 3 — Closure Table

```sql
CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  bug_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME NOT NULL,
  comment TEXT NOT NULL
);

CREATE TABLE TreePaths (
  ancestor BIGINT UNSIGNED NOT NULL,
  descendant BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (ancestor, descendant),
  FOREIGN KEY (ancestor) REFERENCES Comments(comment_id),
  FOREIGN KEY (descendant) REFERENCES Comments(comment_id)
);

-- Get descendants of comment #4
SELECT c.*
FROM Comments AS c
JOIN TreePaths AS t ON c.comment_id = t.descendant
WHERE t.ancestor = 4;

-- Get ancestors of comment #6
SELECT c.*
FROM Comments AS c
JOIN TreePaths AS t ON c.comment_id = t.ancestor
WHERE t.descendant = 6;

-- Insert new child of comment #5
INSERT INTO TreePaths (ancestor, descendant)
  SELECT t.ancestor, 8 FROM TreePaths AS t WHERE t.descendant = 5
  UNION ALL SELECT 8, 8;

-- Delete leaf node #7
DELETE FROM TreePaths WHERE descendant = 7;

-- Delete subtree rooted at #4
DELETE FROM TreePaths
WHERE descendant IN (
  SELECT descendant FROM TreePaths WHERE ancestor = 4
);

-- Move subtree: first disconnect
DELETE FROM TreePaths
WHERE descendant IN (SELECT descendant FROM TreePaths WHERE ancestor = 6)
  AND ancestor IN (
    SELECT ancestor FROM TreePaths WHERE descendant = 6 AND ancestor != descendant
  );

-- Then reconnect under comment #3
INSERT INTO TreePaths (ancestor, descendant)
SELECT supertree.ancestor, subtree.descendant
FROM TreePaths AS supertree
CROSS JOIN TreePaths AS subtree
WHERE supertree.descendant = 3 AND subtree.ancestor = 6;

-- Add path_length for direct parent/child queries
SELECT * FROM TreePaths WHERE ancestor = 4 AND path_length = 1;
```

> **Note:** Model both tree entries and relationships to suit your work.

---

## Chapter 4 — ID Required (Bad Primary Key Conventions)

### ❌ Antipattern

```sql
-- Generic id column on every table
CREATE TABLE Bugs (
  id SERIAL PRIMARY KEY,
  description VARCHAR(1000)
);

-- Allows duplicates in intersection tables
CREATE TABLE BugsProducts (
  id SERIAL PRIMARY KEY,
  bug_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id),
  FOREIGN KEY (product_id) REFERENCES Products(product_id)
);
-- Three identical rows allowed:
INSERT INTO BugsProducts (bug_id, product_id)
VALUES (1234, 1), (1234, 1), (1234, 1);
```

### ✅ Solution

```sql
-- Use descriptive names for primary keys
CREATE TABLE Bugs (
  bug_id SERIAL PRIMARY KEY,
  description VARCHAR(1000)
);

-- Use compound primary key for intersection tables
CREATE TABLE BugsProducts (
  bug_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (bug_id, product_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id),
  FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

-- Descriptive foreign key names
CREATE TABLE Bugs (
  reported_by BIGINT UNSIGNED NOT NULL,
  FOREIGN KEY (reported_by) REFERENCES Accounts(account_id)
);

-- Ruby on Rails: override primary key name
-- class Bug < ActiveRecord::Base
--   set_primary_key "bug_id"
-- end

-- Use USING syntax when column names match
SELECT * FROM Bugs JOIN BugsProducts USING (bug_id);
```

> **Note:** Conventions are good only if they are helpful.

---

## Chapter 5 — Keyless Entry (No Foreign Keys)

### ❌ Antipattern — Manual integrity checks

```sql
-- Check before insert
SELECT account_id FROM Accounts WHERE account_id = 1;
INSERT INTO Bugs (reported_by) VALUES (1);

-- Check before delete
SELECT bug_id FROM Bugs WHERE reported_by = 1;
DELETE FROM Accounts WHERE account_id = 1;

-- Find orphans manually
SELECT b.bug_id, b.status
FROM Bugs b LEFT OUTER JOIN BugStatus s ON (b.status = s.status)
WHERE s.status IS NULL;

-- Fix invalid data
UPDATE Bugs SET status = DEFAULT WHERE status = 'BANANA';
```

### ✅ Solution — Declare Foreign Key Constraints

```sql
CREATE TABLE Bugs (
  reported_by BIGINT UNSIGNED NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'NEW',
  FOREIGN KEY (reported_by) REFERENCES Accounts(account_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
  FOREIGN KEY (status) REFERENCES BugStatus(status)
    ON UPDATE CASCADE ON DELETE SET DEFAULT
);
```

> **Note:** Make your database mistake-proof with constraints.

---

## Chapter 6 — Entity-Attribute-Value (EAV)

### ❌ Antipattern

```sql
CREATE TABLE Issues (issue_id SERIAL PRIMARY KEY);

CREATE TABLE IssueAttributes (
  issue_id BIGINT UNSIGNED NOT NULL,
  attr_name VARCHAR(100) NOT NULL,
  attr_value VARCHAR(100),
  PRIMARY KEY (issue_id, attr_name),
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id)
);

-- Querying a specific attribute is verbose
SELECT issue_id, attr_value AS "date_reported"
FROM IssueAttributes WHERE attr_name = 'date_reported';

-- Reconstructing a row requires many joins
SELECT i.issue_id,
  i1.attr_value AS "date_reported",
  i2.attr_value AS "status",
  i3.attr_value AS "priority",
  i4.attr_value AS "description"
FROM Issues AS i
LEFT OUTER JOIN IssueAttributes AS i1 ON i.issue_id = i1.issue_id AND i1.attr_name = 'date_reported'
LEFT OUTER JOIN IssueAttributes AS i2 ON i.issue_id = i2.issue_id AND i2.attr_name = 'status'
LEFT OUTER JOIN IssueAttributes AS i3 ON i.issue_id = i3.issue_id AND i3.attr_name = 'priority'
LEFT OUTER JOIN IssueAttributes AS i4 ON i.issue_id = i4.issue_id AND i4.attr_name = 'description'
WHERE i.issue_id = 1234;
```

### ✅ Solution A — Single Table Inheritance

```sql
CREATE TABLE Issues (
  issue_id SERIAL PRIMARY KEY,
  reported_by BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED,
  priority VARCHAR(20),
  version_resolved VARCHAR(20),
  status VARCHAR(20),
  issue_type VARCHAR(10),     -- BUG or FEATURE
  severity VARCHAR(20),       -- only for bugs
  version_affected VARCHAR(20), -- only for bugs
  sponsor VARCHAR(50)         -- only for feature requests
);
```

### ✅ Solution B — Concrete Table Inheritance

```sql
CREATE TABLE Bugs (
  issue_id SERIAL PRIMARY KEY,
  reported_by BIGINT UNSIGNED NOT NULL,
  severity VARCHAR(20),
  version_affected VARCHAR(20)
);

CREATE TABLE FeatureRequests (
  issue_id SERIAL PRIMARY KEY,
  reported_by BIGINT UNSIGNED NOT NULL,
  sponsor VARCHAR(50)
);

-- Union view for querying all types
CREATE VIEW Issues AS
  SELECT b.*, 'bug' AS issue_type FROM Bugs AS b
  UNION ALL
  SELECT f.*, 'feature' AS issue_type FROM FeatureRequests AS f;
```

### ✅ Solution C — Class Table Inheritance

```sql
CREATE TABLE Issues (
  issue_id SERIAL PRIMARY KEY,
  reported_by BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED,
  priority VARCHAR(20),
  version_resolved VARCHAR(20),
  status VARCHAR(20)
);

CREATE TABLE Bugs (
  issue_id BIGINT UNSIGNED PRIMARY KEY,
  severity VARCHAR(20),
  version_affected VARCHAR(20),
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id)
);

CREATE TABLE FeatureRequests (
  issue_id BIGINT UNSIGNED PRIMARY KEY,
  sponsor VARCHAR(50),
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id)
);

-- Query all types
SELECT i.*, b.*, f.*
FROM Issues AS i
LEFT OUTER JOIN Bugs AS b USING (issue_id)
LEFT OUTER JOIN FeatureRequests AS f USING (issue_id);
```

### ✅ Solution D — Semistructured Data (Serialized LOB)

```sql
CREATE TABLE Issues (
  issue_id SERIAL PRIMARY KEY,
  reported_by BIGINT UNSIGNED NOT NULL,
  issue_type VARCHAR(10),
  attributes TEXT NOT NULL  -- XML or JSON blob
);
```

### Post-Processing with EAV (if you're stuck with it)

```sql
-- Fetch all attributes as rows, process in app code
SELECT issue_id, attr_name, attr_value
FROM IssueAttributes WHERE issue_id = 1234;
```

> **Note:** Use metadata for metadata.

---

## Chapter 7 — Polymorphic Associations

### ❌ Antipattern

```sql
CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  issue_type VARCHAR(20),  -- "Bugs" or "FeatureRequests"
  issue_id BIGINT UNSIGNED NOT NULL,  -- no foreign key!
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME,
  comment TEXT,
  FOREIGN KEY (author) REFERENCES Accounts(account_id)
);

-- Querying requires awkward outer joins
SELECT *
FROM Comments AS c
LEFT OUTER JOIN Bugs AS b ON (b.issue_id = c.issue_id AND c.issue_type = 'Bugs')
LEFT OUTER JOIN FeatureRequests AS f ON (f.issue_id = c.issue_id AND c.issue_type = 'FeatureRequests');
```

### ✅ Solution A — Reverse the Reference (Intersection Tables)

```sql
CREATE TABLE BugsComments (
  issue_id BIGINT UNSIGNED NOT NULL,
  comment_id BIGINT UNSIGNED NOT NULL,
  UNIQUE KEY (comment_id),
  PRIMARY KEY (issue_id, comment_id),
  FOREIGN KEY (issue_id) REFERENCES Bugs(issue_id),
  FOREIGN KEY (comment_id) REFERENCES Comments(comment_id)
);

CREATE TABLE FeaturesComments (
  issue_id BIGINT UNSIGNED NOT NULL,
  comment_id BIGINT UNSIGNED NOT NULL,
  UNIQUE KEY (comment_id),
  PRIMARY KEY (issue_id, comment_id),
  FOREIGN KEY (issue_id) REFERENCES FeatureRequests(issue_id),
  FOREIGN KEY (comment_id) REFERENCES Comments(comment_id)
);

-- Query comments for bug #1234
SELECT * FROM BugsComments AS b
JOIN Comments AS c USING (comment_id)
WHERE b.issue_id = 1234;
```

### ✅ Solution B — Common Super-Table

```sql
CREATE TABLE Issues (issue_id SERIAL PRIMARY KEY);

CREATE TABLE Bugs (
  issue_id BIGINT UNSIGNED PRIMARY KEY,
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id)
);

CREATE TABLE FeatureRequests (
  issue_id BIGINT UNSIGNED PRIMARY KEY,
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id)
);

CREATE TABLE Comments (
  comment_id SERIAL PRIMARY KEY,
  issue_id BIGINT UNSIGNED NOT NULL,
  author BIGINT UNSIGNED NOT NULL,
  comment_date DATETIME,
  comment TEXT,
  FOREIGN KEY (issue_id) REFERENCES Issues(issue_id),
  FOREIGN KEY (author) REFERENCES Accounts(account_id)
);
```

> **Note:** In every table relationship, there is one referencing table and one referenced table.

---

## Chapter 8 — Multicolumn Attributes

### ❌ Antipattern

```sql
CREATE TABLE Bugs (
  bug_id SERIAL PRIMARY KEY,
  description VARCHAR(1000),
  tag1 VARCHAR(20),
  tag2 VARCHAR(20),
  tag3 VARCHAR(20)
);

-- Searching is tedious
SELECT * FROM Bugs
WHERE 'performance' IN (tag1, tag2, tag3)
  AND 'printing' IN (tag1, tag2, tag3);

-- Remove a tag (complex)
UPDATE Bugs
SET tag1 = NULLIF(tag1, 'performance'),
    tag2 = NULLIF(tag2, 'performance'),
    tag3 = NULLIF(tag3, 'performance')
WHERE bug_id = 3456;

-- Add a column later (costly)
ALTER TABLE Bugs ADD COLUMN tag4 VARCHAR(20);
```

### ✅ Solution — Dependent Table

```sql
CREATE TABLE Tags (
  bug_id BIGINT UNSIGNED NOT NULL,
  tag VARCHAR(20),
  PRIMARY KEY (bug_id, tag),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

INSERT INTO Tags (bug_id, tag)
VALUES (1234, 'crash'), (3456, 'printing'), (3456, 'performance');

-- Simple search
SELECT * FROM Bugs JOIN Tags USING (bug_id) WHERE tag = 'performance';

-- Two-tag search
SELECT * FROM Bugs
JOIN Tags AS t1 USING (bug_id)
JOIN Tags AS t2 USING (bug_id)
WHERE t1.tag = 'printing' AND t2.tag = 'performance';

-- Add/remove
INSERT INTO Tags (bug_id, tag) VALUES (1234, 'save');
DELETE FROM Tags WHERE bug_id = 1234 AND tag = 'crash';
```

> **Note:** Store each value with the same meaning in a single column.

---

## Chapter 9 — Metadata Tribbles (Cloning Tables/Columns)

### ❌ Antipattern — Spawning Tables

```sql
CREATE TABLE Bugs_2008 ( . . . );
CREATE TABLE Bugs_2009 ( . . . );
CREATE TABLE Bugs_2010 ( . . . );

-- Querying across tables requires UNION
SELECT b.status, COUNT(*) AS count_per_status FROM (
  SELECT * FROM Bugs_2008
  UNION SELECT * FROM Bugs_2009
  UNION SELECT * FROM Bugs_2010
) AS b GROUP BY b.status;

-- Referential integrity is broken
CREATE TABLE Comments (
  bug_id BIGINT UNSIGNED NOT NULL,
  FOREIGN KEY (bug_id) REFERENCES Bugs_????(bug_id) -- impossible!
);
```

### ❌ Antipattern — Spawning Columns

```sql
CREATE TABLE ProjectHistory (
  bugs_fixed_2008 INT,
  bugs_fixed_2009 INT,
  bugs_fixed_2010 INT
);
```

### ✅ Solution A — Horizontal Partitioning

```sql
CREATE TABLE Bugs (
  bug_id SERIAL PRIMARY KEY,
  date_reported DATE
) PARTITION BY HASH (YEAR(date_reported)) PARTITIONS 4;
```

### ✅ Solution B — Vertical Partitioning

```sql
CREATE TABLE ProductInstallers (
  product_id BIGINT UNSIGNED PRIMARY KEY,
  installer_image BLOB,
  FOREIGN KEY (product_id) REFERENCES Products(product_id)
);
```

### ✅ Solution C — Dependent Table for Columns

```sql
CREATE TABLE ProjectHistory (
  project_id BIGINT,
  year SMALLINT,
  bugs_fixed INT,
  PRIMARY KEY (project_id, year),
  FOREIGN KEY (project_id) REFERENCES Projects(project_id)
);
```

> **Note:** Don't let data spawn metadata.

---

# PART II — Physical Database Design Antipatterns

---

## Chapter 10 — Rounding Errors (FLOAT Data Type)

### ❌ Antipattern

```sql
ALTER TABLE Bugs ADD COLUMN hours FLOAT;
ALTER TABLE Accounts ADD COLUMN hourly_rate FLOAT;

-- Seemingly fine
SELECT hourly_rate FROM Accounts WHERE account_id = 123;
-- Returns: 59.95, but actual stored value is 59.950000762939...

-- Equality comparison fails silently
SELECT * FROM Accounts WHERE hourly_rate = 59.95; -- empty set!

-- Workaround: threshold comparison (fragile)
SELECT * FROM Accounts WHERE ABS(hourly_rate - 59.95) < 0.000001;

-- Cumulative errors in aggregates
SELECT SUM(b.hours * a.hourly_rate) AS project_cost
FROM Bugs AS b JOIN Accounts AS a ON (b.assigned_to = a.account_id);
```

### ✅ Solution — NUMERIC Data Type

```sql
ALTER TABLE Bugs ADD COLUMN hours NUMERIC(9,2);
ALTER TABLE Accounts ADD COLUMN hourly_rate NUMERIC(9,2);

-- Exact equality works
SELECT hourly_rate FROM Accounts WHERE hourly_rate = 59.95; -- Returns: 59.95
SELECT hourly_rate * 1000000000 FROM Accounts WHERE hourly_rate = 59.95;
-- Returns: 59950000000 (exact)
```

> **Note:** Do not use FLOAT if you can avoid it.

---

## Chapter 11 — 31 Flavors (Column Definition Restrictions)

### ❌ Antipattern

```sql
-- Using CHECK constraint
CREATE TABLE Bugs (
  status VARCHAR(20) CHECK (status IN ('NEW', 'IN PROGRESS', 'FIXED'))
);

-- Using ENUM (MySQL-specific)
CREATE TABLE Bugs (
  status ENUM('NEW', 'IN PROGRESS', 'FIXED')
);

-- Querying allowed values is painful
SELECT column_type
FROM information_schema.columns
WHERE table_schema = 'bugtracker_schema'
  AND table_name = 'bugs' AND column_name = 'status';

-- Adding a value requires ALTER TABLE
ALTER TABLE Bugs MODIFY COLUMN status
  ENUM('NEW', 'IN PROGRESS', 'FIXED', 'DUPLICATE');
```

### ✅ Solution — Lookup Table

```sql
CREATE TABLE BugStatus (status VARCHAR(20) PRIMARY KEY);
INSERT INTO BugStatus (status) VALUES ('NEW'), ('IN PROGRESS'), ('FIXED');

CREATE TABLE Bugs (
  status VARCHAR(20),
  FOREIGN KEY (status) REFERENCES BugStatus(status) ON UPDATE CASCADE
);

-- Query allowed values with SELECT
SELECT status FROM BugStatus ORDER BY status;

-- Add value with INSERT (no downtime!)
INSERT INTO BugStatus (status) VALUES ('DUPLICATE');

-- Rename value with UPDATE
UPDATE BugStatus SET status = 'INVALID' WHERE status = 'BOGUS';

-- Support obsolete values
ALTER TABLE BugStatus ADD COLUMN active ENUM('INACTIVE','ACTIVE') NOT NULL DEFAULT 'ACTIVE';
UPDATE BugStatus SET active = 'INACTIVE' WHERE status = 'DUPLICATE';
SELECT status FROM BugStatus WHERE active = 'ACTIVE';
```

> **Note:** Use metadata when validating against a fixed set. Use data when validating against a fluid set.

---

## Chapter 12 — Phantom Files (External Image Storage)

### ❌ Antipattern

```sql
CREATE TABLE Screenshots (
  bug_id BIGINT UNSIGNED NOT NULL,
  image_id BIGINT UNSIGNED NOT NULL,
  screenshot_path VARCHAR(100),  -- external file path
  caption VARCHAR(100),
  PRIMARY KEY (bug_id, image_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

-- Deleting row does NOT delete the file
DELETE FROM Screenshots WHERE bug_id = 1234 AND image_id = 1;
-- File still exists on disk!
```

### ✅ Solution — BLOB Storage

```sql
CREATE TABLE Screenshots (
  bug_id BIGINT UNSIGNED NOT NULL,
  image_id BIGINT UNSIGNED NOT NULL,
  screenshot_image BLOB,
  caption VARCHAR(100),
  PRIMARY KEY (bug_id, image_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

-- Load file into BLOB (MySQL)
UPDATE Screenshots
SET screenshot_image = LOAD_FILE('images/screenshot1234-1.jpg')
WHERE bug_id = 1234 AND image_id = 1;

-- Export BLOB to file (MySQL)
SELECT screenshot_image INTO DUMPFILE 'images/screenshot1234-1.jpg'
FROM Screenshots WHERE bug_id = 1234 AND image_id = 1;
```

> **Note:** Resources outside the database are not managed by the database.

---

## Chapter 13 — Index Shotgun (Poor Indexing)

### ❌ Antipattern — Useless indexes

```sql
CREATE TABLE Bugs (
  bug_id SERIAL PRIMARY KEY,
  date_reported DATE NOT NULL,
  summary VARCHAR(80) NOT NULL,
  status VARCHAR(10) NOT NULL,
  hours NUMERIC(9,2),
  INDEX (bug_id),                            -- redundant (already PK)
  INDEX (summary),                           -- unlikely to be searched
  INDEX (hours),                             -- unlikely to be searched
  INDEX (bug_id, date_reported, status)      -- compound order matters
);

-- Queries that can't use indexes:
SELECT * FROM Accounts ORDER BY first_name, last_name;  -- wrong order
SELECT * FROM Bugs WHERE MONTH(date_reported) = 4;      -- function on column
SELECT * FROM Bugs WHERE description LIKE '%crash%';    -- leading wildcard
```

### ✅ MENTOR Your Indexes

```sql
-- M: Measure — enable slow query log in MySQL
-- long_query_time = 10  (config)

-- E: Explain — get query execution plan
EXPLAIN SELECT Bugs.*
FROM Bugs
JOIN (BugsProducts JOIN Products USING (product_id)) USING (bug_id)
WHERE summary LIKE '%crash%'
  AND product_name = 'Open RoundFile'
ORDER BY date_reported DESC;

-- N: Nominate — create missing index
CREATE INDEX ON Products(product_name);

-- Covering index example
CREATE INDEX BugCovering ON Bugs (status, bug_id, date_reported, reported_by, summary);
SELECT status, bug_id, date_reported, summary FROM Bugs WHERE status = 'OPEN';

-- Selectivity check
SELECT COUNT(DISTINCT status) / COUNT(status) AS selectivity FROM Bugs;

-- R: Rebuild — maintain indexes
ANALYZE TABLE Bugs;     -- MySQL
OPTIMIZE TABLE Bugs;    -- MySQL
VACUUM ANALYZE;         -- PostgreSQL
```

> **Note:** Know your data, know your queries, and MENTOR your indexes.

---

# PART III — Query Antipatterns

---

## Chapter 14 — Fear of the Unknown (NULL handling)

### ❌ Antipattern

```sql
-- Arithmetic with NULL returns NULL
SELECT hours + 10 FROM Bugs;  -- NULL if hours is NULL

-- Equality to NULL always returns UNKNOWN (not TRUE)
SELECT * FROM Bugs WHERE assigned_to = NULL;   -- returns nothing!
SELECT * FROM Bugs WHERE assigned_to <> NULL;  -- returns nothing!

-- Substituting special values for NULL causes problems
INSERT INTO Bugs (assigned_to, hours) VALUES (-1, -1);
SELECT AVG(hours) FROM Bugs WHERE hours <> -1;  -- must filter manually
```

### NULL Truth Table

| Expression | Expected | Actual |
|---|---|---|
| NULL = 0 | TRUE | NULL |
| NULL + 12345 | 12345 | NULL |
| NULL \|\| 'string' | 'string' | NULL |
| NULL = NULL | TRUE | NULL |
| NULL AND TRUE | FALSE | NULL |
| NULL AND FALSE | FALSE | **FALSE** |
| NULL OR TRUE | FALSE | **TRUE** |
| NOT (NULL) | TRUE | NULL |

### ✅ Solution

```sql
-- IS NULL / IS NOT NULL
SELECT * FROM Bugs WHERE assigned_to IS NULL;
SELECT * FROM Bugs WHERE assigned_to IS NOT NULL;

-- IS DISTINCT FROM (SQL-99)
SELECT * FROM Bugs WHERE assigned_to IS DISTINCT FROM 1;
-- Equivalent to:
SELECT * FROM Bugs WHERE assigned_to IS NULL OR assigned_to <> 1;

-- COALESCE for dynamic defaults
SELECT first_name || COALESCE(' ' || middle_initial || ' ', ' ') || last_name AS full_name
FROM Accounts;

-- Declare NOT NULL for mandatory fields
-- (example): date_reported DATE NOT NULL
```

> **Note:** Use null to signify a missing value for any data type.

---

## Chapter 15 — Ambiguous Groups (GROUP BY Issues)

### ❌ Antipattern

```sql
-- Violates Single-Value Rule — bug_id is ambiguous
SELECT product_id, MAX(date_reported) AS latest, bug_id
FROM Bugs JOIN BugsProducts USING (bug_id)
GROUP BY product_id;
```

### ✅ Solutions

```sql
-- 1. Only query functionally dependent columns
SELECT product_id, MAX(date_reported) AS latest
FROM Bugs JOIN BugsProducts USING (bug_id)
GROUP BY product_id;

-- 2. Correlated subquery
SELECT bp1.product_id, b1.date_reported AS latest, b1.bug_id
FROM Bugs b1 JOIN BugsProducts bp1 USING (bug_id)
WHERE NOT EXISTS (
  SELECT * FROM Bugs b2 JOIN BugsProducts bp2 USING (bug_id)
  WHERE bp1.product_id = bp2.product_id AND b1.date_reported < b2.date_reported
);

-- 3. Derived table
SELECT m.product_id, m.latest, b1.bug_id
FROM Bugs b1 JOIN BugsProducts bp1 USING (bug_id)
JOIN (
  SELECT bp2.product_id, MAX(b2.date_reported) AS latest
  FROM Bugs b2 JOIN BugsProducts bp2 USING (bug_id)
  GROUP BY bp2.product_id
) m ON (bp1.product_id = m.product_id AND b1.date_reported = m.latest);

-- 4. Outer JOIN
SELECT bp1.product_id, b1.date_reported AS latest, b1.bug_id
FROM Bugs b1 JOIN BugsProducts bp1 ON (b1.bug_id = bp1.bug_id)
LEFT OUTER JOIN (Bugs AS b2 JOIN BugsProducts AS bp2 ON b2.bug_id = bp2.bug_id)
  ON (bp1.product_id = bp2.product_id
      AND (b1.date_reported < b2.date_reported
      OR b1.date_reported = b2.date_reported AND b1.bug_id < b2.bug_id))
WHERE b2.bug_id IS NULL;

-- 5. Aggregate on extra column
SELECT product_id, MAX(date_reported) AS latest, MAX(bug_id) AS latest_bug_id
FROM Bugs JOIN BugsProducts USING (bug_id)
GROUP BY product_id;

-- 6. GROUP_CONCAT (MySQL/SQLite)
SELECT product_id, MAX(date_reported) AS latest,
  GROUP_CONCAT(bug_id) AS bug_id_list
FROM Bugs JOIN BugsProducts USING (bug_id)
GROUP BY product_id;
```

> **Note:** Follow the Single-Value Rule to avoid ambiguous query results.

---

## Chapter 16 — Random Selection

### ❌ Antipattern — Random sort (doesn't scale)

```sql
SELECT * FROM Bugs ORDER BY RAND() LIMIT 1;
```

### ✅ Solutions

```sql
-- 1. Random key between 1 and MAX (no gaps)
SELECT b1.*
FROM Bugs AS b1
JOIN (SELECT CEIL(RAND() * (SELECT MAX(bug_id) FROM Bugs)) AS rand_id) AS b2
  ON (b1.bug_id = b2.rand_id);

-- 2. Next higher key value (handles gaps)
SELECT b1.*
FROM Bugs AS b1
JOIN (SELECT CEIL(RAND() * (SELECT MAX(bug_id) FROM Bugs)) AS bug_id) AS b2
WHERE b1.bug_id >= b2.bug_id
ORDER BY b1.bug_id LIMIT 1;

-- 3. Random offset (PHP + SQL)
-- $rand = "SELECT ROUND(RAND() * (SELECT COUNT(*) FROM Bugs))";
-- $offset = $pdo->query($rand)->fetch();
-- SELECT * FROM Bugs LIMIT 1 OFFSET :offset

-- 4. Microsoft SQL Server
SELECT * FROM Bugs TABLESAMPLE (1 ROWS);

-- 5. Oracle
SELECT * FROM (SELECT * FROM Bugs SAMPLE(1) ORDER BY dbms_random.value)
WHERE ROWNUM = 1;
```

> **Note:** Some queries cannot be optimized; take a different approach.

---

## Chapter 17 — Poor Man's Search Engine

### ❌ Antipattern — Pattern matching

```sql
-- LIKE with wildcard
SELECT * FROM Bugs WHERE description LIKE '%crash%';

-- Regex (MySQL)
SELECT * FROM Bugs WHERE description REGEXP 'crash';

-- False matches (matches "money", "prone", etc.)
SELECT * FROM Bugs WHERE description LIKE '%one%';

-- Word boundary regex
SELECT * FROM Bugs WHERE description REGEXP '[[:<:]]one[[:>:]]';
```

### ✅ Solutions — Vendor Full-Text Search

```sql
-- MySQL full-text index
ALTER TABLE Bugs ADD FULLTEXT INDEX bugfts (summary, description);
SELECT * FROM Bugs WHERE MATCH(summary, description) AGAINST ('crash');
SELECT * FROM Bugs WHERE MATCH(summary, description) AGAINST ('+crash -save' IN BOOLEAN MODE);

-- Oracle CONTEXT index
CREATE INDEX BugsText ON Bugs(summary) INDEXTYPE IS CTSSYS.CONTEXT;
SELECT * FROM Bugs WHERE CONTAINS(summary, 'crash') > 0;

-- PostgreSQL
ALTER TABLE Bugs ADD COLUMN ts_bugtext TSVECTOR;
CREATE TRIGGER ts_bugtext BEFORE INSERT OR UPDATE ON Bugs
  FOR EACH ROW EXECUTE PROCEDURE
  tsvector_update_trigger(ts_bugtext, 'pg_catalog.english', summary, description);
CREATE INDEX bugs_ts ON Bugs USING GIN(ts_bugtext);
SELECT * FROM Bugs WHERE ts_bugtext @@ to_tsquery('crash');

-- SQLite FTS virtual table
CREATE VIRTUAL TABLE BugsText USING fts3(summary, description);
INSERT INTO BugsText (docid, summary, description)
  SELECT bug_id, summary, description FROM Bugs;
SELECT b.* FROM BugsText t JOIN Bugs b ON (t.docid = b.bug_id)
WHERE BugsText MATCH 'crash';
```

### ✅ Roll-Your-Own Inverted Index

```sql
CREATE TABLE Keywords (
  keyword_id SERIAL PRIMARY KEY,
  keyword VARCHAR(40) NOT NULL,
  UNIQUE KEY (keyword)
);

CREATE TABLE BugsKeywords (
  keyword_id BIGINT UNSIGNED NOT NULL,
  bug_id BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (keyword_id, bug_id),
  FOREIGN KEY (keyword_id) REFERENCES Keywords(keyword_id),
  FOREIGN KEY (bug_id) REFERENCES Bugs(bug_id)
);

-- Stored procedure for cached search (MySQL)
CREATE PROCEDURE BugsSearch(keyword VARCHAR(40))
BEGIN
  SET @keyword = keyword;
  PREPARE s1 FROM 'SELECT MAX(keyword_id) INTO @k FROM Keywords WHERE keyword = ?';
  EXECUTE s1 USING @keyword; DEALLOCATE PREPARE s1;
  IF (@k IS NULL) THEN
    PREPARE s2 FROM 'INSERT INTO Keywords (keyword) VALUES (?)';
    EXECUTE s2 USING @keyword; DEALLOCATE PREPARE s2;
    SELECT LAST_INSERT_ID() INTO @k;
    PREPARE s3 FROM 'INSERT INTO BugsKeywords (bug_id, keyword_id)
      SELECT bug_id, ? FROM Bugs
      WHERE summary REGEXP CONCAT(''[[:<:]]'', ?, ''[[:>:]]'')
         OR description REGEXP CONCAT(''[[:<:]]'', ?, ''[[:>:]]'')';
    EXECUTE s3 USING @k, @keyword, @keyword; DEALLOCATE PREPARE s3;
  END IF;
  PREPARE s4 FROM 'SELECT b.* FROM Bugs b
    JOIN BugsKeywords k USING (bug_id) WHERE k.keyword_id = ?';
  EXECUTE s4 USING @k; DEALLOCATE PREPARE s4;
END;

CALL BugsSearch('crash');

-- Trigger to index new bugs
CREATE TRIGGER Bugs_Insert AFTER INSERT ON Bugs
FOR EACH ROW BEGIN
  INSERT INTO BugsKeywords (bug_id, keyword_id)
  SELECT NEW.bug_id, k.keyword_id FROM Keywords k
  WHERE NEW.description REGEXP CONCAT('[[:<:]]', k.keyword, '[[:>:]]')
     OR NEW.summary REGEXP CONCAT('[[:<:]]', k.keyword, '[[:>:]]');
END;
```

> **Note:** You don't have to use SQL to solve every problem.

---

## Chapter 18 — Spaghetti Query

### ❌ Antipattern — One monstrous query

```sql
-- Produces Cartesian product (wrong results)
SELECT p.product_id,
  COUNT(f.bug_id) AS count_fixed,
  COUNT(o.bug_id) AS count_open
FROM BugsProducts p
LEFT OUTER JOIN Bugs f ON (p.bug_id = f.bug_id AND f.status = 'FIXED')
LEFT OUTER JOIN Bugs o ON (p.bug_id = o.bug_id AND o.status = 'OPEN')
WHERE p.product_id = 1
GROUP BY p.product_id;
-- Result: 84 for both (12 * 7 = 84 — wrong!)
```

### ✅ Solution — Split into multiple queries

```sql
-- Query 1
SELECT p.product_id, COUNT(f.bug_id) AS count_fixed
FROM BugsProducts p
LEFT OUTER JOIN Bugs f ON (p.bug_id = f.bug_id AND f.status = 'FIXED')
WHERE p.product_id = 1 GROUP BY p.product_id;

-- Query 2
SELECT p.product_id, COUNT(o.bug_id) AS count_open
FROM BugsProducts p
LEFT OUTER JOIN Bugs o ON (p.bug_id = o.bug_id AND o.status = 'OPEN')
WHERE p.product_id = 1 GROUP BY p.product_id;

-- Or UNION them
(SELECT p.product_id, f.status, COUNT(f.bug_id) AS bug_count
 FROM BugsProducts p
 LEFT OUTER JOIN Bugs f ON (p.bug_id = f.bug_id AND f.status = 'FIXED')
 WHERE p.product_id = 1 GROUP BY p.product_id, f.status)
UNION ALL
(SELECT p.product_id, o.status, COUNT(o.bug_id) AS bug_count
 FROM BugsProducts p
 LEFT OUTER JOIN Bugs o ON (p.bug_id = o.bug_id AND o.status = 'OPEN')
 WHERE p.product_id = 1 GROUP BY p.product_id, o.status)
ORDER BY bug_count;
```

### Code-generating SQL

```sql
-- Generate UPDATE statements automatically
SELECT CONCAT('UPDATE Inventory SET last_used = ''', MAX(u.usage_date), ''',',
              ' WHERE inventory_id = ', u.inventory_id, ';') AS update_statement
FROM ComputerUsage u
GROUP BY u.inventory_id;
```

> **Note:** Although SQL makes it seem possible to solve a complex problem in one line, don't build a house of cards.

---

## Chapter 19 — Implicit Columns (Wildcards)

### ❌ Antipattern

```sql
SELECT * FROM Bugs;  -- all columns, order unknown
INSERT INTO Accounts VALUES (DEFAULT, 'bkarwin', 'Bill', 'Karwin', 'bill@example.com', SHA2('xyzzy'), NULL, 49.95);

-- After adding a column, INSERT breaks!
ALTER TABLE Bugs ADD COLUMN date_due DATE;
INSERT INTO Bugs VALUES (DEFAULT, CURDATE(), 'New bug', 'Test T987 fails...',
  NULL, 123, NULL, NULL, DEFAULT, 'Medium', NULL);
-- ERROR: Column count doesn't match value count
```

### ✅ Solution — Name Columns Explicitly

```sql
SELECT bug_id, date_reported, summary, description, resolution,
       reported_by, assigned_to, verified_by, status, priority, hours
FROM Bugs;

INSERT INTO Accounts (account_name, first_name, last_name, email,
                      password_hash, portrait_image, hourly_rate)
VALUES ('bkarwin', 'Bill', 'Karwin', 'bill@example.com',
        SHA2('xyzzy'), NULL, 49.95);

-- Wildcard for one table only (acceptable)
SELECT b.*, a.first_name, a.email
FROM Bugs b JOIN Accounts a ON (b.reported_by = a.account_id);

-- Only select what you need
SELECT date_reported, summary, description, resolution, status, priority FROM Bugs;
```

> **Note:** Take all you want, but eat all you take.

---

# PART IV — Application Development Antipatterns

---

## Chapter 20 — Readable Passwords

### ❌ Antipattern

```sql
CREATE TABLE Accounts (
  account_id SERIAL PRIMARY KEY,
  account_name VARCHAR(20),
  email VARCHAR(100),
  password VARCHAR(30)  -- plain text!
);

INSERT INTO Accounts (account_id, account_name, email, password)
VALUES (123, 'billkarwin', 'bill@example.com', 'xyzzy');

-- Authentication sends password in plain text
SELECT CASE WHEN password = 'opensesame' THEN 1 ELSE 0 END AS password_matches
FROM Accounts WHERE account_id = 123;
```

### ✅ Solution — Salted Hash

```sql
CREATE TABLE Accounts (
  account_id SERIAL PRIMARY KEY,
  account_name VARCHAR(20),
  email VARCHAR(100) NOT NULL,
  password_hash CHAR(64) NOT NULL,
  salt BINARY(8) NOT NULL
);

-- Store hashed password
INSERT INTO Accounts (account_id, account_name, email, password_hash, salt)
VALUES (123, 'billkarwin', 'bill@example.com',
        SHA2('xyzzy' || '-0xT!sp9'), '-0xT!sp9');

-- Authenticate
SELECT (password_hash = SHA2('xyzzy' || salt)) AS password_matches
FROM Accounts WHERE account_id = 123;

-- Lock account
UPDATE Accounts SET password_hash = 'noaccess' WHERE account_id = 123;

-- Password reset token table
CREATE TABLE PasswordResetRequest (
  token CHAR(32) PRIMARY KEY,
  account_id BIGINT UNSIGNED NOT NULL,
  expiration TIMESTAMP NOT NULL,
  FOREIGN KEY (account_id) REFERENCES Accounts(account_id)
);

SET @token = MD5('billkarwin' || CURRENT_TIMESTAMP);
INSERT INTO PasswordResetRequest (token, account_id, expiration)
VALUES (@token, 123, CURRENT_TIMESTAMP + INTERVAL 1 HOUR);
```

> **Note:** If you can read passwords, so can a hacker.

---

## Chapter 21 — SQL Injection

### ❌ Antipattern

```php
// Vulnerable: user input interpolated directly
$sql = "SELECT * FROM Bugs WHERE bug_id = $bug_id";

// Password reset attack: userid = "123 OR TRUE"
// Results in: UPDATE Accounts SET password_hash = SHA2('xyzzy') WHERE account_id = 123 OR TRUE;
```

### ✅ Solutions

```php
// 1. Filter input
$bugid = filter_input(INPUT_GET, "bugid", FILTER_SANITIZE_NUMBER_INT);
$bugid = intval($_GET["bugid"]);

// 2. Use query parameters
$stmt = $pdo->prepare("UPDATE Accounts SET password_hash = SHA2(?) WHERE account_id = ?");
$stmt->execute(array($_REQUEST["password"], $_REQUEST["userid"]));

// 3. Quote dynamic values
$quoted_active = $pdo->quote($_REQUEST["active"]);
$sql = "SELECT * FROM Accounts WHERE is_active = {$quoted_active}";

// 4. Map user choices to predefined SQL values
$sortorders = array("status" => "status", "date" => "date_reported");
$directions = array("up" => "ASC", "down" => "DESC");
$sortorder = "bug_id"; $direction = "ASC";
if (array_key_exists($_REQUEST["order"], $sortorders)) {
    $sortorder = $sortorders[$_REQUEST["order"]];
}
if (array_key_exists($_REQUEST["dir"], $directions)) {
    $direction = $directions[$_REQUEST["dir"]];
}
$sql = "SELECT * FROM Bugs ORDER BY {$sortorder} {$direction}";

// 5. Parameterize IN() predicate
$sql = "SELECT * FROM Bugs WHERE bug_id IN ("
     . join(",", array_fill(0, count($bug_list), "?")) . ")";
$stmt = $pdo->prepare($sql);
$stmt->execute($bug_list);
```

> **Note:** Let users input values, but never let users input code.

---

## Chapter 22 — Pseudokey Neat-Freak (Renumbering IDs)

### ❌ Antipattern

```sql
-- Finding lowest unused ID (inefficient + race-prone)
SELECT b1.bug_id + 1
FROM Bugs b1
LEFT OUTER JOIN Bugs AS b2 ON (b1.bug_id + 1 = b2.bug_id)
WHERE b2.bug_id IS NULL
ORDER BY b1.bug_id LIMIT 1;

-- Renumbering existing rows (dangerous)
UPDATE Bugs SET bug_id = 3 WHERE bug_id = 4;
```

### ✅ Solution — Use ROW_NUMBER() for Pagination

```sql
-- SQL:2003 window function for row numbers (not PKs)
SELECT t1.* FROM (
  SELECT a.account_name, b.bug_id, b.summary,
    ROW_NUMBER() OVER (ORDER BY a.account_name, b.date_reported) AS rn
  FROM Accounts a JOIN Bugs b ON (a.account_id = b.reported_by)
) AS t1
WHERE t1.rn BETWEEN 51 AND 100;
```

### ✅ GUIDs as Primary Keys

```sql
-- Microsoft SQL Server
CREATE TABLE Bugs (
  bug_id UNIQUEIDENTIFIER DEFAULT NEWID(),
  ...
);
INSERT INTO Bugs (bug_id, summary) VALUES (DEFAULT, 'crashes when I save');
```

> **Note:** Use pseudokeys as unique row identifiers; they're not row numbers.

---

## Chapter 23 — See No Evil (Ignoring Error Returns)

### ❌ Antipattern

```php
// No error checking at any step
$pdo = new PDO("mysql:dbname=test;host=db.example.com", "dbuser", "dbpassword");
$stmt = $dbh->prepare($sql);   // could return false
$stmt->execute(array(1, "OPEN")); // could fail
$bug = $stmt->fetch();         // could return false

// Whitespace bug — hard to spot by reading code
$sql = "SELECT * FROM Bugs";
if ($bug_id) {
    $sql .= "WHERE bug_id = " . intval($bug_id);
    // Results in: SELECT * FROM BugsWHERE bug_id = 1234
}
```

### ✅ Solution — Check Every Return Value

```php
try {
    $pdo = new PDO("mysql:dbname=test;host=localhost", "dbuser", "dbpassword");
} catch (PDOException $e) {
    report_error($e->getMessage()); return;
}

if (($stmt = $pdo->prepare($sql)) === false) {
    $error = $pdo->errorInfo();
    report_error($error[2]); return;
}

if ($stmt->execute(array(1, "OPEN")) === false) {
    $error = $stmt->errorInfo();
    report_error($error[2]); return;
}

if (($bug = $stmt->fetch()) === false) {
    $error = $stmt->errorInfo();
    report_error($error[2]); return;
}
```

> **Note:** Troubleshooting code is already hard enough. Don't hinder yourself by doing it blind.

---

## Chapter 24 — Diplomatic Immunity (Skipping Best Practices)

### ✅ Database Unit Testing (PHPUnit)

```php
class DatabaseTest extends PHPUnit_Framework_TestCase {
    protected $pdo;

    public function setUp() {
        $this->pdo = new PDO("mysql:dbname=bugs", "testuser", "xxxxxx");
    }

    public function testTableFooExists() {
        $stmt = $this->pdo->query("SELECT COUNT(*) FROM Bugs");
        $this->assertType("object", $stmt);
        $this->assertEquals("PDOStatement", get_class($stmt));
    }

    public function testTableFooColumnBugIdExists() {
        $stmt = $this->pdo->query("SELECT COUNT(bug_id) FROM Bugs");
        $this->assertType("object", $stmt);
    }
}
```

### ✅ Ruby on Rails Migrations (schema evolution)

```ruby
class AddHoursToBugs < ActiveRecord::Migration
  def self.up
    add_column :bugs, :hours, :decimal
  end
  def self.down
    remove_column :bugs, :hours
  end
end
# Run with: rake db:migrate VERSION=5
```

> **Note:** Use software development best practices — documentation, testing, and source control — for your database as well as your application.

---

## Chapter 25 — Magic Beans (Active Record as Model)

### ❌ Antipattern — Controllers doing DB work directly

```php
class AdminController extends Zend_Controller_Action {
    public function assignAction() {
        $bugsTable = Doctrine_Core::getTable("Bugs");
        $bug = $bugsTable->find($_POST["bug_id"]);
        $bug->assigned_to = $_POST["user_assigned_to"];
        $bug->save();
        // Business logic + DB access tightly coupled in controller
    }
}
```

### ✅ Solution — Domain Model encapsulates DB logic

```php
class BugReport {
    protected $bugsTable, $accountsTable, $productsTable;

    public function __construct() {
        $this->bugsTable = Doctrine_Core::getTable("Bugs");
        $this->accountsTable = Doctrine_Core::getTable("Accounts");
        $this->productsTable = Doctrine_Core::getTable("Products");
    }

    public function create($summary, $description, $reportedBy) {
        $bug = new Bugs();
        $bug->summary = $summary;
        $bug->description = $description;
        $bug->status = "NEW";
        $bug->reported_by = $reportedBy;
        $bug->save();
    }

    public function assignUser($bugId, $assignedTo) {
        $bug = $this->bugsTable->find($bugId);
        $bug->assigned_to = $assignedTo;
        $bug->save();
    }

    public function search($status, $searchString) {
        $q = Doctrine_Query::create()
            ->from("Bugs b")
            ->join("b.Products p")
            ->where("b.status = ?", $status)
            ->andWhere("MATCH(b.summary, b.description) AGAINST (?)", $searchString);
        return $q->fetchArray();
    }
}

// Clean controller
class AdminController extends Zend_Controller_Action {
    public function assignAction() {
        $this->bugReport->assignUser(
            $this->_getParam("bug"),
            $this->_getParam("user")
        );
    }
}
```

> **Note:** Decouple your models from your tables.

---

# APPENDIX A — Rules of Normalization

## Relations Must Satisfy

- Rows have no order top to bottom
- Columns have no order left to right
- No duplicate rows
- Every column has one type, one value per row
- Rows have no hidden components (no physical row IDs)

## Normal Forms Summary

| Normal Form | Rule |
|---|---|
| **1NF** | No repeating groups; table must be a proper relation |
| **2NF** | No partial dependency on compound primary key |
| **3NF** | No transitive dependency on non-key column |
| **BCNF** | All determinants are candidate keys |
| **4NF** | No multi-valued facts in a single table |
| **5NF** | Decompose all join dependencies |
| **DKNF** | Every constraint follows from domain and key constraints |
| **6NF** | No join dependencies at all (for temporal/history data) |

### 2NF Example

```sql
-- Violates 2NF: coiner depends only on tag, not on (bug_id, tag)
CREATE TABLE BugsTags (
  bug_id BIGINT NOT NULL,
  tag VARCHAR(20) NOT NULL,
  tagger BIGINT NOT NULL,
  coiner BIGINT NOT NULL,  -- depends only on tag!
  PRIMARY KEY (bug_id, tag)
);

-- 2NF fix: separate tables
CREATE TABLE Tags (
  tag VARCHAR(20) PRIMARY KEY,
  coiner BIGINT NOT NULL
);
CREATE TABLE BugsTags (
  bug_id BIGINT NOT NULL,
  tag VARCHAR(20) NOT NULL,
  tagger BIGINT NOT NULL,
  PRIMARY KEY (bug_id, tag),
  FOREIGN KEY (tag) REFERENCES Tags(tag)
);
```

### 4NF Example

```sql
-- Violates 4NF: multiple independent multi-valued facts
CREATE TABLE BugsAccounts (
  bug_id BIGINT NOT NULL,
  reported_by BIGINT,
  assigned_to BIGINT,
  verified_by BIGINT
);

-- 4NF fix: one intersection table per relationship
CREATE TABLE BugsReported (bug_id BIGINT NOT NULL, reported_by BIGINT NOT NULL, PRIMARY KEY (bug_id, reported_by));
CREATE TABLE BugsAssigned (bug_id BIGINT NOT NULL, assigned_to BIGINT NOT NULL, PRIMARY KEY (bug_id, assigned_to));
CREATE TABLE BugsVerified (bug_id BIGINT NOT NULL, verified_by BIGINT NOT NULL, PRIMARY KEY (bug_id, verified_by));
```

---

# Quick Reference — Key Notes

| # | Antipattern | Key Takeaway |
|---|---|---|
| 2 | Jaywalking | Store each value in its own column and row |
| 3 | Naive Trees | Model both entries and relationships |
| 4 | ID Required | Conventions are good only if they are helpful |
| 5 | Keyless Entry | Make your database mistake-proof with constraints |
| 6 | EAV | Use metadata for metadata |
| 7 | Polymorphic Assoc. | Every relationship has one referencing and one referenced table |
| 8 | Multicolumn Attr. | Store each value with the same meaning in a single column |
| 9 | Metadata Tribbles | Don't let data spawn metadata |
| 10 | Rounding Errors | Do not use FLOAT if you can avoid it |
| 11 | 31 Flavors | Use data (lookup tables) when validating a fluid set of values |
| 12 | Phantom Files | Resources outside the database are not managed by the database |
| 13 | Index Shotgun | Know your data, know your queries, and MENTOR your indexes |
| 14 | Fear of Unknown | Use null to signify a missing value for any data type |
| 15 | Ambiguous Groups | Follow the Single-Value Rule |
| 16 | Random Selection | Some queries cannot be optimized; take a different approach |
| 17 | Poor Man's Search | You don't have to use SQL to solve every problem |
| 18 | Spaghetti Query | Don't build a house of cards with one complex query |
| 19 | Implicit Columns | Name all columns explicitly |
| 20 | Readable Passwords | If you can read passwords, so can a hacker |
| 21 | SQL Injection | Let users input values, but never let users input code |
| 22 | Pseudokey Neat-Freak | Pseudokeys are row identifiers, not row numbers |
| 23 | See No Evil | Don't debug blind — check return values |
| 24 | Diplomatic Immunity | Apply best practices to database code too |
| 25 | Magic Beans | Decouple your models from your tables |
