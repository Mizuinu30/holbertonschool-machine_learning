-- Creates a table called first_table in MySQL server
-- If first_table already exists, script does not fail
-- firat_table has id INT and name VARCHAR(256) columns
CREATE TABLE IF NOT EXISTS first_table (id INT, name VARCHAR(256));
