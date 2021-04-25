

-- on master database

CREATE LOGIN scu 
WITH PASSWORD = 'UaKiQpg%nZGDAyYkivUq6s59%#K5f^L9@@5RjNKpmkfu9cf!kM^SXxK' 


-- on COMP6004 database

CREATE USER scu FROM LOGIN scu
go
ALTER ROLE db_datawriter ADD MEMBER scu
go
ALTER ROLE db_datareader ADD MEMBER scu
go
ALTER ROLE db_ddladmin ADD MEMBER scu
go
GRANT EXECUTE TO scu
